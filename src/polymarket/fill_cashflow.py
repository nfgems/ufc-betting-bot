"""Observed entry cash cost, separate from exchange gross fill evidence.

No fee schedule or fee_rate_bps value is used to infer a charge. This bounded
first version accepts only a uniquely linked taker BUY activity per trade.
"""
from decimal import Decimal, InvalidOperation
import hashlib
import json


class CashDebitUnavailable(ValueError):
    """Original fill cash cost cannot be attributed without ambiguity."""


def number(value):
    if isinstance(value, bool) or value is None:
        raise CashDebitUnavailable("missing_or_invalid_number")
    try:
        result = Decimal(str(value))
    except (InvalidOperation, ValueError, TypeError) as exc:
        raise CashDebitUnavailable("missing_or_invalid_number") from exc
    if not result.is_finite() or result < 0:
        raise CashDebitUnavailable("missing_or_invalid_number")
    return result


def fill_binding(bet):
    """Bind enrichment to original gross evidence; safe to compare under lock."""
    return {key: bet.get(key) for key in (
        "order_id", "token_id", "actual_fill_amount", "actual_filled_shares",
        "actual_fill_price", "actual_fill_source", "actual_fill_trade_ids",
    )}


def observed_cash_debit(bet, order, activity):
    order_id, token = str(bet.get("order_id") or ""), str(bet.get("token_id") or "")
    components = order.get("actual_fill_trade_components") or []
    expected_ids = bet.get("actual_fill_trade_ids")
    if (not order_id or not token or not isinstance(expected_ids, list)
            or not expected_ids or any(not isinstance(value, str) or not value.strip()
                                      for value in expected_ids)
            or len(expected_ids) != len(set(expected_ids))
            or not isinstance(components, list) or not components
            or any(not isinstance(part, dict) or not isinstance(part.get("trade_id"), str)
                   or not part["trade_id"].strip() for part in components)
            or order.get("actual_fill_activity_conflicts")):
        raise CashDebitUnavailable("original_trade_identity_unavailable_or_shared_transaction")
    if set(expected_ids) != {part.get("trade_id") for part in components}:
        raise CashDebitUnavailable("original_trade_set_incomplete")
    if len(components) != len(expected_ids) or not isinstance(activity, list):
        raise CashDebitUnavailable("duplicate_or_incomplete_trade_evidence")
    matched, used, gross, shares, debit = [], set(), Decimal(0), Decimal(0), Decimal(0)
    for part in components:
        if (part.get("order_id") != order_id or part.get("token_id") != token
                or part.get("role") != "taker" or part.get("side") != "BUY"):
            raise CashDebitUnavailable("unsupported_or_mismatched_fill_role")
        tx = str(part.get("transaction_hash") or "").lower()
        if not tx or tx in used:
            raise CashDebitUnavailable("missing_or_shared_transaction")
        used.add(tx)
        quantity, price = number(part.get("shares")), number(part.get("price"))
        if quantity <= 0 or not 0 < price < 1:
            raise CashDebitUnavailable("invalid_fill_quantity_or_price")
        candidates = [row for row in activity if isinstance(row, dict)
                      and str(row.get("transactionHash") or "").lower() == tx
                      and str(row.get("asset") or "") == token
                      and row.get("type") == "TRADE" and row.get("side") == "BUY"]
        if len(candidates) != 1:
            raise CashDebitUnavailable("missing_or_duplicate_activity")
        row = candidates[0]
        if number(row.get("size")) != quantity or number(row.get("price")) != price:
            raise CashDebitUnavailable("activity_quantity_or_price_mismatch")
        cash = number(row.get("usdcSize"))
        consideration = quantity * price
        if cash < consideration:
            raise CashDebitUnavailable("activity_debit_below_gross")
        timestamp = row.get("timestamp")
        if type(timestamp) is not int or timestamp < 0:
            raise CashDebitUnavailable("activity_timestamp_missing")
        gross += consideration
        shares += quantity
        debit += cash
        matched.append({"trade_id": part["trade_id"], "transaction_hash": tx,
                        "token_id": token, "order_id": order_id, "side": "BUY",
                        "shares": str(quantity), "price": str(price),
                        "gross_consideration": str(consideration),
                        "cash_debit": str(cash), "timestamp": timestamp})
    # Original ledgers round filled quantity to two decimals. Keep that original
    # evidence untouched while retaining the exact uniquely linked trade sum.
    if abs(shares - number(bet.get("actual_filled_shares"))) > Decimal("0.005"):
        raise CashDebitUnavailable("original_filled_quantity_mismatch")
    # Existing gross ledger amounts were rounded to cents. Preserve them as-is.
    if abs(gross - number(bet.get("actual_fill_amount"))) > Decimal("0.005"):
        raise CashDebitUnavailable("original_gross_consideration_mismatch")
    evidence = {"schema_version": 1, "source": "clob_trade_and_polymarket_activity",
                "binding": fill_binding(bet),
                "fills": sorted(matched, key=lambda row: row["trade_id"])}
    fingerprint = hashlib.sha256(json.dumps(evidence, sort_keys=True,
                                            separators=(",", ":")).encode()).hexdigest()
    return {"cash_debit_status": "confirmed", "actual_cash_debit": str(debit),
            "actual_additional_charge": str(debit - gross),
            "actual_gross_consideration_exact": str(gross),
            "actual_filled_shares_exact": str(shares),
            "actual_cash_debit_evidence": evidence,
            "actual_cash_debit_evidence_sha256": fingerprint}


def verified_cash_debit(bet):
    """Validate saved provenance before accounting consumes a fee-inclusive cost."""
    evidence = bet.get("actual_cash_debit_evidence")
    if (bet.get("cash_debit_status") != "confirmed" or not isinstance(evidence, dict)
            or evidence.get("binding") != fill_binding(bet)):
        raise CashDebitUnavailable("cash_debit_not_confirmed_for_original_fill")
    digest = hashlib.sha256(json.dumps(evidence, sort_keys=True,
                                     separators=(",", ":")).encode()).hexdigest()
    if digest != bet.get("actual_cash_debit_evidence_sha256"):
        raise CashDebitUnavailable("cash_debit_provenance_conflict")
    debit, gross, charge = (number(bet.get(key)) for key in (
        "actual_cash_debit", "actual_gross_consideration_exact", "actual_additional_charge"))
    if debit <= 0 or debit != gross + charge:
        raise CashDebitUnavailable("cash_debit_totals_conflict")
    fills = evidence.get("fills")
    if (not isinstance(fills, list) or not fills or any(not isinstance(row, dict) for row in fills)
            or sum((number(row.get("cash_debit")) for row in fills), Decimal(0)) != debit
            or sum((number(row.get("gross_consideration")) for row in fills), Decimal(0)) != gross
            or sum((number(row.get("shares")) for row in fills), Decimal(0))
                != number(bet.get("actual_filled_shares_exact"))):
        raise CashDebitUnavailable("cash_debit_components_conflict")
    rebuilt = observed_cash_debit(bet, {"actual_fill_trade_components": [
        {"trade_id": row.get("trade_id"), "transaction_hash": row.get("transaction_hash"),
         "token_id": row.get("token_id"), "order_id": row.get("order_id"),
         "role": "taker", "side": row.get("side"), "shares": row.get("shares"),
         "price": row.get("price")} for row in fills]}, [
        {"transactionHash": row.get("transaction_hash"), "asset": row.get("token_id"),
         "type": "TRADE", "side": row.get("side"), "size": row.get("shares"),
         "price": row.get("price"), "usdcSize": row.get("cash_debit"),
         "timestamp": row.get("timestamp")} for row in fills])
    if rebuilt["actual_cash_debit_evidence_sha256"] != digest:
        raise CashDebitUnavailable("cash_debit_identity_or_arithmetic_conflict")
    return debit


def verified_fill_shares(bet):
    """Return exact payout quantity only after validating original-fill evidence."""
    verified_cash_debit(bet)
    return number(bet.get("actual_filled_shares_exact"))
