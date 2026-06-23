from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import BTC5M_PAPER_LEDGER_DIR, BTC5M_PAPER_PROFILES, BTC5M_SIGNAL_LOG_PATH
from src.polymarket.btc_5m import run_btc5m_paper_compare_once


def _money(value: object) -> str:
    return f"${float(value or 0.0):,.2f}"


def _pct(value: object) -> str:
    return f"{float(value or 0.0) * 100.0:.2f}%"


def _cell(value: object) -> str:
    return str(value if value is not None else "").replace("|", "/").replace("\n", " ")


def _num(value: object) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _avg(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _quantile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(max(int(round((len(ordered) - 1) * q)), 0), len(ordered) - 1)
    return ordered[index]


def _fmt_optional(value: object, *, places: int = 2, suffix: str = "") -> str:
    number = _num(value)
    if number is None:
        return ""
    return f"{number:.{places}f}{suffix}"


def _empty_profile_liquidity_stats() -> dict:
    return {
        "snapshots": 0,
        "orders": 0,
        "selected_spread": [],
        "selected_top_ask_notional": [],
        "selected_total_ask_notional": [],
        "selected_best_ask": [],
    }


def _book_liquidity_summary(
    signal_log_path: Path | None,
    *,
    expected_profiles: list[str] | None = None,
) -> dict:
    if signal_log_path is None or not signal_log_path.exists():
        return {"available": False, "path": str(signal_log_path) if signal_log_path else ""}

    profiles: dict[str, dict[str, list[float] | int]] = {}
    for profile in expected_profiles or []:
        profiles.setdefault(profile, _empty_profile_liquidity_stats())

    sides: dict[str, dict[str, list[float] | int]] = {
        "up": {"samples": 0, "spread": [], "top_ask_notional": [], "total_ask_notional": []},
        "down": {"samples": 0, "spread": [], "top_ask_notional": [], "total_ask_notional": []},
    }
    rows = 0

    with signal_log_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                snapshot = json.loads(line)
            except json.JSONDecodeError:
                continue

            rows += 1
            profile = str(snapshot.get("profile") or "unknown")
            profile_stats = profiles.setdefault(
                profile,
                _empty_profile_liquidity_stats(),
            )
            profile_stats["snapshots"] = int(profile_stats["snapshots"]) + 1

            books = snapshot.get("books") if isinstance(snapshot.get("books"), dict) else {}
            for side in ("up", "down"):
                book = books.get(side) if isinstance(books.get(side), dict) else {}
                side_stats = sides[side]
                side_stats["samples"] = int(side_stats["samples"]) + 1
                for key in ("spread", "top_ask_notional", "total_ask_notional"):
                    value = _num(book.get(key))
                    if value is not None:
                        side_stats[key].append(value)  # type: ignore[index]

            result = snapshot.get("result") if isinstance(snapshot.get("result"), dict) else {}
            for order in result.get("orders") or []:
                if not isinstance(order, dict) or order.get("status") != "dry_run":
                    continue
                direction = str(order.get("direction") or "").lower()
                book = books.get(direction) if isinstance(books.get(direction), dict) else {}
                if not book:
                    continue
                profile_stats["orders"] = int(profile_stats["orders"]) + 1
                for source_key, target_key in (
                    ("spread", "selected_spread"),
                    ("top_ask_notional", "selected_top_ask_notional"),
                    ("total_ask_notional", "selected_total_ask_notional"),
                    ("best_ask", "selected_best_ask"),
                ):
                    value = _num(book.get(source_key))
                    if value is not None:
                        profile_stats[target_key].append(value)  # type: ignore[index]

    def summarize_values(values: list[float]) -> dict:
        return {
            "avg": _avg(values),
            "p50": _quantile(values, 0.50),
            "p90": _quantile(values, 0.90),
            "min": min(values) if values else None,
            "max": max(values) if values else None,
        }

    profile_summary = {}
    for profile, stats in profiles.items():
        profile_summary[profile] = {
            "snapshots": int(stats["snapshots"]),
            "dry_run_orders": int(stats["orders"]),
            "selected_spread": summarize_values(stats["selected_spread"]),  # type: ignore[arg-type]
            "selected_top_ask_notional": summarize_values(stats["selected_top_ask_notional"]),  # type: ignore[arg-type]
            "selected_total_ask_notional": summarize_values(stats["selected_total_ask_notional"]),  # type: ignore[arg-type]
            "selected_best_ask": summarize_values(stats["selected_best_ask"]),  # type: ignore[arg-type]
        }

    side_summary = {}
    for side, stats in sides.items():
        side_summary[side] = {
            "samples": int(stats["samples"]),
            "spread": summarize_values(stats["spread"]),  # type: ignore[arg-type]
            "top_ask_notional": summarize_values(stats["top_ask_notional"]),  # type: ignore[arg-type]
            "total_ask_notional": summarize_values(stats["total_ask_notional"]),  # type: ignore[arg-type]
        }

    return {
        "available": True,
        "path": str(signal_log_path),
        "snapshot_rows": rows,
        "profiles": profile_summary,
        "missing_profile_snapshots": sorted(
            profile for profile, stats in profile_summary.items()
            if int(stats.get("snapshots") or 0) == 0
        ),
        "sides": side_summary,
    }


def build_markdown(result: dict, *, generated_at: datetime, json_path: Path) -> str:
    requested_profiles = result.get("requested_profiles") or list((result.get("profiles") or {}).keys())
    lines: list[str] = [
        "# BTC 5m Paper Trading Report",
        "",
        f"Generated: {generated_at.isoformat()}",
        f"Profiles: `{', '.join(str(profile) for profile in requested_profiles)}`",
        f"Settlement source: `{result.get('settlement_source')}`",
        f"Best profile: `{result.get('best_profile') or 'none yet'}`",
        "",
        (
            "Settlement uses Polymarket's official resolved outcome by default. "
            "Polymarket BTC 5m markets resolve from the Chainlink BTC/USD Data Stream."
        ),
        "",
        "## Ranking",
        "",
        "| Rank | Profile | Settled | PnL | ROI | Win rate |",
        "|---:|---|---:|---:|---:|---:|",
    ]

    for index, row in enumerate(result.get("ranking", []), 1):
        lines.append(
            f"| {index} | {_cell(row.get('profile'))} | {int(row.get('settled_bets') or 0)} | "
            f"{_money(row.get('realized_pnl'))} | {_pct(row.get('roi'))} | {_pct(row.get('win_rate'))} |"
        )

    lines.extend(
        [
            "",
            "## Profile Detail",
            "",
            (
                "| Profile | Total | Open | Settled | Wins | Losses | Wagered | PnL | ROI | "
                "Last status | Last reason | Settled this pass | Settlement errors |"
            ),
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---:|---|",
        ]
    )

    for profile, payload in result.get("profiles", {}).items():
        summary = payload.get("summary", {})
        settlement = payload.get("settlement", {})
        errors = settlement.get("errors") or []
        lines.append(
            f"| {_cell(profile)} | {int(summary.get('total_bets') or 0)} | "
            f"{int(summary.get('open_bets') or 0)} | {int(summary.get('settled_bets') or 0)} | "
            f"{int(summary.get('wins') or 0)} | {int(summary.get('losses') or 0)} | "
            f"{_money(summary.get('total_wagered'))} | {_money(summary.get('realized_pnl'))} | "
            f"{_pct(summary.get('roi'))} | {_cell(payload.get('last_status'))} | "
            f"{_cell(payload.get('last_reason'))} | {int(settlement.get('settled') or 0)} | "
            f"{_cell('; '.join(errors))} |"
        )

    liquidity = result.get("liquidity") if isinstance(result.get("liquidity"), dict) else {}
    if liquidity.get("available"):
        lines.extend(
            [
                "",
                "## Spread/Liquidity",
                "",
                f"Signal snapshots: `{int(liquidity.get('snapshot_rows') or 0)}`",
                f"Signal log: `{liquidity.get('path')}`",
                (
                    "Profiles with no signal snapshots: "
                    f"`{', '.join(liquidity.get('missing_profile_snapshots') or [])}`"
                    if liquidity.get("missing_profile_snapshots")
                    else "Profiles with no signal snapshots: `none`"
                ),
                "",
                "| Profile | Snapshots | Paper orders | Avg selected spread | P90 selected spread | Min top ask $ | P50 total ask $ |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for profile, stats in sorted((liquidity.get("profiles") or {}).items()):
            selected_spread = stats.get("selected_spread") or {}
            top_ask = stats.get("selected_top_ask_notional") or {}
            total_ask = stats.get("selected_total_ask_notional") or {}
            lines.append(
                f"| {_cell(profile)} | {int(stats.get('snapshots') or 0)} | "
                f"{int(stats.get('dry_run_orders') or 0)} | "
                f"{_fmt_optional(selected_spread.get('avg'), places=4)} | "
                f"{_fmt_optional(selected_spread.get('p90'), places=4)} | "
                f"{_money(top_ask.get('min')) if top_ask.get('min') is not None else ''} | "
                f"{_money(total_ask.get('p50')) if total_ask.get('p50') is not None else ''} |"
            )

        lines.extend(
            [
                "",
                "| Side | Samples | Avg spread | P90 spread | Min top ask $ | P50 total ask $ |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for side, stats in sorted((liquidity.get("sides") or {}).items()):
            spread = stats.get("spread") or {}
            top_ask = stats.get("top_ask_notional") or {}
            total_ask = stats.get("total_ask_notional") or {}
            lines.append(
                f"| {_cell(side)} | {int(stats.get('samples') or 0)} | "
                f"{_fmt_optional(spread.get('avg'), places=4)} | "
                f"{_fmt_optional(spread.get('p90'), places=4)} | "
                f"{_money(top_ask.get('min')) if top_ask.get('min') is not None else ''} | "
                f"{_money(total_ask.get('p50')) if total_ask.get('p50') is not None else ''} |"
            )

    lines.extend(
        [
            "",
            "## Files",
            "",
            f"- JSON snapshot: `{json_path}`",
            f"- Latest Markdown: `{json_path.with_name('btc5m_paper_report_latest.md')}`",
            f"- Latest JSON: `{json_path.with_name('btc5m_paper_report_latest.json')}`",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate a BTC 5m paper comparison report.")
    parser.add_argument("--ledger-dir", type=Path, default=BTC5M_PAPER_LEDGER_DIR)
    parser.add_argument("--profiles", default=BTC5M_PAPER_PROFILES)
    parser.add_argument("--settlement-source", default="polymarket")
    parser.add_argument("--signal-log-path", type=Path, default=BTC5M_SIGNAL_LOG_PATH)
    args = parser.parse_args()

    generated_at = datetime.now(timezone.utc)
    args.ledger_dir.mkdir(parents=True, exist_ok=True)

    result = run_btc5m_paper_compare_once(
        profile_names=args.profiles,
        ledger_dir=args.ledger_dir,
        settle_only=True,
        settlement_source=args.settlement_source,
    )
    result["requested_profiles"] = list((result.get("profiles") or {}).keys())
    result["liquidity"] = _book_liquidity_summary(
        args.signal_log_path,
        expected_profiles=result["requested_profiles"],
    )

    stamp = generated_at.strftime("%Y%m%d_%H%M%S")
    json_path = args.ledger_dir / f"btc5m_paper_report_{stamp}.json"
    markdown_path = args.ledger_dir / f"btc5m_paper_report_{stamp}.md"
    latest_json_path = args.ledger_dir / "btc5m_paper_report_latest.json"
    latest_markdown_path = args.ledger_dir / "btc5m_paper_report_latest.md"

    json_text = json.dumps(result, indent=2, sort_keys=True)
    markdown_text = build_markdown(result, generated_at=generated_at, json_path=json_path)

    json_path.write_text(json_text + "\n", encoding="utf-8")
    latest_json_path.write_text(json_text + "\n", encoding="utf-8")
    markdown_path.write_text(markdown_text + "\n", encoding="utf-8")
    latest_markdown_path.write_text(markdown_text + "\n", encoding="utf-8")

    print(markdown_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
