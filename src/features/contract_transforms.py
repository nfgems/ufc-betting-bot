"""Self-contained declared transforms shared with legacy feature callers."""

import pandas as pd


def _coerce_nullable_binary(series: pd.Series) -> pd.Series:
    """Coerce a mixed binary series to float 0/1 while preserving unknowns as NaN."""
    coerced = pd.to_numeric(series, errors="coerce").astype(float)
    unresolved = coerced.isna() & series.notna()
    if unresolved.any():
        normalized = series.astype(str).str.strip().str.lower()
        truthy = {"1", "true", "t", "yes", "y"}
        falsy = {"0", "false", "f", "no", "n"}
        coerced.loc[unresolved & normalized.isin(truthy)] = 1.0
        coerced.loc[unresolved & normalized.isin(falsy)] = 0.0
    return coerced


def materialize_honest_context_features(features_df: pd.DataFrame) -> pd.DataFrame:
    """
    Rebuild context/ranking features directly from raw columns without fake defaults.

    Historical unknowns must stay NaN instead of being rewritten as title_bout=0,
    empty_arena=0, or rank=16 unless the raw value is actually present.
    """
    features = features_df.copy()

    if "title_bout" in features.columns:
        features["is_title_bout"] = _coerce_nullable_binary(features["title_bout"])

    if "num_rounds" in features.columns:
        features["num_rounds_feat"] = pd.to_numeric(features["num_rounds"], errors="coerce").astype(float)

    if "empty_arena" in features.columns:
        features["is_empty_arena"] = _coerce_nullable_binary(features["empty_arena"])

    for prefix in ["a_", "b_"]:
        wc_rank_col = f"{prefix}wc_rank"
        if wc_rank_col in features.columns:
            features[f"{prefix}wc_rank_feat"] = pd.to_numeric(
                features[wc_rank_col],
                errors="coerce",
            ).astype(float)

        pfp_rank_col = f"{prefix}pfp_rank"
        if pfp_rank_col in features.columns:
            features[f"{prefix}pfp_rank_feat"] = pd.to_numeric(
                features[pfp_rank_col],
                errors="coerce",
            ).astype(float)

    if "a_wc_rank_feat" in features.columns and "b_wc_rank_feat" in features.columns:
        features["diff_wc_rank"] = features["a_wc_rank_feat"] - features["b_wc_rank_feat"]
    if "a_pfp_rank_feat" in features.columns and "b_pfp_rank_feat" in features.columns:
        features["diff_pfp_rank"] = features["a_pfp_rank_feat"] - features["b_pfp_rank_feat"]

    return features


def add_rematch_features(features_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add rematch detection features:
    - is_rematch: binary flag if fighters have fought before
    - h2h_record_diff: A's wins minus B's wins in prior meetings
    """
    features_df = features_df.sort_values("event_date").copy()

    # Build H2H history chronologically
    h2h: dict[tuple[str, str], list[str]] = {}  # (sorted pair) -> list of winners

    is_rematch = []
    h2h_diff = []

    for _, row in features_df.iterrows():
        fa = row.get("fighter_a", "")
        fb = row.get("fighter_b", "")
        winner = row.get("winner", "")
        pair = tuple(sorted([fa, fb]))

        prior_fights = h2h.get(pair, [])
        if prior_fights:
            is_rematch.append(1)
            a_prior_wins = sum(1 for w in prior_fights if w == fa)
            b_prior_wins = sum(1 for w in prior_fights if w == fb)
            h2h_diff.append(a_prior_wins - b_prior_wins)
        else:
            is_rematch.append(0)
            h2h_diff.append(0)

        # Record this fight's result
        if pair not in h2h:
            h2h[pair] = []
        if winner:
            h2h[pair].append(winner)

    features_df["is_rematch"] = is_rematch
    features_df["h2h_record_diff"] = h2h_diff

    return features_df


