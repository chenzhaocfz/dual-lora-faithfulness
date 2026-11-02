"""Metrics and paired statistical tests (Cochran's Q, McNemar, Holm)."""
from itertools import combinations

import numpy as np
import pandas as pd

CONDITIONS = ["monolithic", "soft_modular", "hard_bottleneck"]
CONDITION_LABELS = {
    "monolithic": "Monolithic",
    "soft_modular": "Soft Modular",
    "hard_bottleneck": "Hard Bottleneck",
}
PAIRWISE = [("monolithic", "soft_modular"), ("soft_modular", "hard_bottleneck"), ("monolithic", "hard_bottleneck")]


def exact_match_table(clean):
    """clean: DataFrame with columns id, condition, correct (bool)."""
    table = clean.groupby("condition")["correct"].agg(correct="sum", total="count")
    table["EM"] = table["correct"] / table["total"]
    return table.reindex(CONDITIONS)


def trap_summary(trap):
    """trap: DataFrame for one step, with columns id, condition, outcome, clean_outcome, answer_changed."""
    rows = []
    for cond in CONDITIONS:
        sub = trap[trap["condition"] == cond]
        n = len(sub)
        rows.append({
            "condition": cond,
            "n": n,
            "trap_follows": int((sub["outcome"] == "follow").sum()),
            "TD": (sub["outcome"] == "follow").mean() if n else np.nan,
            "original_answer": int((sub["outcome"] == "original").sum()),
            "other_errors": int((sub["outcome"] == "other").sum()),
            "answer_changed_vs_clean_trace": int(sub["answer_changed"].sum()),
            "clean_trace_correct": int((sub["clean_outcome"] == "original").sum()),
        })
    return pd.DataFrame(rows).set_index("condition")


def paired_matrix(trap, outcome="follow"):
    """Problems x conditions binary matrix (1 = followed the trap). Only problems present in all conditions."""
    wide = (trap.assign(y=(trap["outcome"] == outcome).astype(int))
                .pivot(index="id", columns="condition", values="y"))
    return wide[CONDITIONS].dropna().astype(int)


def paired_tests(trap, alpha=0.05):
    """Cochran's Q across the three conditions, then exact McNemar tests with Holm correction."""
    from statsmodels.stats.contingency_tables import cochrans_q, mcnemar
    from statsmodels.stats.multitest import multipletests

    wide = paired_matrix(trap)
    q = cochrans_q(wide.values)
    omnibus = {"n_problems": len(wide), "Q": float(q.statistic), "df": int(q.df), "p_value": float(q.pvalue)}

    rows = []
    for a, b in PAIRWISE:
        both = int(((wide[a] == 1) & (wide[b] == 1)).sum())
        only_a = int(((wide[a] == 1) & (wide[b] == 0)).sum())
        only_b = int(((wide[a] == 0) & (wide[b] == 1)).sum())
        neither = int(((wide[a] == 0) & (wide[b] == 0)).sum())
        result = mcnemar([[both, only_a], [only_b, neither]], exact=True)
        # Discordant pairs: only the first / only the second condition followed the trap
        rows.append({"comparison": f"{CONDITION_LABELS[a]} vs. {CONDITION_LABELS[b]}",
                     "first_only_follows": only_a, "second_only_follows": only_b,
                     "p_value": float(result.pvalue)})
    pairwise = pd.DataFrame(rows)
    reject, p_holm, _, _ = multipletests(pairwise["p_value"], alpha=alpha, method="holm")
    pairwise["p_holm"] = p_holm
    pairwise["significant"] = reject
    pairwise["omnibus_significant"] = omnibus["p_value"] < alpha
    return omnibus, pairwise


def td_by_step(trap):
    """Trace dependence per checkpoint (rows) and condition (columns); trap needs a step column."""
    td = (trap.assign(follow=trap["outcome"] == "follow")
              .groupby(["step", "condition"], sort=False)["follow"].mean().unstack())
    return td[CONDITIONS].rename(columns=CONDITION_LABELS)
