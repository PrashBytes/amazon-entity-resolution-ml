
import pandas as pd
import time

# ============================================================
# CONFIG
# ============================================================

CANDIDATE_FILE = "v7_3_lite_candidates.tsv"
GT_FILE = "dataset/train/train_ground_truth.tsv"

print("=" * 70)
print("EVALUATING V7.3-LITE CANDIDATE GENERATION")
print("=" * 70)


# ============================================================
# LOAD CANDIDATES
# ============================================================

print("\nLoading V7.3-Lite candidate file...")

start = time.time()

candidates = pd.read_csv(
    CANDIDATE_FILE,
    sep="\t",
    dtype=str
)

print("Candidate rows:", len(candidates))
print("Candidate columns:", list(candidates.columns))


# ============================================================
# LOAD GROUND TRUTH
# ============================================================

print("\nLoading ground truth...")

gt = pd.read_csv(
    GT_FILE,
    sep="\t",
    dtype=str
)

print("Ground truth rows:", len(gt))


# ============================================================
# DETECT COLUMNS
# ============================================================

source_candidates = [
    "source1_entity_id",
    "source_entity_id",
    "entity_id"
]

candidate_columns = [
    "candidate_entity_ids",
    "candidate_ids",
    "matched_entity_ids",
    "candidates"
]

gt_match_columns = [
    "matched_entity_ids",
    "candidate_entity_ids",
    "matched_ids"
]


# ------------------------------------------------------------
# Source column
# ------------------------------------------------------------

source_col = None

for col in source_candidates:
    if col in candidates.columns:
        source_col = col
        break

if source_col is None:
    raise ValueError(
        "Could not find Source-1 ID column.\n"
        f"Available columns: {list(candidates.columns)}"
    )


# ------------------------------------------------------------
# Candidate column
# ------------------------------------------------------------

candidate_col = None

for col in candidate_columns:
    if col in candidates.columns:
        candidate_col = col
        break

if candidate_col is None:
    raise ValueError(
        "Could not find candidate ID column.\n"
        f"Available columns: {list(candidates.columns)}"
    )


# ------------------------------------------------------------
# Ground-truth columns
# ------------------------------------------------------------

gt_source_col = None

for col in source_candidates:
    if col in gt.columns:
        gt_source_col = col
        break

if gt_source_col is None:
    raise ValueError(
        "Could not find Source-1 column in ground truth."
    )


gt_match_col = None

for col in gt_match_columns:
    if col in gt.columns:
        gt_match_col = col
        break

if gt_match_col is None:
    raise ValueError(
        "Could not find matched entity column in ground truth."
    )


print("\nUsing source column:", source_col)
print("Using candidate column:", candidate_col)
print("Using ground-truth source column:", gt_source_col)
print("Using ground-truth column:", gt_match_col)


# ============================================================
# BUILD GROUND TRUTH INDEX
# ============================================================

print("\nBuilding ground-truth index...")

gt_dict = {}

for source_id, matches in zip(
    gt[gt_source_col],
    gt[gt_match_col]
):

    if pd.isna(matches):
        gt_dict[source_id] = set()
        continue

    match_set = {
        x.strip()
        for x in str(matches).split(",")
        if x.strip()
    }

    gt_dict[source_id] = match_set


print(
    "Ground-truth source IDs:",
    len(gt_dict)
)


# ============================================================
# VALIDATE CANDIDATE FILE
# ============================================================

print("\nValidating candidate file...")

duplicate_ids = candidates[source_col].duplicated().sum()

empty_candidates = (
    candidates[candidate_col]
    .isna()
    .sum()
)

print(
    "Duplicate Source-1 IDs:",
    duplicate_ids
)

print(
    "Rows with empty candidates:",
    empty_candidates
)


# ============================================================
# EVALUATION
# ============================================================

print("\nEvaluating V7.3-Lite candidate recall...")

evaluated = 0
missing_gt = 0
no_candidates = 0

candidate_counts = []

hits = {
    1: 0,
    5: 0,
    10: 0,
    20: 0,
    50: 0,
    100: 0
}


# ------------------------------------------------------------
# Faster iteration
# ------------------------------------------------------------

for source_id, raw_candidates in zip(
    candidates[source_col],
    candidates[candidate_col]
):

    # --------------------------------------------------------
    # Ground truth lookup
    # --------------------------------------------------------

    if source_id not in gt_dict:
        missing_gt += 1
        continue

    truth = gt_dict[source_id]

    # --------------------------------------------------------
    # Candidate parsing
    # --------------------------------------------------------

    if pd.isna(raw_candidates):

        candidate_ids = []

    else:

        candidate_ids = [
            x.strip()
            for x in str(raw_candidates).split(",")
            if x.strip()
        ]

    candidate_counts.append(len(candidate_ids))

    if not candidate_ids:
        no_candidates += 1

    evaluated += 1

    # --------------------------------------------------------
    # Recall@K
    # --------------------------------------------------------

    for k in hits:

        top_k = candidate_ids[:k]

        if any(
            candidate_id in truth
            for candidate_id in top_k
        ):
            hits[k] += 1


# ============================================================
# RESULTS
# ============================================================

print("\n" + "=" * 70)
print("V7.3-LITE CANDIDATE RECALL RESULTS")
print("=" * 70)

print(
    "Source 1 records evaluated:",
    evaluated
)

print(
    "Source IDs without ground truth:",
    missing_gt
)

print(
    "Rows with no candidates:",
    no_candidates
)

if candidate_counts:

    print(
        "Average candidates:",
        f"{sum(candidate_counts) / len(candidate_counts):.2f}"
    )

    print(
        "Maximum candidates:",
        max(candidate_counts)
    )

else:

    print("Average candidates: 0")
    print("Maximum candidates: 0")


print("\nRecall results:")

recall_results = {}

for k in [1, 5, 10, 20, 50, 100]:

    if evaluated > 0:

        recall = (
            hits[k] /
            evaluated
            * 100
        )

    else:

        recall = 0.0

    recall_results[k] = recall

    print(
        f"Candidate Recall@{k:<3}: "
        f"{recall:.2f}%"
    )


# ============================================================
# V7 BASELINE
# ============================================================

v7_baseline = {
    1: 81.40,
    5: 86.70,
    10: 87.50,
    20: 88.00,
    50: 88.00
}


print("\n" + "=" * 70)
print("V7 BASELINE COMPARISON")
print("=" * 70)

for k in [1, 5, 10, 20, 50]:

    v7 = v7_baseline[k]
    current = recall_results[k]

    change = current - v7

    print(
        f"Recall@{k:<3}: "
        f"V7 = {v7:.2f}% | "
        f"V7.3-Lite = {current:.2f}% | "
        f"Change = {change:+.2f} points"
    )


# ============================================================
# DIAGNOSTIC
# ============================================================

print("\n" + "=" * 70)
print("QUICK DIAGNOSTIC")
print("=" * 70)

if duplicate_ids == 0:
    print("✓ No duplicate Source-1 IDs.")
else:
    print(
        "WARNING:",
        duplicate_ids,
        "duplicate Source-1 IDs found."
    )


if no_candidates == 0:
    print("✓ All candidate rows contain candidates.")
else:
    print(
        "WARNING:",
        no_candidates,
        "rows contain no candidates."
    )


if missing_gt == 0:
    print("✓ All candidate rows have ground truth.")
else:
    print(
        "WARNING:",
        missing_gt,
        "candidate rows have no ground truth."
    )


print(
    f"\nRecall@100: "
    f"{recall_results[100]:.2f}%"
)


# ============================================================
# FINAL DECISION
# ============================================================

print("\n" + "=" * 70)
print("V7.3-LITE STATUS")
print("=" * 70)

if recall_results[100] > 88.0:

    print(
        "V7.3-Lite IMPROVES over the V7 baseline."
    )

    print(
        "Further optimization may be worthwhile."
    )

else:

    print(
        "V7.3-Lite does NOT improve over V7."
    )

    print(
        "Do NOT run the full 2.2M-row generation yet."
    )


print("\nEvaluation time:", f"{time.time() - start:.2f}s")

print("\n" + "=" * 70)
print("DONE!")
print("=" * 70)

