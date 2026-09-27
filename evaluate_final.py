import pandas as pd
import time

print("=" * 70)
print("FINAL EVALUATION - V7 + LIGHTGBM RANKER")
print("=" * 70)

START = time.time()

# ============================================================
# FILES
# ============================================================

RANKED_FILE = "v7_ranked_final.tsv"
GROUND_TRUTH_FILE = "dataset/train/train_ground_truth.tsv"

# ============================================================
# LOAD RANKED FILE
# ============================================================

print("\nLoading final ranked candidates...")

ranked = pd.read_csv(
    RANKED_FILE,
    sep="\t",
    dtype=str
)

print("Ranked rows:", len(ranked))
print("Columns:", ranked.columns.tolist())

# ============================================================
# LOAD GROUND TRUTH
# ============================================================

print("\nLoading ground truth...")

gt = pd.read_csv(
    GROUND_TRUTH_FILE,
    sep="\t",
    dtype=str
)

print("Ground-truth rows:", len(gt))
print("Ground-truth columns:", gt.columns.tolist())

# ============================================================
# DETECT COLUMNS
# ============================================================

source_col = "source1_entity_id"
candidate_col = "candidate_entity_ids"

gt_source_col = "source1_entity_id"
gt_match_col = "matched_entity_ids"

# ============================================================
# BUILD GROUND TRUTH INDEX
# ============================================================

print("\nBuilding ground-truth index...")

ground_truth = {}

for _, row in gt.iterrows():

    source_id = str(row[gt_source_col]).strip()

    matches = str(row[gt_match_col]).strip()

    if not matches or matches.lower() == "nan":
        ground_truth[source_id] = set()
        continue

    match_set = set(
        x.strip()
        for x in matches.split(",")
        if x.strip()
    )

    ground_truth[source_id] = match_set

print(
    "Ground-truth source IDs:",
    len(ground_truth)
)

# ============================================================
# VALIDATION
# ============================================================

print("\nValidating final ranked file...")

duplicate_ids = ranked[source_col].duplicated().sum()

empty_candidates = 0
missing_gt = 0

for _, row in ranked.iterrows():

    source_id = str(row[source_col]).strip()

    candidates = str(row[candidate_col]).strip()

    if not candidates or candidates.lower() == "nan":
        empty_candidates += 1

    if source_id not in ground_truth:
        missing_gt += 1

print("Duplicate Source-1 IDs:", duplicate_ids)
print("Rows with empty candidates:", empty_candidates)
print("Rows without ground truth:", missing_gt)

# ============================================================
# RECALL EVALUATION
# ============================================================

print("\nEvaluating final ranked candidates...")

recall_at = [1, 5, 10, 20, 50, 100]

hits = {
    k: 0
    for k in recall_at
}

evaluated = 0
candidate_counts = []

for _, row in ranked.iterrows():

    source_id = str(row[source_col]).strip()

    if source_id not in ground_truth:
        continue

    truth = ground_truth[source_id]

    if not truth:
        continue

    candidate_string = str(row[candidate_col]).strip()

    candidates = [
        x.strip()
        for x in candidate_string.split(",")
        if x.strip()
    ]

    if not candidates:
        continue

    evaluated += 1
    candidate_counts.append(len(candidates))

    for k in recall_at:

        top_k = candidates[:k]

        if any(candidate in truth for candidate in top_k):
            hits[k] += 1

# ============================================================
# RESULTS
# ============================================================

print("\n" + "=" * 70)
print("FINAL RANKED RECALL RESULTS")
print("=" * 70)

print("Source 1 records evaluated:", evaluated)

if candidate_counts:
    print(
        "Average candidates:",
        round(sum(candidate_counts) / len(candidate_counts), 2)
    )

    print(
        "Maximum candidates:",
        max(candidate_counts)
    )

print("\nRecall results:")

for k in recall_at:

    recall = (
        hits[k] / evaluated * 100
        if evaluated > 0
        else 0
    )

    print(
        f"Final Recall@{k:<3}: {recall:.2f}%"
    )

# ============================================================
# V7 BASELINE
# ============================================================

baseline = {
    1: 81.40,
    5: 86.70,
    10: 87.50,
    20: 88.00,
    50: 88.00,
    100: 88.00
}

print("\n" + "=" * 70)
print("V7 BASELINE COMPARISON")
print("=" * 70)

for k in recall_at:

    final_recall = (
        hits[k] / evaluated * 100
        if evaluated > 0
        else 0
    )

    change = final_recall - baseline[k]

    sign = "+" if change >= 0 else ""

    print(
        f"Recall@{k:<3}: "
        f"V7 = {baseline[k]:.2f}% | "
        f"Final = {final_recall:.2f}% | "
        f"Change = {sign}{change:.2f} points"
    )

# ============================================================
# STATUS
# ============================================================

final_r1 = (
    hits[1] / evaluated * 100
    if evaluated > 0
    else 0
)

final_r100 = (
    hits[100] / evaluated * 100
    if evaluated > 0
    else 0
)

print("\n" + "=" * 70)
print("FINAL STATUS")
print("=" * 70)

print(f"Final Recall@1   : {final_r1:.2f}%")
print(f"Final Recall@100 : {final_r100:.2f}%")

if final_r100 > 88.0:
    print("\n✓ FINAL MODEL BEATS V7 BASELINE")
elif final_r100 == 88.0:
    print("\n= FINAL MODEL MATCHES V7 BASELINE")
else:
    print("\n✗ FINAL MODEL IS BELOW V7 BASELINE")

print("\nEvaluation time:", round(time.time() - START, 2), "seconds")

print("=" * 70)
print("DONE!")
print("=" * 70)
