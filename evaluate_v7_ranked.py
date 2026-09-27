import pandas as pd

RANKED_FILE = "v7_ranked_candidates.tsv"
GT_FILE = "dataset/train/train_ground_truth.tsv"

print("=" * 70)
print("EVALUATING V7 + LIGHTGBM RANKER")
print("=" * 70)

print("\nLoading ranked candidates...")

ranked = pd.read_csv(
    RANKED_FILE,
    sep="\t",
    dtype=str
)

print("Ranked rows:", len(ranked))
print("Columns:", ranked.columns.tolist())


print("\nLoading ground truth...")

gt = pd.read_csv(
    GT_FILE,
    sep="\t",
    dtype=str
)

print("Ground truth rows:", len(gt))


# ============================================================
# BUILD GROUND TRUTH INDEX
# ============================================================

print("\nBuilding ground-truth index...")

gt_dict = {}

for _, row in gt.iterrows():

    source_id = row["source1_entity_id"]

    raw = row["matched_entity_ids"]

    if pd.isna(raw):
        truth = set()
    else:
        truth = {
            x.strip()
            for x in str(raw).split(",")
            if x.strip()
        }

    gt_dict[source_id] = truth


# ============================================================
# EVALUATE
# ============================================================

hits = {
    1: 0,
    5: 0,
    10: 0,
    20: 0,
    50: 0,
    100: 0
}

evaluated = 0
empty = 0
missing_gt = 0
total_candidates = 0
max_candidates = 0


for _, row in ranked.iterrows():

    source_id = row["source1_entity_id"]

    if source_id not in gt_dict:
        missing_gt += 1
        continue

    raw = row["candidate_entity_ids"]

    if pd.isna(raw) or not str(raw).strip():

        empty += 1
        continue

    candidate_ids = [
        x.strip()
        for x in str(raw).split(",")
        if x.strip()
    ]

    truth = gt_dict[source_id]

    evaluated += 1

    total_candidates += len(candidate_ids)

    max_candidates = max(
        max_candidates,
        len(candidate_ids)
    )

    for k in hits:

        top_k = candidate_ids[:k]

        if any(
            candidate in truth
            for candidate in top_k
        ):
            hits[k] += 1


# ============================================================
# RESULTS
# ============================================================

print("\n" + "=" * 70)
print("V7 + LIGHTGBM RANKED RECALL")
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
    empty
)

if evaluated:

    print(
        "Average candidates:",
        f"{total_candidates / evaluated:.2f}"
    )

else:

    print(
        "Average candidates: 0"
    )

print(
    "Maximum candidates:",
    max_candidates
)


print("\nRecall results:")

for k in [
    1,
    5,
    10,
    20,
    50,
    100
]:

    if evaluated:

        recall = (
            hits[k] /
            evaluated *
            100
        )

    else:

        recall = 0

    print(
        f"Ranked Recall@{k:<3}: "
        f"{recall:.2f}%"
    )


print("\n" + "=" * 70)
print("V7 BASELINE")
print("=" * 70)

baseline = {
    1: 81.40,
    5: 86.70,
    10: 87.50,
    20: 88.00,
    50: 88.00,
    100: 88.00
}

for k in baseline:

    if evaluated:

        ranked_recall = (
            hits[k] /
            evaluated *
            100
        )

    else:

        ranked_recall = 0

    change = ranked_recall - baseline[k]

    print(
        f"Recall@{k:<3}: "
        f"V7 = {baseline[k]:.2f}% | "
        f"LightGBM = {ranked_recall:.2f}% | "
        f"Change = {change:+.2f} points"
    )


print("\n" + "=" * 70)

if evaluated:

    r100 = hits[100] / evaluated * 100

    print(
        f"Candidate pool Recall@100: "
        f"{r100:.2f}%"
    )

print("=" * 70)
print("DONE!")
print("=" * 70)