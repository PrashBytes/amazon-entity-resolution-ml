import pandas as pd


# ============================================================
# LOAD V5 RESULTS
# ============================================================

print("Loading V5 candidates...")

v5 = pd.read_csv("v5_candidates.csv")

print("V5 candidate rows:", len(v5))


# ============================================================
# LOAD GROUND TRUTH
# ============================================================

print("Loading ground truth...")

ground_truth = pd.read_csv(
    "dataset/train/train_ground_truth.tsv",
    sep="\t"
)

print("Ground truth rows:", len(ground_truth))


# ============================================================
# CREATE GROUND-TRUTH LOOKUP
# ============================================================

truth = {}

for _, row in ground_truth.iterrows():

    source1_id = row["source1_entity_id"]

    matched_ids = str(
        row["matched_entity_ids"]
    )

    true_ids = set(
        x.strip()
        for x in matched_ids.split(",")
        if x.strip()
    )

    truth[source1_id] = true_ids


# ============================================================
# EVALUATE
# ============================================================

print("\n======================================")
print("EVALUATING V5")
print("======================================")


total_records = 0

found_top1 = 0
found_top5 = 0
found_top10 = 0

total_true_matches = 0
found_true_matches_top10 = 0


# Group candidates by Source 1 business

grouped = v5.groupby(
    "source1_entity_id"
)


for source1_id, group in grouped:

    if source1_id not in truth:
        continue

    true_ids = truth[source1_id]

    if not true_ids:
        continue

    total_records += 1

    total_true_matches += len(true_ids)


    # Sort candidates from highest score
    # to lowest score

    group = group.sort_values(
        "final_score",
        ascending=False
    )


    # --------------------------------------------------------
    # TOP 1
    # --------------------------------------------------------

    top1 = set(
        group.head(1)[
            "candidate_entity_id"
        ]
    )

    if true_ids.intersection(top1):
        found_top1 += 1


    # --------------------------------------------------------
    # TOP 5
    # --------------------------------------------------------

    top5 = set(
        group.head(5)[
            "candidate_entity_id"
        ]
    )

    if true_ids.intersection(top5):
        found_top5 += 1


    # --------------------------------------------------------
    # TOP 10
    # --------------------------------------------------------

    top10 = set(
        group.head(10)[
            "candidate_entity_id"
        ]
    )

    matched = true_ids.intersection(
        top10
    )

    if matched:
        found_top10 += 1

    found_true_matches_top10 += len(
        matched
    )


# ============================================================
# RESULTS
# ============================================================

print("\n======================================")
print("V5 EVALUATION RESULTS")
print("======================================")

print(
    "Source 1 records evaluated:",
    total_records
)

print()

if total_records > 0:

    recall_top1 = (
        found_top1
        / total_records
    ) * 100

    recall_top5 = (
        found_top5
        / total_records
    ) * 100

    recall_top10 = (
        found_top10
        / total_records
    ) * 100

    print(
        f"Recall@1 : {recall_top1:.2f}%"
    )

    print(
        f"Recall@5 : {recall_top5:.2f}%"
    )

    print(
        f"Recall@10: {recall_top10:.2f}%"
    )


print()

print(
    "Total true matches:",
    total_true_matches
)

print(
    "True matches found in Top 10:",
    found_true_matches_top10
)

if total_true_matches > 0:

    pair_recall = (
        found_true_matches_top10
        / total_true_matches
    ) * 100

    print(
        f"True-match coverage@10: "
        f"{pair_recall:.2f}%"
    )


print("\nDONE!")