import pandas as pd


# ============================================================
# 1. LOAD V6 CANDIDATES
# ============================================================

print("Loading V6 candidates...")

v6 = pd.read_csv("v6_candidates.csv")

print("V6 candidate rows:", len(v6))


# ============================================================
# 2. LOAD GROUND TRUTH
# ============================================================

print("Loading ground truth...")

ground_truth = pd.read_csv(
    "dataset/train/train_ground_truth.tsv",
    sep="\t"
)

print("Ground truth rows:", len(ground_truth))


# ============================================================
# 3. CREATE GROUND-TRUTH LOOKUP
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
# 4. EVALUATION
# ============================================================

print("\n======================================")
print("EVALUATING V6")
print("======================================")


total_records = 0

found_top1 = 0
found_top5 = 0
found_top10 = 0
found_top20 = 0
found_top50 = 0
found_top100 = 0


for source1_id, group in v6.groupby(
    "source1_entity_id"
):

    # Skip if no ground truth
    if source1_id not in truth:
        continue

    true_ids = truth[source1_id]

    if not true_ids:
        continue

    total_records += 1


    # ========================================================
    # SORT BY RAPIDFUZZ FINAL SCORE
    # ========================================================

    group = group.sort_values(
        "final_score",
        ascending=False
    )


    # ========================================================
    # TOP 1
    # ========================================================

    top1 = set(
        group.head(1)["candidate_entity_id"]
    )

    if true_ids.intersection(top1):
        found_top1 += 1


    # ========================================================
    # TOP 5
    # ========================================================

    top5 = set(
        group.head(5)["candidate_entity_id"]
    )

    if true_ids.intersection(top5):
        found_top5 += 1


    # ========================================================
    # TOP 10
    # ========================================================

    top10 = set(
        group.head(10)["candidate_entity_id"]
    )

    if true_ids.intersection(top10):
        found_top10 += 1


    # ========================================================
    # TOP 20
    # ========================================================

    top20 = set(
        group.head(20)["candidate_entity_id"]
    )

    if true_ids.intersection(top20):
        found_top20 += 1


    # ========================================================
    # TOP 50
    # ========================================================

    top50 = set(
        group.head(50)["candidate_entity_id"]
    )

    if true_ids.intersection(top50):
        found_top50 += 1


    # ========================================================
    # TOP 100
    # ========================================================

    top100 = set(
        group.head(100)["candidate_entity_id"]
    )

    if true_ids.intersection(top100):
        found_top100 += 1


# ============================================================
# 5. RESULTS
# ============================================================

print("\n======================================")
print("V6 EVALUATION RESULTS")
print("======================================")

print(
    "Source 1 records evaluated:",
    total_records
)

print()


if total_records > 0:

    recall1 = (
        found_top1 / total_records
    ) * 100

    recall5 = (
        found_top5 / total_records
    ) * 100

    recall10 = (
        found_top10 / total_records
    ) * 100

    recall20 = (
        found_top20 / total_records
    ) * 100

    recall50 = (
        found_top50 / total_records
    ) * 100

    recall100 = (
        found_top100 / total_records
    ) * 100


    print(
        f"Recall@1   : {recall1:.2f}%"
    )

    print(
        f"Recall@5   : {recall5:.2f}%"
    )

    print(
        f"Recall@10  : {recall10:.2f}%"
    )

    print(
        f"Recall@20  : {recall20:.2f}%"
    )

    print(
        f"Recall@50  : {recall50:.2f}%"
    )

    print(
        f"Recall@100 : {recall100:.2f}%"
    )


print("\n======================================")
print("DONE!")
print("======================================")