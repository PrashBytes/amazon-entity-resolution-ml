import pandas as pd

PRED_FILE = "output/matching_results_dynamic_train.tsv"
GT_FILE = "dataset/train/train_ground_truth.tsv"

print("=" * 70)
print("EVALUATING DYNAMIC-THRESHOLD TRAIN TEST")
print("=" * 70)

print("\nLoading predictions...")
pred = pd.read_csv(PRED_FILE, sep="\t", dtype=str)

print("Prediction rows:", len(pred))

print("\nLoading ground truth...")
gt = pd.read_csv(GT_FILE, sep="\t", dtype=str)

print("Ground truth rows:", len(gt))

# ---------------------------------------------------------
# Build ground-truth dictionary
# ---------------------------------------------------------

gt_dict = {}

for _, row in gt.iterrows():
    source_id = row["source1_entity_id"]

    matches = str(row["matched_entity_ids"])

    if matches == "nan":
        matches = ""

    gt_dict[source_id] = set(
        x.strip()
        for x in matches.split(",")
        if x.strip()
    )

# ---------------------------------------------------------
# Evaluate
# ---------------------------------------------------------

results = {
    1: 0,
    5: 0,
    10: 0,
    20: 0,
    50: 0,
}

evaluated = 0

for _, row in pred.iterrows():

    source_id = row["source1_entity_id"]

    if source_id not in gt_dict:
        continue

    predicted = str(row["matched_entity_ids"])

    if predicted == "nan" or predicted.strip() == "":
        predicted_ids = []
    else:
        predicted_ids = [
            x.strip()
            for x in predicted.split(",")
            if x.strip()
        ]

    truth = gt_dict[source_id]

    evaluated += 1

    for k in results:
        top_k = predicted_ids[:k]

        if any(x in truth for x in top_k):
            results[k] += 1

# ---------------------------------------------------------
# Print results
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("DYNAMIC TRAIN EVALUATION RESULTS")
print("=" * 70)

print("Source 1 records evaluated:", evaluated)

for k in [1, 5, 10, 20, 50]:

    if evaluated > 0:
        recall = results[k] / evaluated * 100
    else:
        recall = 0

    print(f"Recall@{k:<3}: {recall:.2f}%")

print("\n" + "=" * 70)
print("DONE!")
print("=" * 70)