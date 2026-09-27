import pandas as pd


# ============================================================
# CONFIG
# ============================================================

CANDIDATE_FILE = "v8_fts_candidates.tsv"
GROUND_TRUTH_FILE = "dataset/train/train_ground_truth.tsv"


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("EVALUATING V8 FTS5 CANDIDATE GENERATION")
print("=" * 70)


# ============================================================
# LOAD CANDIDATES
# ============================================================

print("\nLoading V8 candidate file...")

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
    GROUND_TRUTH_FILE,
    sep="\t",
    dtype=str
)

print("Ground truth rows:", len(gt))


# ============================================================
# FIND SOURCE ID COLUMN
# ============================================================

source_columns = [
    "source1_entity_id",
    "source_id"
]

source_col = None

for col in source_columns:

    if col in candidates.columns:

        source_col = col
        break


if source_col is None:

    print("\nERROR: Could not find Source-1 ID column.")

    print("Available columns:")

    for col in candidates.columns:
        print(" -", col)

    raise SystemExit


# ============================================================
# FIND CANDIDATE COLUMN
# ============================================================

candidate_columns = [
    "candidate_entity_ids",
    "candidate_ids",
    "matched_entity_ids",
    "candidates"
]

candidate_col = None

for col in candidate_columns:

    if col in candidates.columns:

        candidate_col = col
        break


if candidate_col is None:

    print("\nERROR: Could not find candidate column.")

    print("Available columns:")

    for col in candidates.columns:
        print(" -", col)

    raise SystemExit


print("\nUsing source column:", source_col)
print("Using candidate column:", candidate_col)


# ============================================================
# BUILD GROUND TRUTH INDEX
# ============================================================

print("\nBuilding ground-truth index...")

gt_dict = {}

for _, row in gt.iterrows():

    source_id = str(
        row["source1_entity_id"]
    ).strip()

    raw_matches = row["matched_entity_ids"]

    if pd.isna(raw_matches):

        matches = set()

    else:

        matches = set(
            x.strip()
            for x in str(raw_matches).split(",")
            if x.strip()
        )

    gt_dict[source_id] = matches


print(
    "Ground-truth source IDs:",
    len(gt_dict)
)


# ============================================================
# VALIDATION
# ============================================================

print("\nValidating candidate file...")

duplicate_ids = (
    candidates[source_col]
    .duplicated()
    .sum()
)

empty_candidates = 0

for value in candidates[candidate_col]:

    if pd.isna(value):

        empty_candidates += 1

    elif not str(value).strip():

        empty_candidates += 1


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

print("\nEvaluating V8 candidate recall...")

evaluated = 0

missing_ground_truth = 0
no_candidates = 0

hits = {
    1: 0,
    5: 0,
    10: 0,
    20: 0,
    50: 0,
    100: 0
}


total_candidates = 0
maximum_candidates = 0


for _, row in candidates.iterrows():

    source_id = str(
        row[source_col]
    ).strip()


    # --------------------------------------------------------
    # Ground truth
    # --------------------------------------------------------

    if source_id not in gt_dict:

        missing_ground_truth += 1

        continue


    truth = gt_dict[source_id]


    # --------------------------------------------------------
    # Candidate list
    # --------------------------------------------------------

    raw_candidates = row[candidate_col]


    if pd.isna(raw_candidates):

        candidate_ids = []

    else:

        candidate_ids = [
            x.strip()
            for x in str(raw_candidates).split(",")
            if x.strip()
        ]


    if len(candidate_ids) == 0:

        no_candidates += 1


    total_candidates += len(candidate_ids)

    maximum_candidates = max(
        maximum_candidates,
        len(candidate_ids)
    )


    evaluated += 1


    # --------------------------------------------------------
    # Recall @ K
    # --------------------------------------------------------

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
print("V8 FTS5 CANDIDATE RECALL RESULTS")
print("=" * 70)

print(
    "Source 1 records evaluated:",
    evaluated
)

print(
    "Source IDs without ground truth:",
    missing_ground_truth
)

print(
    "Rows with no candidates:",
    no_candidates
)


if evaluated > 0:

    average_candidates = (
        total_candidates / evaluated
    )

else:

    average_candidates = 0


print(
    "Average candidates:",
    f"{average_candidates:.2f}"
)

print(
    "Maximum candidates:",
    maximum_candidates
)


print("\nRecall results:")

for k in [1, 5, 10, 20, 50, 100]:

    if evaluated:

        recall = (
            hits[k] /
            evaluated *
            100
        )

    else:

        recall = 0


    print(
        f"Candidate Recall@{k:<3}: "
        f"{recall:.2f}%"
    )


# ============================================================
# V7 COMPARISON
# ============================================================

v7_scores = {
    1: 81.40,
    5: 86.70,
    10: 87.50,
    20: 88.00,
    50: 88.00
}


print("\n" + "=" * 70)
print("V7 vs V8 COMPARISON")
print("=" * 70)

for k in [1, 5, 10, 20, 50]:

    if evaluated:

        v8_recall = (
            hits[k] /
            evaluated *
            100
        )

    else:

        v8_recall = 0


    v7_recall = v7_scores[k]

    change = v8_recall - v7_recall


    print(
        f"Recall@{k:<3}: "
        f"V7 = {v7_recall:.2f}% | "
        f"V8 = {v8_recall:.2f}% | "
        f"Change = {change:+.2f} points"
    )


# ============================================================
# FINAL
# ============================================================

print("\n" + "=" * 70)
print("DONE!")
print("=" * 70)