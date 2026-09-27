
import pandas as pd


# ============================================================
# CONFIG
# ============================================================

CANDIDATE_FILE = "v7_2_candidates.tsv"
GT_FILE = "dataset/train/train_ground_truth.tsv"


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("EVALUATING V7.2 CANDIDATE GENERATION")
print("=" * 70)


# ============================================================
# LOAD CANDIDATES
# ============================================================

print("\nLoading V7.2 candidate file...")

candidates = pd.read_csv(
    CANDIDATE_FILE,
    sep="\t",
    dtype=str,
    keep_default_na=False
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
    dtype=str,
    keep_default_na=False
)

print("Ground truth rows:", len(gt))


# ============================================================
# REQUIRED COLUMNS
# ============================================================

SOURCE_COL = "source1_entity_id"
CANDIDATE_COL = "candidate_entity_ids"
GT_MATCH_COL = "matched_entity_ids"


required_candidate_columns = [
    SOURCE_COL,
    CANDIDATE_COL
]

required_gt_columns = [
    SOURCE_COL,
    GT_MATCH_COL
]


for col in required_candidate_columns:

    if col not in candidates.columns:

        print("\nERROR: Missing candidate column:")
        print(col)

        print("\nAvailable columns:")
        for x in candidates.columns:
            print(" -", x)

        raise SystemExit(1)


for col in required_gt_columns:

    if col not in gt.columns:

        print("\nERROR: Missing ground-truth column:")
        print(col)

        print("\nAvailable columns:")
        for x in gt.columns:
            print(" -", x)

        raise SystemExit(1)


print("\nUsing source column:", SOURCE_COL)
print("Using candidate column:", CANDIDATE_COL)
print("Using ground-truth column:", GT_MATCH_COL)


# ============================================================
# BUILD GROUND-TRUTH INDEX
# ============================================================

print("\nBuilding ground-truth index...")

gt_dict = {}


for source_id, matched_ids in zip(
    gt[SOURCE_COL].values,
    gt[GT_MATCH_COL].values
):

    source_id = str(source_id).strip()

    if not source_id:
        continue

    matched_ids = str(matched_ids).strip()

    if not matched_ids:

        truth = set()

    else:

        # Ground truth normally uses comma-separated IDs.
        # Also handle semicolon-separated values just in case.

        matched_ids = matched_ids.replace(
            ";",
            ","
        )

        truth = {
            x.strip()
            for x in matched_ids.split(",")
            if x.strip()
        }

    gt_dict[source_id] = truth


print(
    "Ground-truth source IDs:",
    len(gt_dict)
)


# ============================================================
# VALIDATE CANDIDATE FILE
# ============================================================

print("\nValidating candidate file...")


duplicate_ids = int(
    candidates[SOURCE_COL].duplicated().sum()
)


empty_candidates = 0


for value in candidates[CANDIDATE_COL].values:

    value = str(value).strip()

    if not value:

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

print("\nEvaluating V7.2 candidate recall...")


K_VALUES = [
    1,
    5,
    10,
    20,
    50,
    100
]


hits = {
    k: 0
    for k in K_VALUES
}


evaluated = 0

missing_ground_truth = 0

no_candidate_rows = 0

total_candidate_count = 0

maximum_candidates = 0


# ============================================================
# PROCESS CANDIDATES
# ============================================================

for _, row in candidates.iterrows():

    source_id = str(
        row[SOURCE_COL]
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

    raw_candidates = str(
        row[CANDIDATE_COL]
    ).strip()


    if not raw_candidates:

        candidate_ids = []

    else:

        # Candidate generator uses comma-separated IDs.

        candidate_ids = [
            x.strip()
            for x in raw_candidates.split(",")
            if x.strip()
        ]


    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    candidate_count = len(
        candidate_ids
    )

    total_candidate_count += (
        candidate_count
    )

    maximum_candidates = max(
        maximum_candidates,
        candidate_count
    )


    if candidate_count == 0:

        no_candidate_rows += 1


    evaluated += 1


    # --------------------------------------------------------
    # Recall@K
    # --------------------------------------------------------

    for k in K_VALUES:

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
print("V7.2 CANDIDATE RECALL RESULTS")
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
    no_candidate_rows
)


if evaluated > 0:

    average_candidates = (
        total_candidate_count
        / evaluated
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


# ============================================================
# RECALL RESULTS
# ============================================================

print("\nRecall results:")


recall_values = {}


for k in K_VALUES:

    if evaluated > 0:

        recall = (
            hits[k]
            / evaluated
            * 100.0
        )

    else:

        recall = 0.0


    recall_values[k] = recall


    print(
        f"Candidate Recall@{k:<3}: "
        f"{recall:.2f}%"
    )


# ============================================================
# V7 BASELINE
# ============================================================

V7_BASELINE = {
    1: 81.40,
    5: 86.70,
    10: 87.50,
    20: 88.00,
    50: 88.00
}


print("\n" + "=" * 70)
print("V7 BASELINE COMPARISON")
print("=" * 70)


for k in [
    1,
    5,
    10,
    20,
    50
]:

    v7 = V7_BASELINE[k]

    v72 = recall_values[k]

    change = v72 - v7


    print(
        f"Recall@{k:<3}: "
        f"V7 = {v7:.2f}% | "
        f"V7.2 = {v72:.2f}% | "
        f"Change = {change:+.2f} points"
    )


# ============================================================
# QUICK DIAGNOSTIC
# ============================================================

print("\n" + "=" * 70)
print("QUICK DIAGNOSTIC")
print("=" * 70)


if evaluated == 0:

    print(
        "ERROR: No records were evaluated."
    )

elif missing_ground_truth > 0:

    print(
        "WARNING:",
        missing_ground_truth,
        "candidate rows have no ground truth."
    )

elif no_candidate_rows > 0:

    print(
        "WARNING:",
        no_candidate_rows,
        "rows have zero candidates."
    )

else:

    print(
        "All candidate rows have ground truth."
    )

    print(
        "All candidate rows contain at least one candidate."
    )


print(
    "\nRecall@100:",
    f"{recall_values[100]:.2f}%"
)


# ============================================================
# FINAL
# ============================================================

print("\n" + "=" * 70)
print("DONE!")
print("=" * 70)