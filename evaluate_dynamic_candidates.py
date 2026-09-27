import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

CANDIDATE_FILE = "v6_candidates_improved.csv"
GT_FILE = "dataset/train/train_ground_truth.tsv"


# ============================================================
# START
# ============================================================

print("=" * 70)
print("EVALUATING V7 IMPROVED CANDIDATE GENERATION")
print("=" * 70)


# ============================================================
# LOAD CANDIDATES
# ============================================================

print("\nLoading candidate file...")

candidates = pd.read_csv(
    CANDIDATE_FILE,
    sep="\t",
    dtype=str
)

print(
    "Candidate rows:",
    len(candidates)
)


# ============================================================
# LOAD GROUND TRUTH
# ============================================================

print("\nLoading ground truth...")

gt = pd.read_csv(
    GT_FILE,
    sep="\t",
    dtype=str
)

print(
    "Ground truth rows:",
    len(gt)
)


# ============================================================
# CHECK COLUMNS
# ============================================================

print("\nCandidate columns:")

for col in candidates.columns:
    print(" -", col)


required_source_column = "source1_entity_id"

if required_source_column not in candidates.columns:

    print(
        "\nERROR: Missing source1_entity_id column."
    )

    raise SystemExit


# ============================================================
# BUILD GROUND-TRUTH DICTIONARY
# ============================================================

print("\nBuilding ground-truth index...")

gt_dict = {}

for _, row in gt.iterrows():

    source_id = str(
        row["source1_entity_id"]
    ).strip()

    raw_matches = row.get(
        "matched_entity_ids",
        ""
    )

    if pd.isna(raw_matches):

        raw_matches = ""

    raw_matches = str(
        raw_matches
    ).strip()


    if not raw_matches:

        matches = set()

    else:

        matches = set(
            x.strip()
            for x in raw_matches.split(",")
            if x.strip()
        )


    gt_dict[source_id] = matches


print(
    "Ground-truth source IDs:",
    len(gt_dict)
)


# ============================================================
# FIND CANDIDATE COLUMN
# ============================================================

possible_candidate_cols = [

    "candidate_entity_ids",

    "candidate_ids",

    "matched_entity_ids",

    "candidates"
]


candidate_col = None


for col in possible_candidate_cols:

    if col in candidates.columns:

        candidate_col = col
        break


if candidate_col is None:

    print(
        "\nERROR: Could not find candidate ID column."
    )

    print(
        "Available columns:"
    )

    for col in candidates.columns:
        print(
            " -",
            col
        )

    raise SystemExit


print(
    "\nUsing source column:",
    required_source_column
)

print(
    "Using candidate column:",
    candidate_col
)


# ============================================================
# BASIC VALIDATION
# ============================================================

print("\nValidating candidate file...")

duplicate_source_ids = (
    candidates[
        required_source_column
    ]
    .duplicated()
    .sum()
)


print(
    "Duplicate Source-1 IDs:",
    duplicate_source_ids
)


empty_candidate_rows = 0


for value in candidates[
    candidate_col
]:

    if pd.isna(value):

        empty_candidate_rows += 1

    elif not str(value).strip():

        empty_candidate_rows += 1


print(
    "Rows with empty candidates:",
    empty_candidate_rows
)


# ============================================================
# EVALUATION
# ============================================================

print("\nEvaluating candidate recall...")


evaluated = 0

hits = {

    1: 0,

    5: 0,

    10: 0,

    20: 0,

    50: 0
}


# Extra diagnostic counters

no_ground_truth = 0

no_candidates = 0

total_candidate_count = 0

max_candidate_count = 0


# ============================================================
# LOOP THROUGH PREDICTIONS
# ============================================================

for _, row in candidates.iterrows():

    source_id = str(
        row[required_source_column]
    ).strip()


    # --------------------------------------------------------
    # Ground truth lookup
    # --------------------------------------------------------

    if source_id not in gt_dict:

        no_ground_truth += 1

        continue


    truth = gt_dict[source_id]


    # --------------------------------------------------------
    # Parse candidate IDs
    # --------------------------------------------------------

    raw_candidates = row[
        candidate_col
    ]


    if pd.isna(raw_candidates):

        candidate_ids = []

    else:

        raw_candidates = str(
            raw_candidates
        ).strip()


        if not raw_candidates:

            candidate_ids = []

        else:

            candidate_ids = [

                x.strip()

                for x in raw_candidates.split(",")

                if x.strip()

            ]


    # --------------------------------------------------------
    # Candidate statistics
    # --------------------------------------------------------

    candidate_count = len(
        candidate_ids
    )


    total_candidate_count += (
        candidate_count
    )


    max_candidate_count = max(
        max_candidate_count,
        candidate_count
    )


    if candidate_count == 0:

        no_candidates += 1


    evaluated += 1


    # --------------------------------------------------------
    # Recall @ K
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
print("V7 CANDIDATE RECALL RESULTS")
print("=" * 70)


print(
    "Source 1 records evaluated:",
    evaluated
)


print(
    "Source IDs without ground truth:",
    no_ground_truth
)


print(
    "Rows with no candidates:",
    no_candidates
)


if evaluated > 0:

    average_candidates = (
        total_candidate_count
        /
        evaluated
    )

else:

    average_candidates = 0


print(
    "Average candidates:",
    f"{average_candidates:.2f}"
)


print(
    "Maximum candidates:",
    max_candidate_count
)


print("\nRecall results:")


for k in [
    1,
    5,
    10,
    20,
    50
]:

    if evaluated > 0:

        recall = (
            hits[k]
            /
            evaluated
            *
            100
        )

    else:

        recall = 0


    print(
        f"Candidate Recall@{k:<3}: "
        f"{recall:.2f}%"
    )


# ============================================================
# COMPARISON WITH V6
# ============================================================

print("\n" + "=" * 70)
print("V6 BASELINE COMPARISON")
print("=" * 70)


v6_baseline = {

    1: 57.20,

    5: 70.90,

    10: 75.40,

    20: 79.00,

    50: 83.00
}


for k in [

    1,
    5,
    10,
    20,
    50

]:

    if evaluated > 0:

        v7_recall = (
            hits[k]
            /
            evaluated
            *
            100
        )

    else:

        v7_recall = 0


    improvement = (
        v7_recall
        -
        v6_baseline[k]
    )


    print(
        f"Recall@{k:<3}: "
        f"V6 = {v6_baseline[k]:.2f}% | "
        f"V7 = {v7_recall:.2f}% | "
        f"Change = {improvement:+.2f} points"
    )


# ============================================================
# FINAL MESSAGE
# ============================================================

print("\n" + "=" * 70)

if evaluated == 0:

    print(
        "ERROR: No records were evaluated."
    )

elif (
    hits[50] / evaluated * 100
    > v6_baseline[50]
):

    print(
        "V7 BEATS THE V6 BASELINE!"
    )

    print(
        "Keep V7 and continue testing."
    )

else:

    print(
        "V7 DOES NOT BEAT THE V6 BASELINE YET."
    )

    print(
        "Do NOT run the full dataset yet."
    )


print("=" * 70)
print("DONE!")
print("=" * 70)