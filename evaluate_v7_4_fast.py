import pandas as pd
import ast
import os
import time


# ============================================================
# CONFIGURATION
# ============================================================

CANDIDATE_FILE = "v7_4_fast_candidates.tsv"

GROUND_TRUTH_FILE = "dataset/train/train_ground_truth.tsv"

TOP_K_VALUES = [1, 5, 10, 20, 50, 100]


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("EVALUATING V7.4 FAST CANDIDATE GENERATION")
print("=" * 70)

start_time = time.time()


# ============================================================
# CHECK FILES
# ============================================================

if not os.path.exists(CANDIDATE_FILE):

    print()
    print("ERROR: Candidate file not found:")
    print(CANDIDATE_FILE)
    print()
    print("Make sure you have already run:")
    print("python candidate_generation_v7_4_fast.py")
    raise SystemExit


if not os.path.exists(GROUND_TRUTH_FILE):

    print()
    print("ERROR: Ground-truth file not found:")
    print(GROUND_TRUTH_FILE)
    raise SystemExit


# ============================================================
# LOAD CANDIDATES
# ============================================================

print()
print("Loading V7.4 Fast candidate file...")

candidates = pd.read_csv(
    CANDIDATE_FILE,
    sep="\t"
)

print(
    "Candidate rows:",
    len(candidates)
)

print(
    "Candidate columns:",
    list(candidates.columns)
)


# ============================================================
# DETECT COLUMNS
# ============================================================

source_column = "source1_entity_id"

candidate_column = "candidate_entity_ids"


if source_column not in candidates.columns:

    print()
    print("ERROR: Missing source1_entity_id column.")
    raise SystemExit


if candidate_column not in candidates.columns:

    print()
    print("ERROR: Missing candidate_entity_ids column.")
    raise SystemExit


# ============================================================
# LOAD GROUND TRUTH
# ============================================================

print()
print("Loading ground truth...")

ground_truth = pd.read_csv(
    GROUND_TRUTH_FILE,
    sep="\t"
)

print(
    "Ground truth rows:",
    len(ground_truth)
)


# ============================================================
# DETECT GROUND-TRUTH COLUMNS
# ============================================================

ground_source_column = "source1_entity_id"

ground_match_column = "matched_entity_ids"


if ground_source_column not in ground_truth.columns:

    # fallback detection

    possible_source_columns = [
        "source1_entity_id",
        "entity_id",
        "source_entity_id"
    ]

    for col in possible_source_columns:

        if col in ground_truth.columns:

            ground_source_column = col
            break


if ground_match_column not in ground_truth.columns:

    possible_match_columns = [
        "matched_entity_ids",
        "matched_entity_id",
        "match_entity_id"
    ]

    for col in possible_match_columns:

        if col in ground_truth.columns:

            ground_match_column = col
            break


print(
    "Using ground-truth source column:",
    ground_source_column
)

print(
    "Using ground-truth column:",
    ground_match_column
)


# ============================================================
# BUILD GROUND-TRUTH INDEX
# ============================================================

print()
print("Building ground-truth index...")

ground_truth_index = {}


for _, row in ground_truth.iterrows():

    source_id = str(
        row[ground_source_column]
    )

    value = row[ground_match_column]


    if pd.isna(value):

        ground_truth_index[source_id] = set()

        continue


    # Handle list-like strings
    if isinstance(value, str):

        value = value.strip()

        if not value:

            ground_truth_index[source_id] = set()

            continue


        # Try Python list representation
        try:

            parsed = ast.literal_eval(value)

            if isinstance(parsed, (list, tuple, set)):

                matches = {
                    str(x)
                    for x in parsed
                }

            else:

                matches = {
                    str(parsed)
                }

        except Exception:

            # Handle comma-separated IDs
            if "," in value:

                matches = {
                    x.strip()
                    for x in value.split(",")
                    if x.strip()
                }

            else:

                matches = {
                    value
                }


    elif isinstance(value, (list, tuple, set)):

        matches = {
            str(x)
            for x in value
        }

    else:

        matches = {
            str(value)
        }


    ground_truth_index[source_id] = matches


print(
    "Ground-truth source IDs:",
    len(ground_truth_index)
)


# ============================================================
# VALIDATION
# ============================================================

print()
print("Validating candidate file...")

duplicate_ids = (
    candidates[source_column]
    .astype(str)
    .duplicated()
    .sum()
)

print(
    "Duplicate Source-1 IDs:",
    duplicate_ids
)


def parse_candidates(value):

    if pd.isna(value):

        return []


    value = str(value).strip()

    if not value:

        return []


    # Main V7.4 output format:
    # ID1,ID2,ID3

    if "," in value:

        return [
            x.strip()
            for x in value.split(",")
            if x.strip()
        ]


    # Also support list representations
    try:

        parsed = ast.literal_eval(value)

        if isinstance(parsed, (list, tuple, set)):

            return [
                str(x)
                for x in parsed
            ]

    except Exception:

        pass


    return [value]


candidate_lists = (
    candidates[candidate_column]
    .apply(parse_candidates)
)


empty_rows = (
    candidate_lists
    .apply(len)
    .eq(0)
    .sum()
)

print(
    "Rows with empty candidates:",
    empty_rows
)


# ============================================================
# EVALUATION
# ============================================================

print()
print("Evaluating V7.4 Fast candidate recall...")

recall_counts = {
    k: 0
    for k in TOP_K_VALUES
}

evaluated = 0
missing_ground_truth = 0


for source_id, candidate_list in zip(
    candidates[source_column].astype(str),
    candidate_lists
):

    if source_id not in ground_truth_index:

        missing_ground_truth += 1

        continue


    true_matches = (
        ground_truth_index[source_id]
    )


    if not true_matches:

        continue


    evaluated += 1


    # Remove accidental duplicates while
    # preserving candidate order
    seen = set()

    unique_candidates = []

    for candidate in candidate_list:

        candidate = str(candidate)

        if candidate not in seen:

            seen.add(candidate)

            unique_candidates.append(
                candidate
            )


    for k in TOP_K_VALUES:

        top_candidates = set(
            unique_candidates[:k]
        )

        if top_candidates & true_matches:

            recall_counts[k] += 1


# ============================================================
# RESULTS
# ============================================================

print()
print("=" * 70)
print("V7.4 FAST CANDIDATE RECALL RESULTS")
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
    empty_rows
)


candidate_counts = (
    candidate_lists
    .apply(len)
)

print(
    "Average candidates:",
    round(
        candidate_counts.mean(),
        2
    )
)

print(
    "Maximum candidates:",
    candidate_counts.max()
)


print()
print("Recall results:")


for k in TOP_K_VALUES:

    if evaluated > 0:

        recall = (
            recall_counts[k]
            / evaluated
            * 100
        )

    else:

        recall = 0.0


    print(
        f"Candidate Recall@{k:<3}: "
        f"{recall:.2f}%"
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


print()
print("=" * 70)
print("V7 BASELINE COMPARISON")
print("=" * 70)


for k in TOP_K_VALUES:

    if evaluated > 0:

        recall = (
            recall_counts[k]
            / evaluated
            * 100
        )

    else:

        recall = 0.0


    change = (
        recall
        - baseline[k]
    )


    sign = "+" if change >= 0 else ""


    print(
        f"Recall@{k:<3}: "
        f"V7 = {baseline[k]:.2f}% | "
        f"V7.4 = {recall:.2f}% | "
        f"Change = {sign}{change:.2f} points"
    )


# ============================================================
# DIAGNOSTICS
# ============================================================

print()
print("=" * 70)
print("QUICK DIAGNOSTIC")
print("=" * 70)


if duplicate_ids == 0:

    print(
        "✓ No duplicate Source-1 IDs."
    )

else:

    print(
        "⚠ Duplicate Source-1 IDs:",
        duplicate_ids
    )


if empty_rows == 0:

    print(
        "✓ All candidate rows contain candidates."
    )

else:

    print(
        "⚠ Empty candidate rows:",
        empty_rows
    )


if missing_ground_truth == 0:

    print(
        "✓ All candidate rows have ground truth."
    )

else:

    print(
        "⚠ Missing ground truth rows:",
        missing_ground_truth
    )


if evaluated > 0:

    recall100 = (
        recall_counts[100]
        / evaluated
        * 100
    )

else:

    recall100 = 0.0


print()
print(
    f"Recall@100: {recall100:.2f}%"
)


# ============================================================
# DECISION
# ============================================================

print()
print("=" * 70)
print("V7.4 STATUS")
print("=" * 70)


if recall100 >= 88.0:

    print(
        "✓ V7.4 reaches the V7 baseline."
    )

    print(
        "Next: run LightGBM ranking on V7.4."
    )

elif recall100 >= 86.0:

    print(
        "⚠ V7.4 is close to the V7 baseline."
    )

    print(
        "Do NOT run the full 2.2M rows yet."
    )

    print(
        "Next: inspect blocking recall before scaling."
    )

else:

    print(
        "✗ V7.4 is below the V7 baseline."
    )

    print(
        "Do NOT run the full 2.2M rows."
    )

    print(
        "We need to restore/add a blocking signal."
    )


# ============================================================
# END
# ============================================================

elapsed = time.time() - start_time

print()
print(
    f"Evaluation time: {elapsed:.2f}s"
)

print()
print("=" * 70)
print("DONE!")
print("=" * 70)