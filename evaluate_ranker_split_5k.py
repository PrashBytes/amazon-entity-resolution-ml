import os
import re
import time
import warnings
import numpy as np
import pandas as pd

from rapidfuzz.fuzz import ratio, token_set_ratio, WRatio
from lightgbm import LGBMRanker


# ============================================================
# CONFIG
# ============================================================

BASE = "dataset/train"

SOURCE1_FILE = f"{BASE}/train_source1.tsv"
SOURCE2_FILE = f"{BASE}/train_source2.tsv"
SOURCE3_FILE = f"{BASE}/train_source3.tsv"
GROUND_TRUTH_FILE = f"{BASE}/train_ground_truth.tsv"

# IMPORTANT:
# This must be your V7 candidate file.
CANDIDATE_FILE = "v7_candidates.tsv"

MODEL_FILE = "v7_lgbm_5k_split_model.txt"
OUTPUT_FILE = "v7_ranked_5k_test.tsv"

# ============================================================
# 5K VALIDATION
# ============================================================

TOTAL_ROWS = 5000
TRAIN_ROWS = 4000
TEST_ROWS = 1000

RANDOM_STATE = 42
MAX_CANDIDATES = 100


# ============================================================
# SUPPRESS HARMLESS FEATURE-NAME WARNING
# ============================================================

warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names"
)


# ============================================================
# FEATURES
# ============================================================

FEATURE_NAMES = [
    "name_ratio",
    "name_token",
    "name_wratio",
    "exact_name",

    "address_ratio",
    "address_token",
    "address_wratio",
    "exact_address",

    "name_jaccard",
    "address_jaccard",

    "name_length_ratio",
    "address_length_ratio",

    "country_match"
]


# ============================================================
# NORMALIZATION
# ============================================================

def normalize(text):

    if pd.isna(text):
        return ""

    text = str(text).lower()

    text = re.sub(
        r"[^a-z0-9]+",
        " ",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    return text


def token_set(text):

    if not text:
        return set()

    return set(text.split())


def jaccard(a, b):

    if not a and not b:
        return 0.0

    union = a | b

    if not union:
        return 0.0

    return len(a & b) / len(union)


# ============================================================
# FEATURES FOR ONE PAIR
# ============================================================

def make_features(source, candidate):

    name1 = source["name_norm"]
    name2 = candidate["name_norm"]

    address1 = source["address_norm"]
    address2 = candidate["address_norm"]

    country1 = source["country_norm"]
    country2 = candidate["country_norm"]

    # --------------------------------------------------------
    # NAME
    # --------------------------------------------------------

    name_ratio = ratio(
        name1,
        name2
    ) / 100.0

    name_token = token_set_ratio(
        name1,
        name2
    ) / 100.0

    name_wratio = WRatio(
        name1,
        name2
    ) / 100.0

    exact_name = int(
        bool(name1)
        and name1 == name2
    )

    # --------------------------------------------------------
    # ADDRESS
    # --------------------------------------------------------

    address_ratio = ratio(
        address1,
        address2
    ) / 100.0

    address_token = token_set_ratio(
        address1,
        address2
    ) / 100.0

    address_wratio = WRatio(
        address1,
        address2
    ) / 100.0

    exact_address = int(
        bool(address1)
        and address1 == address2
    )

    # --------------------------------------------------------
    # JACCARD
    # --------------------------------------------------------

    name_jaccard = jaccard(
        token_set(name1),
        token_set(name2)
    )

    address_jaccard = jaccard(
        token_set(address1),
        token_set(address2)
    )

    # --------------------------------------------------------
    # LENGTH RATIOS
    # --------------------------------------------------------

    name_length_ratio = (
        min(len(name1), len(name2))
        /
        max(len(name1), len(name2), 1)
    )

    address_length_ratio = (
        min(len(address1), len(address2))
        /
        max(len(address1), len(address2), 1)
    )

    # --------------------------------------------------------
    # COUNTRY
    # --------------------------------------------------------

    country_match = int(
        bool(country1)
        and country1 == country2
    )

    return [
        name_ratio,
        name_token,
        name_wratio,
        exact_name,

        address_ratio,
        address_token,
        address_wratio,
        exact_address,

        name_jaccard,
        address_jaccard,

        name_length_ratio,
        address_length_ratio,

        country_match
    ]


# ============================================================
# START
# ============================================================

start_time = time.time()

print("=" * 70)
print("V7 + LIGHTGBM 5K PROPER TRAIN / TEST SPLIT")
print("=" * 70)


# ============================================================
# CHECK FILES
# ============================================================

required_files = [
    CANDIDATE_FILE,
    SOURCE1_FILE,
    SOURCE2_FILE,
    SOURCE3_FILE,
    GROUND_TRUTH_FILE
]

for file in required_files:

    if not os.path.exists(file):

        raise FileNotFoundError(
            f"\nMissing file:\n{file}\n"
        )


# ============================================================
# LOAD CANDIDATES
# ============================================================

print("\nLoading candidate file...")

candidates_df = pd.read_csv(
    CANDIDATE_FILE,
    sep="\t"
)

print(
    "Candidate rows:",
    len(candidates_df)
)

print(
    "Candidate columns:",
    candidates_df.columns.tolist()
)


# ============================================================
# MAKE SURE WE HAVE 5K
# ============================================================

if len(candidates_df) < TOTAL_ROWS:

    raise RuntimeError(
        "\n"
        + "=" * 70
        + "\nNOT ENOUGH CANDIDATE ROWS\n"
        + "=" * 70
        + f"\nCurrent candidate rows: {len(candidates_df)}"
        + f"\nRequired candidate rows: {TOTAL_ROWS}"
        + "\n\n"
        + "Your current V7 candidate file is still only "
        + "1,000 rows.\n"
        + "Generate V7 candidates for 5,000 rows first.\n"
    )


# ============================================================
# TAKE EXACTLY 5K
# ============================================================

candidates_df = candidates_df.iloc[
    :TOTAL_ROWS
].copy()


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

print("\nCreating train/test split...")

rng = np.random.default_rng(
    RANDOM_STATE
)

indices = np.arange(
    TOTAL_ROWS
)

rng.shuffle(indices)

train_indices = indices[
    :TRAIN_ROWS
]

test_indices = indices[
    TRAIN_ROWS:
]

train_candidates = candidates_df.iloc[
    train_indices
].reset_index(drop=True)

test_candidates = candidates_df.iloc[
    test_indices
].reset_index(drop=True)

print(
    "Training rows:",
    len(train_candidates)
)

print(
    "Testing rows:",
    len(test_candidates)
)

print(
    "Random seed:",
    RANDOM_STATE
)


# ============================================================
# LOAD GROUND TRUTH
# ============================================================

print("\nLoading ground truth...")

gt = pd.read_csv(
    GROUND_TRUTH_FILE,
    sep="\t"
)

print(
    "Ground-truth IDs:",
    len(gt)
)


# ============================================================
# BUILD GROUND-TRUTH INDEX
# ============================================================

truth = {}

for row in gt.itertuples(
    index=False
):

    source_id = str(
        row.source1_entity_id
    )

    matched = row.matched_entity_ids

    if pd.isna(matched):

        truth[source_id] = set()

    else:

        matched = str(
            matched
        ).strip()

        if not matched:

            truth[source_id] = set()

        else:

            truth[source_id] = set(
                x.strip()
                for x in matched.split(",")
                if x.strip()
            )


# ============================================================
# LOAD SOURCES
# ============================================================

print("\nLoading Source 1...")

s1 = pd.read_csv(
    SOURCE1_FILE,
    sep="\t"
)

print(
    "Source 1:",
    len(s1)
)


print("\nLoading Source 2...")

s2 = pd.read_csv(
    SOURCE2_FILE,
    sep="\t"
)

print(
    "Source 2:",
    len(s2)
)


print("\nLoading Source 3...")

s3 = pd.read_csv(
    SOURCE3_FILE,
    sep="\t"
)

print(
    "Source 3:",
    len(s3)
)


# ============================================================
# NORMALIZE SOURCES
# ============================================================

print("\nNormalizing sources...")

for df in [
    s1,
    s2,
    s3
]:

    df["name_norm"] = (
        df["business_name"]
        .fillna("")
        .map(normalize)
    )

    df["address_norm"] = (
        df["business_address"]
        .fillna("")
        .map(normalize)
    )

    df["country_norm"] = (
        df["country"]
        .fillna("")
        .map(normalize)
    )


# ============================================================
# SOURCE 1 LOOKUP
# ============================================================

print("\nBuilding Source-1 lookup...")

s1_lookup = {
    str(row.entity_id): row
    for row in s1.itertuples(
        index=False
    )
}


# ============================================================
# CANDIDATE ID SET
# ============================================================

print("\nCollecting candidate IDs...")

candidate_ids = set()

for value in candidates_df[
    "candidate_entity_ids"
].fillna(""):

    if not value:
        continue

    for entity_id in str(
        value
    ).split(","):

        entity_id = entity_id.strip()

        if entity_id:

            candidate_ids.add(
                entity_id
            )


print(
    "Candidate IDs needed:",
    len(candidate_ids)
)


# ============================================================
# BUILD CANDIDATE LOOKUP
# ============================================================

print("\nBuilding candidate lookup...")

s2_lookup = {
    str(row.entity_id): row
    for row in s2.itertuples(
        index=False
    )
}

s3_lookup = {
    str(row.entity_id): row
    for row in s3.itertuples(
        index=False
    )
}

candidate_lookup = {}

for entity_id in candidate_ids:

    if entity_id.startswith("S2-"):

        row = s2_lookup.get(
            entity_id
        )

    elif entity_id.startswith("S3-"):

        row = s3_lookup.get(
            entity_id
        )

    else:

        row = None

    if row is not None:

        candidate_lookup[
            entity_id
        ] = row


print(
    "Candidate lookup:",
    len(candidate_lookup)
)


# ============================================================
# BUILD TRAINING DATA
# ============================================================

print("\n" + "=" * 70)
print("BUILDING TRAINING DATA")
print("=" * 70)

training_X = []
training_y = []
training_groups = []

processed = 0

for _, row in train_candidates.iterrows():

    source_id = str(
        row.source1_entity_id
    )

    source = s1_lookup.get(
        source_id
    )

    if source is None:
        continue

    true_ids = truth.get(
        source_id,
        set()
    )

    candidate_string = row.candidate_entity_ids

    if pd.isna(candidate_string):

        candidate_string = ""

    candidate_list = [
        x.strip()
        for x in str(
            candidate_string
        ).split(",")
        if x.strip()
    ]

    group_count = 0

    for candidate_id in candidate_list:

        candidate = candidate_lookup.get(
            candidate_id
        )

        if candidate is None:
            continue

        features = make_features(
            source,
            candidate
        )

        label = int(
            candidate_id in true_ids
        )

        training_X.append(
            features
        )

        training_y.append(
            label
        )

        group_count += 1

    if group_count > 0:

        training_groups.append(
            group_count
        )

    processed += 1

    if processed % 500 == 0:

        print(
            f"Training rows processed: "
            f"{processed}/{TRAIN_ROWS}"
        )


training_X = pd.DataFrame(
    training_X,
    columns=FEATURE_NAMES
)

training_y = np.asarray(
    training_y,
    dtype=np.int32
)

training_groups = np.asarray(
    training_groups,
    dtype=np.int32
)

print(
    "\nTraining examples:",
    len(training_X)
)

print(
    "Positive matches:",
    int(training_y.sum())
)

print(
    "Training groups:",
    len(training_groups)
)


# ============================================================
# SAFETY CHECK
# ============================================================

if len(training_X) == 0:

    raise RuntimeError(
        "No training examples were generated."
    )


if training_y.sum() == 0:

    raise RuntimeError(
        "No positive matches found in training data."
    )


# ============================================================
# TRAIN LIGHTGBM RANKER
# ============================================================

print("\n" + "=" * 70)
print("TRAINING LIGHTGBM RANKER")
print("=" * 70)

model = LGBMRanker(

    objective="lambdarank",

    metric="ndcg",

    n_estimators=500,

    learning_rate=0.035,

    num_leaves=31,

    max_depth=-1,

    min_child_samples=20,

    subsample=0.85,

    colsample_bytree=0.9,

    random_state=RANDOM_STATE,

    n_jobs=-1,

    verbosity=-1
)


model.fit(
    training_X,
    training_y,
    group=training_groups
)


model.booster_.save_model(
    MODEL_FILE
)

print(
    "\nModel saved:",
    MODEL_FILE
)


# ============================================================
# RANK UNSEEN TEST ROWS
# ============================================================

print("\n" + "=" * 70)
print("RANKING UNSEEN TEST ROWS")
print("=" * 70)

ranked_results = []

test_processed = 0

for _, row in test_candidates.iterrows():

    source_id = str(
        row.source1_entity_id
    )

    source = s1_lookup.get(
        source_id
    )

    if source is None:
        continue

    candidate_string = row.candidate_entity_ids

    if pd.isna(candidate_string):

        candidate_string = ""

    candidate_list = [
        x.strip()
        for x in str(
            candidate_string
        ).split(",")
        if x.strip()
    ]

    valid_ids = []
    feature_rows = []

    for candidate_id in candidate_list:

        candidate = candidate_lookup.get(
            candidate_id
        )

        if candidate is None:
            continue

        valid_ids.append(
            candidate_id
        )

        feature_rows.append(
            make_features(
                source,
                candidate
            )
        )

    if not feature_rows:

        ranked_results.append({
            "source1_entity_id":
                source_id,

            "candidate_entity_ids":
                ""
        })

        continue

    X_test = pd.DataFrame(
        feature_rows,
        columns=FEATURE_NAMES
    )

    scores = model.predict(
        X_test
    )

    ranked = sorted(
        zip(
            valid_ids,
            scores
        ),
        key=lambda x: x[1],
        reverse=True
    )

    ranked_ids = [
        entity_id
        for entity_id, score
        in ranked
    ]

    ranked_results.append({

        "source1_entity_id":
            source_id,

        "candidate_entity_ids":
            ",".join(ranked_ids)
    })

    test_processed += 1

    if test_processed % 100 == 0:

        print(
            f"Ranked test rows: "
            f"{test_processed}/{TEST_ROWS}"
        )


ranked_df = pd.DataFrame(
    ranked_results
)


ranked_df.to_csv(
    OUTPUT_FILE,
    sep="\t",
    index=False
)


# ============================================================
# EVALUATION
# ============================================================

print("\n" + "=" * 70)
print("UNSEEN TEST RESULTS")
print("=" * 70)

evaluated = 0

hits = {
    1: 0,
    5: 0,
    10: 0,
    20: 0,
    50: 0,
    100: 0
}

for _, row in ranked_df.iterrows():

    source_id = str(
        row.source1_entity_id
    )

    true_ids = truth.get(
        source_id,
        set()
    )

    if not true_ids:
        continue

    candidate_string = row.candidate_entity_ids

    if pd.isna(candidate_string):

        candidate_string = ""

    ranked_ids = [
        x.strip()
        for x in str(
            candidate_string
        ).split(",")
        if x.strip()
    ]

    if not ranked_ids:
        continue

    evaluated += 1

    for k in hits:

        top_k = ranked_ids[
            :k
        ]

        if any(
            entity_id in true_ids
            for entity_id in top_k
        ):

            hits[k] += 1


print(
    "\nTest rows evaluated:",
    evaluated
)

for k in [
    1,
    5,
    10,
    20,
    50,
    100
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

        recall = 0.0

    print(
        f"Unseen Recall@{k:<3}: "
        f"{recall:.2f}%"
    )


# ============================================================
# FINAL
# ============================================================

elapsed = (
    time.time()
    - start_time
)

print("\n" + "=" * 70)
print("5K SPLIT EVALUATION COMPLETE")
print("=" * 70)

print(
    "Training rows:",
    TRAIN_ROWS
)

print(
    "Unseen test rows:",
    TEST_ROWS
)

print(
    "Ranked rows:",
    len(ranked_df)
)

print(
    "Evaluated rows:",
    evaluated
)

print(
    "Saved:",
    OUTPUT_FILE
)

print(
    f"Total runtime: {elapsed:.1f}s"
)

print("=" * 70)