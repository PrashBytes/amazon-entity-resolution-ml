import os
import re
import numpy as np
import pandas as pd

from rapidfuzz.fuzz import ratio, token_set_ratio, WRatio
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score


# ============================================================
# CONFIG
# ============================================================

BASE = "dataset/train"

SOURCE1 = f"{BASE}/train_source1.tsv"
SOURCE2 = f"{BASE}/train_source2.tsv"
SOURCE3 = f"{BASE}/train_source3.tsv"
GROUND_TRUTH = f"{BASE}/train_ground_truth.tsv"

MODEL_FILE = "match_model.txt"

# Start small so we can test the ML pipeline quickly.
# We will increase this later.
MAX_SOURCE1 = 10000

# Maximum candidates used per Source 1 record during training
MAX_CANDIDATES = 100

RANDOM_STATE = 42


# ============================================================
# NORMALIZATION
# ============================================================

def normalize(text):
    if pd.isna(text):
        return ""

    text = str(text).lower()

    # Replace punctuation with spaces
    text = re.sub(r"[^a-z0-9]+", " ", text)

    # Collapse spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


def token_set(text):
    return set(text.split())


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 60)
print("LOADING TRAINING DATA")
print("=" * 60)

s1 = pd.read_csv(SOURCE1, sep="\t")
s2 = pd.read_csv(SOURCE2, sep="\t")
s3 = pd.read_csv(SOURCE3, sep="\t")
gt = pd.read_csv(GROUND_TRUTH, sep="\t")

print("Source 1:", len(s1))
print("Source 2:", len(s2))
print("Source 3:", len(s3))
print("Ground truth:", len(gt))


# ============================================================
# NORMALIZE
# ============================================================

print("\nNormalizing data...")

for df in [s1, s2, s3]:

    df["name_norm"] = df["business_name"].fillna("").map(normalize)
    df["address_norm"] = df["business_address"].fillna("").map(normalize)
    df["country_norm"] = df["country"].fillna("").map(normalize)


# ============================================================
# LOOKUP TABLES
# ============================================================

print("Creating lookup tables...")

s2_records = {
    row.entity_id: row
    for row in s2.itertuples(index=False)
}

s3_records = {
    row.entity_id: row
    for row in s3.itertuples(index=False)
}


# ============================================================
# GROUND TRUTH
# ============================================================

print("Preparing ground truth...")

truth = {}

for row in gt.itertuples(index=False):

    s1_id = row.source1_entity_id

    value = row.matched_entity_ids

    if pd.isna(value) or str(value).strip() == "":
        truth[s1_id] = set()
    else:
        truth[s1_id] = set(
            x.strip()
            for x in str(value).split(",")
            if x.strip()
        )

print("Ground-truth mappings:", len(truth))


# ============================================================
# BLOCKING
# ============================================================

print("\nBuilding blocking indexes...")

name_index = {}
address_index = {}
country_index = {}

for record in list(s2.itertuples(index=False)) + list(s3.itertuples(index=False)):

    entity_id = record.entity_id

    name = record.name_norm
    address = record.address_norm
    country = record.country_norm

    # First 6 characters of normalized name
    if name:
        key = name[:6]

        name_index.setdefault(key, []).append(entity_id)

    # First 8 characters of address
    if address:
        key = address[:8]

        address_index.setdefault(key, []).append(entity_id)

    # Country
    if country:
        country_index.setdefault(country, []).append(entity_id)


print("Name blocks:", len(name_index))
print("Address blocks:", len(address_index))
print("Country blocks:", len(country_index))


# ============================================================
# FEATURE CREATION
# ============================================================

def get_record(entity_id):

    if entity_id.startswith("S2-"):
        return s2_records.get(entity_id)

    return s3_records.get(entity_id)


def make_features(source_record, candidate_record):

    name1 = source_record.name_norm
    name2 = candidate_record.name_norm

    addr1 = source_record.address_norm
    addr2 = candidate_record.address_norm

    country1 = source_record.country_norm
    country2 = candidate_record.country_norm

    # --------------------------------------------------------
    # NAME FEATURES
    # --------------------------------------------------------

    name_ratio = ratio(name1, name2) / 100.0

    name_token = token_set_ratio(name1, name2) / 100.0

    name_wratio = WRatio(name1, name2) / 100.0

    exact_name = int(name1 == name2 and name1 != "")

    # --------------------------------------------------------
    # ADDRESS FEATURES
    # --------------------------------------------------------

    address_ratio = ratio(addr1, addr2) / 100.0

    address_token = token_set_ratio(addr1, addr2) / 100.0

    address_wratio = WRatio(addr1, addr2) / 100.0

    exact_address = int(addr1 == addr2 and addr1 != "")

    # --------------------------------------------------------
    # TOKEN FEATURES
    # --------------------------------------------------------

    name_tokens_1 = token_set(name1)
    name_tokens_2 = token_set(name2)

    if name_tokens_1 or name_tokens_2:

        intersection = len(name_tokens_1 & name_tokens_2)
        union = len(name_tokens_1 | name_tokens_2)

        name_jaccard = intersection / max(union, 1)

    else:

        name_jaccard = 0.0

    # Address token overlap

    addr_tokens_1 = token_set(addr1)
    addr_tokens_2 = token_set(addr2)

    if addr_tokens_1 or addr_tokens_2:

        intersection = len(addr_tokens_1 & addr_tokens_2)
        union = len(addr_tokens_1 | addr_tokens_2)

        address_jaccard = intersection / max(union, 1)

    else:

        address_jaccard = 0.0

    # --------------------------------------------------------
    # LENGTH FEATURES
    # --------------------------------------------------------

    name_length_ratio = (
        min(len(name1), len(name2)) /
        max(len(name1), len(name2), 1)
    )

    address_length_ratio = (
        min(len(addr1), len(addr2)) /
        max(len(addr1), len(addr2), 1)
    )

    # --------------------------------------------------------
    # COUNTRY
    # --------------------------------------------------------

    country_match = int(
        country1 != "" and
        country1 == country2
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
# CANDIDATE GENERATION FOR TRAINING
# ============================================================

print("\nGenerating training pairs...")

training_rows = []

processed = 0

# Only use Source 1 rows that have ground truth.
train_s1 = s1[
    s1["entity_id"].isin(truth.keys())
].head(MAX_SOURCE1)

for source_record in train_s1.itertuples(index=False):

    source_id = source_record.entity_id

    true_ids = truth.get(source_id, set())

    candidates = set()

    # --------------------------------------------------------
    # NAME BLOCK
    # --------------------------------------------------------

    name = source_record.name_norm

    if name:

        key = name[:6]

        candidates.update(
            name_index.get(key, [])
        )

    # --------------------------------------------------------
    # ADDRESS BLOCK
    # --------------------------------------------------------

    address = source_record.address_norm

    if address:

        key = address[:8]

        candidates.update(
            address_index.get(key, [])
        )

    # --------------------------------------------------------
    # COUNTRY BLOCK
    # --------------------------------------------------------

    country = source_record.country_norm

    if country:

        country_candidates = country_index.get(
            country, []
        )

        # Don't add every country record.
        # Add only a limited sample.
        candidates.update(
            country_candidates[:200]
        )

    # --------------------------------------------------------
    # ALWAYS INCLUDE POSITIVE MATCHES
    # --------------------------------------------------------

    candidates.update(true_ids)

    # Remove anything that isn't S2/S3
    candidates = {
        x for x in candidates
        if x.startswith("S2-") or x.startswith("S3-")
    }

    # --------------------------------------------------------
    # LIMIT NEGATIVE CANDIDATES
    # --------------------------------------------------------

    candidates = list(candidates)

    if len(candidates) > MAX_CANDIDATES:

        positives = [
            x for x in candidates
            if x in true_ids
        ]

        negatives = [
            x for x in candidates
            if x not in true_ids
        ]

        # Deterministic sampling
        rng = np.random.default_rng(
            RANDOM_STATE + processed
        )

        keep_count = max(
            0,
            MAX_CANDIDATES - len(positives)
        )

        if len(negatives) > keep_count:

            negatives = list(
                rng.choice(
                    negatives,
                    size=keep_count,
                    replace=False
                )
            )

        candidates = positives + negatives

    # --------------------------------------------------------
    # CREATE FEATURES
    # --------------------------------------------------------

    for candidate_id in candidates:

        candidate_record = get_record(candidate_id)

        if candidate_record is None:
            continue

        features = make_features(
            source_record,
            candidate_record
        )

        label = int(
            candidate_id in true_ids
        )

        training_rows.append(
            features + [label]
        )

    processed += 1

    if processed % 500 == 0:

        print(
            f"Processed {processed}/{len(train_s1)}"
        )


# ============================================================
# DATAFRAME
# ============================================================

columns = FEATURE_NAMES + ["label"]

training_df = pd.DataFrame(
    training_rows,
    columns=columns
)

print("\n" + "=" * 60)
print("TRAINING DATASET")
print("=" * 60)

print("Training pairs:", len(training_df))
print(
    "Positive matches:",
    int(training_df["label"].sum())
)

print(
    "Negative pairs:",
    int((training_df["label"] == 0).sum())
)


# ============================================================
# TRAIN / VALIDATION SPLIT
# ============================================================

X = training_df[FEATURE_NAMES]
y = training_df["label"]

X_train, X_valid, y_train, y_valid = train_test_split(

    X,
    y,

    test_size=0.20,

    random_state=RANDOM_STATE,

    stratify=y
)


# ============================================================
# LIGHTGBM
# ============================================================

print("\nTraining LightGBM...")

model = LGBMClassifier(

    objective="binary",

    n_estimators=500,

    learning_rate=0.05,

    num_leaves=31,

    max_depth=-1,

    subsample=0.8,

    colsample_bytree=0.8,

    random_state=RANDOM_STATE,

    n_jobs=-1,

    verbosity=-1
)


model.fit(
    X_train,
    y_train
)


# ============================================================
# VALIDATION
# ============================================================

print("\n" + "=" * 60)
print("MODEL VALIDATION")
print("=" * 60)

probabilities = model.predict_proba(
    X_valid
)[:, 1]


# Try several thresholds
for threshold in [0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]:

    predictions = (
        probabilities >= threshold
    ).astype(int)

    precision = precision_score(
        y_valid,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_valid,
        predictions,
        zero_division=0
    )

    # F0.5
    beta = 0.5

    f05 = (
        (1 + beta**2)
        * precision
        * recall
        /
        (
            beta**2 * precision
            + recall
            + 1e-12
        )
    )

    print(
        f"Threshold {threshold:.2f} | "
        f"Precision {precision:.4f} | "
        f"Recall {recall:.4f} | "
        f"F0.5 {f05:.4f}"
    )


# ============================================================
# SAVE MODEL
# ============================================================

model.booster_.save_model(
    MODEL_FILE
)

print("\n" + "=" * 60)
print("MODEL SAVED")
print("=" * 60)

print(MODEL_FILE)

print("\nDONE!")