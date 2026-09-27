import re
import numpy as np
import pandas as pd

from rapidfuzz.fuzz import ratio, token_set_ratio, WRatio
from lightgbm import LGBMClassifier
from sklearn.metrics import precision_score, recall_score


# ============================================================
# CONFIG
# ============================================================

BASE = "dataset/train"

SOURCE1 = f"{BASE}/train_source1.tsv"
SOURCE2 = f"{BASE}/train_source2.tsv"
SOURCE3 = f"{BASE}/train_source3.tsv"
GROUND_TRUTH = f"{BASE}/train_ground_truth.tsv"

MODEL_FILE = "match_model_v2.txt"

MAX_SOURCE1 = 20000
MAX_CANDIDATES = 100

RANDOM_STATE = 42


# ============================================================
# NORMALIZATION
# ============================================================

def normalize(text):

    if pd.isna(text):
        return ""

    text = str(text).lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text


def token_set(text):
    return set(text.split())


# ============================================================
# LOAD
# ============================================================

print("=" * 60)
print("LOADING DATA")
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

print("\nNormalizing...")

for df in [s1, s2, s3]:

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
# LOOKUPS
# ============================================================

print("Creating lookups...")

s2_records = {
    row.entity_id: row
    for row in s2.itertuples(index=False)
}

s3_records = {
    row.entity_id: row
    for row in s3.itertuples(index=False)
}


def get_record(entity_id):

    if entity_id.startswith("S2-"):
        return s2_records.get(entity_id)

    if entity_id.startswith("S3-"):
        return s3_records.get(entity_id)

    return None


# ============================================================
# GROUND TRUTH
# ============================================================

print("Preparing ground truth...")

truth = {}

for row in gt.itertuples(index=False):

    entity_id = row.source1_entity_id
    matches = row.matched_entity_ids

    if pd.isna(matches) or str(matches).strip() == "":
        truth[entity_id] = set()

    else:
        truth[entity_id] = set(
            x.strip()
            for x in str(matches).split(",")
            if x.strip()
        )

print("Ground truth mappings:", len(truth))


# ============================================================
# BLOCKING INDEXES
# ============================================================

print("\nBuilding indexes...")

name_index = {}
address_index = {}
country_index = {}

for record in list(s2.itertuples(index=False)) + list(
    s3.itertuples(index=False)
):

    entity_id = record.entity_id

    name = record.name_norm
    address = record.address_norm
    country = record.country_norm

    if name:

        key = name[:6]

        name_index.setdefault(
            key, []
        ).append(entity_id)

    if address:

        key = address[:8]

        address_index.setdefault(
            key, []
        ).append(entity_id)

    if country:

        country_index.setdefault(
            country, []
        ).append(entity_id)


print("Indexes ready!")


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


def make_features(a, b):

    name1 = a.name_norm
    name2 = b.name_norm

    addr1 = a.address_norm
    addr2 = b.address_norm

    country1 = a.country_norm
    country2 = b.country_norm

    # NAME

    name_ratio = ratio(
        name1, name2
    ) / 100.0

    name_token = token_set_ratio(
        name1, name2
    ) / 100.0

    name_wratio = WRatio(
        name1, name2
    ) / 100.0

    exact_name = int(
        name1 != "" and
        name1 == name2
    )

    # ADDRESS

    address_ratio = ratio(
        addr1, addr2
    ) / 100.0

    address_token = token_set_ratio(
        addr1, addr2
    ) / 100.0

    address_wratio = WRatio(
        addr1, addr2
    ) / 100.0

    exact_address = int(
        addr1 != "" and
        addr1 == addr2
    )

    # NAME JACCARD

    n1 = token_set(name1)
    n2 = token_set(name2)

    name_jaccard = (
        len(n1 & n2) /
        max(len(n1 | n2), 1)
    )

    # ADDRESS JACCARD

    a1 = token_set(addr1)
    a2 = token_set(addr2)

    address_jaccard = (
        len(a1 & a2) /
        max(len(a1 | a2), 1)
    )

    # LENGTH

    name_length_ratio = (
        min(len(name1), len(name2))
        /
        max(len(name1), len(name2), 1)
    )

    address_length_ratio = (
        min(len(addr1), len(addr2))
        /
        max(len(addr1), len(addr2), 1)
    )

    # COUNTRY

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


# ============================================================
# CANDIDATE GENERATION
# ============================================================

def generate_pairs(source_rows):

    rows = []

    processed = 0

    for source in source_rows:

        source_id = source.entity_id

        true_ids = truth.get(
            source_id,
            set()
        )

        candidates = set()

        # NAME BLOCK

        if source.name_norm:

            key = source.name_norm[:6]

            candidates.update(
                name_index.get(key, [])
            )

        # ADDRESS BLOCK

        if source.address_norm:

            key = source.address_norm[:8]

            candidates.update(
                address_index.get(key, [])
            )

        # COUNTRY BLOCK

        if source.country_norm:

            candidates.update(
                country_index.get(
                    source.country_norm,
                    []
                )[:200]
            )

        # IMPORTANT:
        # Always keep true matches.

        candidates.update(true_ids)

        candidates = [
            x for x in candidates
            if x.startswith("S2-")
            or x.startswith("S3-")
        ]

        # Limit candidates while preserving positives.

        if len(candidates) > MAX_CANDIDATES:

            positives = [
                x for x in candidates
                if x in true_ids
            ]

            negatives = [
                x for x in candidates
                if x not in true_ids
            ]

            keep = max(
                0,
                MAX_CANDIDATES - len(positives)
            )

            if len(negatives) > keep:

                rng = np.random.default_rng(
                    RANDOM_STATE + processed
                )

                negatives = list(
                    rng.choice(
                        negatives,
                        size=keep,
                        replace=False
                    )
                )

            candidates = positives + negatives

        # FEATURES

        for candidate_id in candidates:

            candidate = get_record(
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

            rows.append(
                features + [label]
            )

        processed += 1

        if processed % 1000 == 0:

            print(
                f"Processed {processed}/"
                f"{len(source_rows)}"
            )

    return pd.DataFrame(
        rows,
        columns=FEATURE_NAMES + ["label"]
    )


# ============================================================
# SELECT SOURCE 1 ENTITIES
# ============================================================

available = s1[
    s1.entity_id.isin(truth.keys())
].copy()

available = available.head(
    MAX_SOURCE1
)

# THIS IS THE IMPORTANT DIFFERENCE:
# Split by Source 1 ENTITY, not candidate pair.

rng = np.random.default_rng(
    RANDOM_STATE
)

indices = np.arange(
    len(available)
)

rng.shuffle(indices)

split = int(
    len(indices) * 0.80
)

train_indices = indices[:split]
valid_indices = indices[split:]

train_s1 = available.iloc[
    train_indices
]

valid_s1 = available.iloc[
    valid_indices
]

print("\n" + "=" * 60)
print("ENTITY-LEVEL SPLIT")
print("=" * 60)

print(
    "Training Source 1:",
    len(train_s1)
)

print(
    "Validation Source 1:",
    len(valid_s1)
)


# ============================================================
# GENERATE TRAINING PAIRS
# ============================================================

print("\nGenerating TRAINING pairs...")

train_df = generate_pairs(
    list(train_s1.itertuples(index=False))
)

print(
    "Training pairs:",
    len(train_df)
)

print(
    "Training positives:",
    int(train_df.label.sum())
)


# ============================================================
# GENERATE VALIDATION PAIRS
# ============================================================

print("\nGenerating VALIDATION pairs...")

valid_df = generate_pairs(
    list(valid_s1.itertuples(index=False))
)

print(
    "Validation pairs:",
    len(valid_df)
)

print(
    "Validation positives:",
    int(valid_df.label.sum())
)


# ============================================================
# TRAIN MODEL
# ============================================================

print("\n" + "=" * 60)
print("TRAINING LIGHTGBM")
print("=" * 60)

X_train = train_df[
    FEATURE_NAMES
]

y_train = train_df[
    "label"
]

X_valid = valid_df[
    FEATURE_NAMES
]

y_valid = valid_df[
    "label"
]

model = LGBMClassifier(

    objective="binary",

    n_estimators=700,

    learning_rate=0.04,

    num_leaves=31,

    max_depth=-1,

    min_child_samples=30,

    subsample=0.8,

    colsample_bytree=0.9,

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
print("HONEST ENTITY-LEVEL VALIDATION")
print("=" * 60)

probabilities = model.predict_proba(
    X_valid
)[:, 1]


# ============================================================
# THRESHOLD SEARCH
# ============================================================

best_threshold = 0
best_f05 = 0

for threshold in np.arange(
    0.30,
    0.96,
    0.05
):

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

    beta = 0.5

    f05 = (
        (1 + beta ** 2)
        * precision
        * recall
        /
        (
            beta ** 2 * precision
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

    if f05 > best_f05:

        best_f05 = f05
        best_threshold = threshold


# ============================================================
# SAVE
# ============================================================

model.booster_.save_model(
    MODEL_FILE
)

print("\n" + "=" * 60)
print("BEST RESULT")
print("=" * 60)

print(
    f"Best threshold: "
    f"{best_threshold:.2f}"
)

print(
    f"Best F0.5: "
    f"{best_f05:.4f}"
)

print(
    f"Model saved: "
    f"{MODEL_FILE}"
)

print("\nDONE!")