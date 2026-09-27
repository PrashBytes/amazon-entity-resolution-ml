import pandas as pd
import numpy as np
import lightgbm as lgb
from rapidfuzz.fuzz import ratio, token_set_ratio
import re
import time

# ============================================================
# CONFIG
# ============================================================

CANDIDATE_FILE = "v6_candidates_improved.csv"
GT_FILE = "dataset/train/train_ground_truth.tsv"

SOURCE1_FILE = "dataset/train/train_source1.tsv"
SOURCE2_FILE = "dataset/train/train_source2.tsv"
SOURCE3_FILE = "dataset/train/train_source3.tsv"

MODEL_FILE = "v7_lgbm_split_model.txt"
OUTPUT_FILE = "v7_ranked_split_test.tsv"

TOTAL_ROWS = 1000
TRAIN_ROWS = 800
TEST_ROWS = 200

TOP_K = 100

SEED = 42


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("V7 + LIGHTGBM PROPER TRAIN / TEST SPLIT")
print("=" * 70)

start_total = time.time()


# ============================================================
# NORMALIZATION
# ============================================================

def normalize_text(x):

    if pd.isna(x):
        return ""

    x = str(x).lower()

    x = x.replace("&", " and ")

    x = re.sub(r"[^a-z0-9\s]", " ", x)

    x = re.sub(r"\s+", " ", x).strip()

    return x


def normalize_country(x):

    if pd.isna(x):
        return ""

    return str(x).lower().strip()


def extract_number(x):

    if not x:
        return ""

    m = re.search(r"\b\d+[a-zA-Z]?\b", x)

    return m.group(0) if m else ""


def first_word(x):

    if not x:
        return ""

    return x.split()[0]


def last_word(x):

    if not x:
        return ""

    return x.split()[-1]


def prefix(x, n):

    return x[:n] if x else ""


def suffix(x, n):

    return x[-n:] if x else ""


# ============================================================
# SIMILARITY
# ============================================================

def similarity(a, b):

    if not a or not b:
        return 0.0

    if a == b:
        return 100.0

    return ratio(a, b)


def token_similarity(a, b):

    if not a or not b:
        return 0.0

    return token_set_ratio(a, b)


# ============================================================
# FEATURE GENERATION
# ============================================================

def make_features(source, candidate):

    sn = source["name"]
    sa = source["address"]
    sc = source["country"]

    cn = candidate["name"]
    ca = candidate["address"]
    cc = candidate["country"]

    # NAME

    name_exact = int(
        bool(sn) and bool(cn) and sn == cn
    )

    name_ratio = similarity(sn, cn)

    name_token = token_similarity(sn, cn)

    name_first_match = int(
        bool(source["name_first"])
        and source["name_first"]
        == candidate["name_first"]
    )

    name_last_match = int(
        bool(source["name_last"])
        and source["name_last"]
        == candidate["name_last"]
    )

    name_prefix4_match = int(
        bool(source["name_prefix4"])
        and source["name_prefix4"]
        == candidate["name_prefix4"]
    )

    name_prefix6_match = int(
        bool(source["name_prefix6"])
        and source["name_prefix6"]
        == candidate["name_prefix6"]
    )

    name_suffix_match = int(
        bool(source["name_suffix3"])
        and source["name_suffix3"]
        == candidate["name_suffix3"]
    )

    # ADDRESS

    address_exact = int(
        bool(sa) and bool(ca) and sa == ca
    )

    address_ratio = similarity(sa, ca)

    address_token = token_similarity(sa, ca)

    address_number_match = int(
        bool(source["address_number"])
        and source["address_number"]
        == candidate["address_number"]
    )

    address_first_match = int(
        bool(source["address_first"])
        and source["address_first"]
        == candidate["address_first"]
    )

    address_last_match = int(
        bool(source["address_last"])
        and source["address_last"]
        == candidate["address_last"]
    )

    # COUNTRY

    country_match = int(
        bool(sc)
        and bool(cc)
        and sc == cc
    )

    # COMBINED

    combined_score = (
        name_ratio * 0.55
        + address_ratio * 0.35
        + country_match * 10
    )

    strong_name_address = (
        name_ratio * address_ratio / 100.0
    )

    exact_combo = (
        name_exact
        + address_exact
        + country_match
    )

    return [

        name_exact,
        name_ratio,
        name_token,
        name_first_match,
        name_last_match,
        name_prefix4_match,
        name_prefix6_match,
        name_suffix_match,

        address_exact,
        address_ratio,
        address_token,
        address_number_match,
        address_first_match,
        address_last_match,

        country_match,

        combined_score,
        strong_name_address,
        exact_combo
    ]


FEATURE_NAMES = [

    "name_exact",
    "name_ratio",
    "name_token",
    "name_first_match",
    "name_last_match",
    "name_prefix4_match",
    "name_prefix6_match",
    "name_suffix_match",

    "address_exact",
    "address_ratio",
    "address_token",
    "address_number_match",
    "address_first_match",
    "address_last_match",

    "country_match",

    "combined_score",
    "strong_name_address",
    "exact_combo"
]


# ============================================================
# LOAD CANDIDATES
# ============================================================

print("\nLoading candidate file...")

candidates = pd.read_csv(
    CANDIDATE_FILE,
    sep="\t",
    dtype=str
)

candidates = candidates.head(
    TOTAL_ROWS
).copy()

print(
    "Candidate rows:",
    len(candidates)
)


# ============================================================
# SHUFFLE + SPLIT
# ============================================================

print("\nCreating train/test split...")

rng = np.random.default_rng(SEED)

indices = np.arange(
    len(candidates)
)

rng.shuffle(indices)

train_indices = indices[:TRAIN_ROWS]

test_indices = indices[
    TRAIN_ROWS:
    TRAIN_ROWS + TEST_ROWS
]

train_candidates = (
    candidates.iloc[train_indices]
    .copy()
)

test_candidates = (
    candidates.iloc[test_indices]
    .copy()
)

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
    SEED
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

gt_dict = {}

for _, row in gt.iterrows():

    source_id = row[
        "source1_entity_id"
    ]

    raw = row[
        "matched_entity_ids"
    ]

    if pd.isna(raw):

        gt_dict[source_id] = set()

    else:

        gt_dict[source_id] = {
            x.strip()
            for x in str(raw).split(",")
            if x.strip()
        }

print(
    "Ground-truth IDs:",
    len(gt_dict)
)


# ============================================================
# LOAD SOURCE DATA
# ============================================================

print("\nLoading Source 1...")

s1 = pd.read_csv(
    SOURCE1_FILE,
    sep="\t",
    dtype=str
)

print(
    "Source 1:",
    len(s1)
)


print("\nLoading Source 2...")

s2 = pd.read_csv(
    SOURCE2_FILE,
    sep="\t",
    dtype=str
)

print(
    "Source 2:",
    len(s2)
)


print("\nLoading Source 3...")

s3 = pd.read_csv(
    SOURCE3_FILE,
    sep="\t",
    dtype=str
)

print(
    "Source 3:",
    len(s3)
)


# ============================================================
# SOURCE 1 LOOKUP
# ============================================================

print("\nBuilding Source-1 lookup...")

needed_source_ids = set(
    candidates[
        "source1_entity_id"
    ].astype(str)
)

s1 = s1[
    s1["entity_id"]
    .astype(str)
    .isin(needed_source_ids)
].copy()

source1_lookup = {}

for _, row in s1.iterrows():

    entity_id = str(
        row["entity_id"]
    )

    name = normalize_text(
        row["business_name"]
    )

    address = normalize_text(
        row["business_address"]
    )

    country = normalize_country(
        row["country"]
    )

    source1_lookup[entity_id] = {

        "name": name,

        "address": address,

        "country": country,

        "name_first":
            first_word(name),

        "name_last":
            last_word(name),

        "name_prefix4":
            prefix(name, 4),

        "name_prefix6":
            prefix(name, 6),

        "name_suffix3":
            suffix(name, 3),

        "address_number":
            extract_number(address),

        "address_first":
            first_word(address),

        "address_last":
            last_word(address)
    }


# ============================================================
# CANDIDATE LOOKUP
# ============================================================

print("\nBuilding candidate lookup...")

candidate_ids_needed = set()

for value in candidates[
    "candidate_entity_ids"
].astype(str):

    candidate_ids_needed.update(
        x.strip()
        for x in value.split(",")
        if x.strip()
    )

print(
    "Candidate IDs needed:",
    len(candidate_ids_needed)
)


candidate_lookup = {}


for df in [s2, s3]:

    for _, row in df.iterrows():

        entity_id = str(
            row["entity_id"]
        )

        if entity_id not in candidate_ids_needed:
            continue

        name = normalize_text(
            row["business_name"]
        )

        address = normalize_text(
            row["business_address"]
        )

        country = normalize_country(
            row["country"]
        )

        candidate_lookup[entity_id] = {

            "name": name,

            "address": address,

            "country": country,

            "name_first":
                first_word(name),

            "name_last":
                last_word(name),

            "name_prefix4":
                prefix(name, 4),

            "name_prefix6":
                prefix(name, 6),

            "name_suffix3":
                suffix(name, 3),

            "address_number":
                extract_number(address),

            "address_first":
                first_word(address),

            "address_last":
                last_word(address)
        }


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

X_train = []
y_train = []
groups = []

for counter, (_, row) in enumerate(
    train_candidates.iterrows(),
    1
):

    source_id = str(
        row["source1_entity_id"]
    )

    source = source1_lookup.get(
        source_id
    )

    if source is None:
        continue

    raw = str(
        row["candidate_entity_ids"]
    )

    candidate_ids = [
        x.strip()
        for x in raw.split(",")
        if x.strip()
    ]

    truth = gt_dict.get(
        source_id,
        set()
    )

    group_count = 0

    for candidate_id in candidate_ids:

        candidate = candidate_lookup.get(
            candidate_id
        )

        if candidate is None:
            continue

        X_train.append(
            make_features(
                source,
                candidate
            )
        )

        y_train.append(
            1
            if candidate_id in truth
            else 0
        )

        group_count += 1

    if group_count > 0:

        groups.append(
            group_count
        )

    if counter % 100 == 0:

        print(
            f"Training rows processed: "
            f"{counter}/{len(train_candidates)}"
        )


X_train = np.asarray(
    X_train,
    dtype=np.float32
)

y_train = np.asarray(
    y_train,
    dtype=np.int8
)

groups = np.asarray(
    groups,
    dtype=np.int32
)

print("\nTraining examples:", len(X_train))
print("Positive matches:", int(y_train.sum()))
print("Training groups:", len(groups))


# ============================================================
# TRAIN
# ============================================================

print("\n" + "=" * 70)
print("TRAINING LIGHTGBM")
print("=" * 70)

model = lgb.LGBMRanker(

    objective="lambdarank",

    metric="ndcg",

    n_estimators=400,

    learning_rate=0.04,

    num_leaves=31,

    max_depth=-1,

    min_child_samples=20,

    subsample=0.9,

    colsample_bytree=0.9,

    reg_alpha=0.1,

    reg_lambda=0.5,

    random_state=SEED,

    n_jobs=-1,

    verbosity=-1
)


model.fit(

    X_train,

    y_train,

    group=groups,

    feature_name=FEATURE_NAMES
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

ranked_rows = []

for counter, (_, row) in enumerate(
    test_candidates.iterrows(),
    1
):

    source_id = str(
        row["source1_entity_id"]
    )

    source = source1_lookup.get(
        source_id
    )

    if source is None:
        continue

    raw = str(
        row["candidate_entity_ids"]
    )

    candidate_ids = [
        x.strip()
        for x in raw.split(",")
        if x.strip()
    ]

    valid_ids = []
    features = []

    for candidate_id in candidate_ids:

        candidate = candidate_lookup.get(
            candidate_id
        )

        if candidate is None:
            continue

        valid_ids.append(
            candidate_id
        )

        features.append(
            make_features(
                source,
                candidate
            )
        )

    if not valid_ids:
        continue

    features = np.asarray(
        features,
        dtype=np.float32
    )

    scores = model.predict(
        features
    )

    order = np.argsort(
        -scores
    )

    ranked_ids = [
        valid_ids[i]
        for i in order[:TOP_K]
    ]

    ranked_rows.append({

        "source1_entity_id":
            source_id,

        "candidate_entity_ids":
            ",".join(ranked_ids)
    })

    if counter % 50 == 0:

        print(
            f"Ranked test rows: "
            f"{counter}/{len(test_candidates)}"
        )


# ============================================================
# SAVE
# ============================================================

ranked = pd.DataFrame(
    ranked_rows
)

ranked.to_csv(
    OUTPUT_FILE,
    sep="\t",
    index=False
)


# ============================================================
# DIRECT EVALUATION
# ============================================================

print("\n" + "=" * 70)
print("UNSEEN TEST RESULTS")
print("=" * 70)

hits = {
    1: 0,
    5: 0,
    10: 0,
    20: 0,
    50: 0,
    100: 0
}

evaluated = 0

for _, row in ranked.iterrows():

    source_id = str(
        row["source1_entity_id"]
    )

    truth = gt_dict.get(
        source_id,
        set()
    )

    if not truth:
        continue

    candidate_ids = [
        x.strip()
        for x in str(
            row["candidate_entity_ids"]
        ).split(",")
        if x.strip()
    ]

    evaluated += 1

    for k in hits:

        top_k = candidate_ids[:k]

        if any(
            x in truth
            for x in top_k
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

    recall = (
        hits[k] / evaluated * 100
        if evaluated
        else 0
    )

    print(
        f"Unseen Recall@{k:<3}: "
        f"{recall:.2f}%"
    )


# ============================================================
# FINAL
# ============================================================

print("\n" + "=" * 70)
print("SPLIT EVALUATION COMPLETE")
print("=" * 70)

print(
    "Training rows:",
    len(train_candidates)
)

print(
    "Unseen test rows:",
    len(test_candidates)
)

print(
    "Ranked rows:",
    len(ranked)
)

print(
    "Saved:",
    OUTPUT_FILE
)

print(
    "Total runtime:",
    f"{time.time() - start_total:.1f}s"
)

print("=" * 70)