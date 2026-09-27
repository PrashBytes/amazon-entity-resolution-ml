import re
import os
import time
import numpy as np
import pandas as pd

from collections import defaultdict
from rapidfuzz.fuzz import ratio, token_set_ratio, WRatio
from lightgbm import Booster


# ============================================================
# CONFIG
# ============================================================

TEST_BASE = "dataset/test"

SOURCE1 = f"{TEST_BASE}/test_source1.tsv"
SOURCE2 = f"{TEST_BASE}/test_source2.tsv"
SOURCE3 = f"{TEST_BASE}/test_source3.tsv"

MODEL_FILE = "match_model_v2.txt"

OUTPUT_MATCHES = "output/matching_results.tsv"
OUTPUT_CANDIDATES = "output/candidate_pairs.tsv"

THRESHOLD = 0.75
MAX_CANDIDATES = 100


# ============================================================
# FEATURES
# MUST MATCH TRAINING EXACTLY
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
# SAME AS TRAINING
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
    return set(text.split())


# ============================================================
# LOAD TEST DATA
# ============================================================

print("=" * 60)
print("LOADING TEST DATA")
print("=" * 60)

s1 = pd.read_csv(
    SOURCE1,
    sep="\t"
)

s2 = pd.read_csv(
    SOURCE2,
    sep="\t"
)

s3 = pd.read_csv(
    SOURCE3,
    sep="\t"
)

print("Source 1:", len(s1))
print("Source 2:", len(s2))
print("Source 3:", len(s3))


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
# HIGH-RECALL BLOCKING
# ============================================================

print("Building blocking indexes...")

name_index = defaultdict(list)
address_index = defaultdict(list)
first_word_index = defaultdict(list)


def add_to_indexes(record):

    entity_id = record.entity_id

    name = record.name_norm
    address = record.address_norm

    # NAME BLOCKS
    if name:

        name_index[
            name[:6]
        ].append(entity_id)

        name_index[
            name[:4]
        ].append(entity_id)

        first_word = name.split()[0]

        first_word_index[
            first_word
        ].append(entity_id)

    # ADDRESS BLOCK
    if address:

        address_index[
            address[:8]
        ].append(entity_id)


for record in s2.itertuples(index=False):
    add_to_indexes(record)

for record in s3.itertuples(index=False):
    add_to_indexes(record)


print("Indexes ready!")


# ============================================================
# FEATURE CALCULATION
# EXACTLY SAME AS TRAINING
# ============================================================

def make_features(a, b):

    name1 = a.name_norm
    name2 = b.name_norm

    addr1 = a.address_norm
    addr2 = b.address_norm

    country1 = a.country_norm
    country2 = b.country_norm


    # -------------------------
    # NAME
    # -------------------------

    name_ratio = (
        ratio(name1, name2)
        / 100.0
    )

    name_token = (
        token_set_ratio(name1, name2)
        / 100.0
    )

    name_wratio = (
        WRatio(name1, name2)
        / 100.0
    )

    exact_name = int(
        name1 != "" and
        name1 == name2
    )


    # -------------------------
    # ADDRESS
    # -------------------------

    address_ratio = (
        ratio(addr1, addr2)
        / 100.0
    )

    address_token = (
        token_set_ratio(addr1, addr2)
        / 100.0
    )

    address_wratio = (
        WRatio(addr1, addr2)
        / 100.0
    )

    exact_address = int(
        addr1 != "" and
        addr1 == addr2
    )


    # -------------------------
    # JACCARD
    # -------------------------

    n1 = token_set(name1)
    n2 = token_set(name2)

    name_jaccard = (
        len(n1 & n2)
        /
        max(len(n1 | n2), 1)
    )


    a1 = token_set(addr1)
    a2 = token_set(addr2)

    address_jaccard = (
        len(a1 & a2)
        /
        max(len(a1 | a2), 1)
    )


    # -------------------------
    # LENGTH
    # -------------------------

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


    # -------------------------
    # COUNTRY
    # -------------------------

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
# LOAD LIGHTGBM MODEL
# ============================================================

print("\nLoading LightGBM model...")

model = Booster(
    model_file=MODEL_FILE
)

print("Model loaded!")


# ============================================================
# CANDIDATE GENERATION
# ============================================================

def generate_candidates(source):

    candidates = set()

    name = source.name_norm
    address = source.address_norm
    country = source.country_norm


    # --------------------------------------------------------
    # NAME BLOCKS
    # --------------------------------------------------------

    if name:

        candidates.update(
            name_index.get(
                name[:6],
                []
            )
        )

        candidates.update(
            name_index.get(
                name[:4],
                []
            )
        )


        words = name.split()

        if words:

            candidates.update(
                first_word_index.get(
                    words[0],
                    []
                )
            )


    # --------------------------------------------------------
    # ADDRESS BLOCK
    # --------------------------------------------------------

    if address:

        candidates.update(
            address_index.get(
                address[:8],
                []
            )
        )


    # --------------------------------------------------------
    # COUNTRY FILTER
    # --------------------------------------------------------

    candidates = [
        entity_id
        for entity_id in candidates
        if (
            entity_id.startswith("S2-")
            or entity_id.startswith("S3-")
        )
    ]


    # --------------------------------------------------------
    # COUNTRY-FIRST FILTER
    # --------------------------------------------------------

    if country:

        filtered = []

        for entity_id in candidates:

            candidate = get_record(
                entity_id
            )

            if candidate is None:
                continue

            if (
                candidate.country_norm
                == country
            ):
                filtered.append(
                    entity_id
                )

        candidates = filtered


    # --------------------------------------------------------
    # REMOVE DUPLICATES
    # --------------------------------------------------------

    candidates = list(
        dict.fromkeys(candidates)
    )


    # --------------------------------------------------------
    # CHEAP NAME RANKING
    # Keep only top 100 before expensive features
    # --------------------------------------------------------

    scored = []

    for entity_id in candidates:

        candidate = get_record(
            entity_id
        )

        if candidate is None:
            continue

        score = ratio(
            source.name_norm,
            candidate.name_norm
        )

        scored.append(
            (
                entity_id,
                score
            )
        )


    scored.sort(
        key=lambda x: x[1],
        reverse=True
    )


    return [
        x[0]
        for x in scored[:MAX_CANDIDATES]
    ]


# ============================================================
# PROCESS TEST DATA
# ============================================================

print("\n" + "=" * 60)
print("GENERATING ML SUBMISSION")
print("=" * 60)

results = []
candidate_results = []

start_time = time.time()

total = len(s1)


for i, source in enumerate(
    s1.itertuples(index=False)
):

    source_id = source.entity_id


    # --------------------------------------------------------
    # CANDIDATES
    # --------------------------------------------------------

    candidates = generate_candidates(
        source
    )


    candidate_results.append({

        "source1_entity_id":
            source_id,

        "candidate_entity_ids":
            ",".join(candidates)
    })


    # --------------------------------------------------------
    # NO CANDIDATES
    # --------------------------------------------------------

    if not candidates:

        results.append({

            "source1_entity_id":
                source_id,

            "matched_entity_ids":
                ""
        })

        continue


    # --------------------------------------------------------
    # FEATURES
    # --------------------------------------------------------

    feature_rows = []
    valid_ids = []


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

        feature_rows.append(
            features
        )

        valid_ids.append(
            candidate_id
        )


    # --------------------------------------------------------
    # MODEL PREDICTION
    # --------------------------------------------------------

    if feature_rows:

        X = pd.DataFrame(
            feature_rows,
            columns=FEATURE_NAMES
        )

        probabilities = model.predict(
            X
        )


        matches = []

        for entity_id, probability in zip(
            valid_ids,
            probabilities
        ):

            if probability >= THRESHOLD:

                matches.append(
                    (
                        entity_id,
                        probability
                    )
                )


        # Highest probability first
        matches.sort(
            key=lambda x: x[1],
            reverse=True
        )


        predicted_ids = [
            entity_id
            for entity_id, probability
            in matches
        ]

    else:

        predicted_ids = []


    # --------------------------------------------------------
    # SAVE RESULT
    # --------------------------------------------------------

    results.append({

        "source1_entity_id":
            source_id,

        "matched_entity_ids":
            ",".join(predicted_ids)
    })


    # --------------------------------------------------------
    # PROGRESS
    # --------------------------------------------------------

    if (i + 1) % 10000 == 0:

        elapsed = (
            time.time()
            - start_time
        )

        rate = (
            (i + 1)
            / elapsed
        )

        remaining = (
            total - i - 1
        ) / rate


        print(
            f"Processed "
            f"{i + 1:,}/"
            f"{total:,}"
            f" | {rate:.1f} rows/sec"
            f" | ETA "
            f"{remaining / 60:.1f} min"
        )


# ============================================================
# SAVE OUTPUT
# ============================================================

print("\nSaving outputs...")

os.makedirs(
    "output",
    exist_ok=True
)


matching_df = pd.DataFrame(
    results,
    columns=[
        "source1_entity_id",
        "matched_entity_ids"
    ]
)

matching_df.to_csv(
    OUTPUT_MATCHES,
    sep="\t",
    index=False
)


candidate_df = pd.DataFrame(
    candidate_results,
    columns=[
        "source1_entity_id",
        "candidate_entity_ids"
    ]
)

candidate_df.to_csv(
    OUTPUT_CANDIDATES,
    sep="\t",
    index=False
)


# ============================================================
# VALIDATION
# ============================================================

print("\n" + "=" * 60)
print("SUBMISSION VALIDATION")
print("=" * 60)

valid_s2 = set(
    s2.entity_id
)

valid_s3 = set(
    s3.entity_id
)

valid_ids = (
    valid_s2 |
    valid_s3
)


invalid_ids = 0

for values in matching_df[
    "matched_entity_ids"
]:

    if not values:
        continue

    for entity_id in values.split(","):

        if entity_id not in valid_ids:

            invalid_ids += 1


print(
    "Expected rows:",
    len(s1)
)

print(
    "Output rows:",
    len(matching_df)
)

print(
    "Duplicate Source 1 IDs:",
    matching_df[
        "source1_entity_id"
    ].duplicated().sum()
)

print(
    "Invalid predicted IDs:",
    invalid_ids
)

print(
    "Source 1 records with matches:",
    (
        matching_df[
            "matched_entity_ids"
        ]
        .astype(str)
        .str.len()
        .gt(0)
        .sum()
    )
)

print("\n" + "=" * 60)
print("ML SUBMISSION READY")
print("=" * 60)

print(
    OUTPUT_MATCHES
)

print(
    OUTPUT_CANDIDATES
)

print("\nDONE!")