import pandas as pd
import re
import unicodedata
from collections import defaultdict
from rapidfuzz.fuzz import ratio
import os
import time


# ============================================================
# SETTINGS
# ============================================================

FUZZY_THRESHOLD = 88
MAX_FUZZY_CANDIDATES = 20


# ============================================================
# NORMALIZATION
# ============================================================

def normalize_text(text):

    if pd.isna(text):
        return ""

    text = unicodedata.normalize("NFKD", str(text))
    text = text.encode("ascii", "ignore").decode("ascii")
    text = text.lower()

    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


# ============================================================
# LOAD DATA
# ============================================================

print("======================================")
print("LOADING TEST DATA")
print("======================================")

s1 = pd.read_csv(
    "dataset/test/test_source1.tsv",
    sep="\t"
)

s2 = pd.read_csv(
    "dataset/test/test_source2.tsv",
    sep="\t"
)

s3 = pd.read_csv(
    "dataset/test/test_source3.tsv",
    sep="\t"
)

print("Source 1:", len(s1))
print("Source 2:", len(s2))
print("Source 3:", len(s3))


# ============================================================
# NORMALIZATION
# ============================================================

print("\nNormalizing...")

for df in [s1, s2, s3]:

    df["name_norm"] = (
        df["business_name"]
        .fillna("")
        .map(normalize_text)
    )

    df["address_norm"] = (
        df["business_address"]
        .fillna("")
        .map(normalize_text)
    )

    df["country_norm"] = (
        df["country"]
        .fillna("")
        .astype(str)
        .str.lower()
        .str.strip()
    )


# ============================================================
# BUILD BLOCKING KEYS
# ============================================================

print("Creating blocking keys...")


def add_keys(df):

    name = df["name_norm"]

    df["key1"] = (
        df["country_norm"]
        + "|"
        + name.str[:6]
    )

    df["key2"] = (
        df["country_norm"]
        + "|"
        + name.str[:4]
    )

    # First word of business name
    df["first_word"] = (
        name.str.split()
        .str[0]
        .fillna("")
    )

    df["key3"] = (
        df["country_norm"]
        + "|"
        + df["first_word"]
    )

    return df


s1 = add_keys(s1)
s2 = add_keys(s2)
s3 = add_keys(s3)


# ============================================================
# BUILD EXACT NAME INDEX
# ============================================================

print("Building exact-name indexes...")

name_index_s2 = defaultdict(list)
name_index_s3 = defaultdict(list)

for entity_id, name in zip(
    s2["entity_id"],
    s2["name_norm"]
):

    if name:
        name_index_s2[name].append(entity_id)


for entity_id, name in zip(
    s3["entity_id"],
    s3["name_norm"]
):

    if name:
        name_index_s3[name].append(entity_id)


# ============================================================
# BUILD BLOCK INDEXES
# ============================================================

print("Building blocking indexes...")


def build_index(df):

    index = defaultdict(list)

    for entity_id, k1, k2, k3 in zip(
        df["entity_id"],
        df["key1"],
        df["key2"],
        df["key3"]
    ):

        if k1:
            index[k1].append(entity_id)

        if k2:
            index[k2].append(entity_id)

        if k3:
            index[k3].append(entity_id)

    return index


index_s2 = build_index(s2)
index_s3 = build_index(s3)

print("Indexes ready!")


# ============================================================
# LOOKUPS
# ============================================================

print("Creating lookups...")

s2_lookup = s2.set_index("entity_id")[
    ["name_norm", "address_norm", "country_norm"]
].to_dict("index")

s3_lookup = s3.set_index("entity_id")[
    ["name_norm", "address_norm", "country_norm"]
].to_dict("index")


# ============================================================
# GENERATE ONE PREDICTION
# ============================================================

def predict(row):

    source_name = row["name_norm"]
    source_address = row["address_norm"]
    source_country = row["country_norm"]

    matches = []

    # --------------------------------------------------------
    # 1. EXACT NAME MATCHES
    # --------------------------------------------------------

    exact_s2 = name_index_s2.get(
        source_name,
        []
    )

    exact_s3 = name_index_s3.get(
        source_name,
        []
    )

    for entity_id in exact_s2:

        candidate = s2_lookup[entity_id]

        if candidate["country_norm"] == source_country:

            matches.append(
                (entity_id, 100.0)
            )

    for entity_id in exact_s3:

        candidate = s3_lookup[entity_id]

        if candidate["country_norm"] == source_country:

            matches.append(
                (entity_id, 100.0)
            )

    # Exact names are extremely strong.
    if matches:
        return list(
            dict.fromkeys(
                x[0] for x in matches
            )
        )

    # --------------------------------------------------------
    # 2. BLOCKING
    # --------------------------------------------------------

    candidate_ids_s2 = set()
    candidate_ids_s3 = set()

    for key in [
        row["key1"],
        row["key2"],
        row["key3"]
    ]:

        if not key:
            continue

        candidate_ids_s2.update(
            index_s2.get(key, [])
        )

        candidate_ids_s3.update(
            index_s3.get(key, [])
        )

    # --------------------------------------------------------
    # 3. CHEAP PRE-FILTER
    # --------------------------------------------------------

    candidates = []

    for entity_id in candidate_ids_s2:

        candidate = s2_lookup[entity_id]

        if candidate["country_norm"] != source_country:
            continue

        name_score = ratio(
            source_name,
            candidate["name_norm"]
        )

        # Only expensive address comparison
        # for reasonably similar names.
        if name_score >= FUZZY_THRESHOLD:

            candidates.append(
                (
                    entity_id,
                    name_score,
                    candidate["address_norm"]
                )
            )


    for entity_id in candidate_ids_s3:

        candidate = s3_lookup[entity_id]

        if candidate["country_norm"] != source_country:
            continue

        name_score = ratio(
            source_name,
            candidate["name_norm"]
        )

        if name_score >= FUZZY_THRESHOLD:

            candidates.append(
                (
                    entity_id,
                    name_score,
                    candidate["address_norm"]
                )
            )


    # --------------------------------------------------------
    # 4. ADDRESS CHECK ONLY FOR TOP NAME CANDIDATES
    # --------------------------------------------------------

    candidates.sort(
        key=lambda x: x[1],
        reverse=True
    )

    candidates = candidates[
        :MAX_FUZZY_CANDIDATES
    ]

    final_matches = []

    for entity_id, name_score, candidate_address in candidates:

        address_score = ratio(
            source_address,
            candidate_address
        )

        final_score = (
            0.65 * name_score
            +
            0.35 * address_score
        )

        # Conservative final threshold
        if final_score >= FUZZY_THRESHOLD:

            final_matches.append(
                (
                    entity_id,
                    final_score
                )
            )

    final_matches.sort(
        key=lambda x: x[1],
        reverse=True
    )

    return [
        x[0]
        for x in final_matches
    ]


# ============================================================
# PROCESS TEST DATA
# ============================================================

print("\n======================================")
print("GENERATING SUBMISSION")
print("======================================")

start_time = time.time()

results = []

total = len(s1)

for i, row in enumerate(
    s1.to_dict("records")
):

    predicted = predict(row)

    results.append({
        "source1_entity_id":
            row["entity_id"],

        "matched_entity_ids":
            ",".join(predicted)
    })

    if (i + 1) % 10000 == 0:

        elapsed = time.time() - start_time

        rate = (
            (i + 1) / elapsed
            if elapsed > 0
            else 0
        )

        remaining = (
            (total - i - 1) / rate
            if rate > 0
            else 0
        )

        print(
            f"Processed {i + 1:,}/{total:,} "
            f"| {rate:.1f} rows/sec "
            f"| ETA {remaining/60:.1f} min"
        )


# ============================================================
# SAVE
# ============================================================

print("\nSaving output...")

os.makedirs(
    "output",
    exist_ok=True
)

submission = pd.DataFrame(
    results,
    columns=[
        "source1_entity_id",
        "matched_entity_ids"
    ]
)

submission.to_csv(
    "output/matching_results.tsv",
    sep="\t",
    index=False
)


# ============================================================
# VALIDATION
# ============================================================

print("\n======================================")
print("VALIDATION")
print("======================================")

print(
    "Expected rows:",
    len(s1)
)

print(
    "Output rows:",
    len(submission)
)

print(
    "Duplicate Source 1 IDs:",
    submission["source1_entity_id"]
    .duplicated()
    .sum()
)

valid_s2 = set(
    s2["entity_id"]
)

valid_s3 = set(
    s3["entity_id"]
)

valid_ids = valid_s2 | valid_s3

invalid = 0

for values in submission[
    "matched_entity_ids"
]:

    if not values:
        continue

    for entity_id in values.split(","):

        if entity_id not in valid_ids:
            invalid += 1


print(
    "Invalid predicted IDs:",
    invalid
)

non_empty = (
    submission["matched_entity_ids"]
    .astype(str)
    .str.len()
    .gt(0)
    .sum()
)

print(
    "Source 1 records with matches:",
    non_empty
)

print("\n======================================")
print("SUBMISSION READY")
print("======================================")

print(
    "output/matching_results.tsv"
)

print("\nDONE!")