import re
import os
import time
import numpy as np
import pandas as pd

from collections import defaultdict
from rapidfuzz import fuzz, process
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

# Process in chunks so RAM stays under control
BATCH_SIZE = 5000


# ============================================================
# NORMALIZATION
# ============================================================

def normalize_series(series):

    return (
        series.fillna("")
        .astype(str)
        .str.lower()
        .str.replace(r"[^a-z0-9]+", " ", regex=True)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )


def token_set(text):

    if not text:
        return set()

    return set(text.split())


# ============================================================
# LOAD DATA
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

for df in (s1, s2, s3):

    df["name_norm"] = normalize_series(
        df["business_name"]
    )

    df["address_norm"] = normalize_series(
        df["business_address"]
    )

    df["country_norm"] = normalize_series(
        df["country"]
    )


# ============================================================
# COMBINE SOURCE 2 + SOURCE 3
# ============================================================

print("Combining candidate sources...")

candidates_df = pd.concat(
    [
        s2[
            [
                "entity_id",
                "name_norm",
                "address_norm",
                "country_norm"
            ]
        ],
        s3[
            [
                "entity_id",
                "name_norm",
                "address_norm",
                "country_norm"
            ]
        ]
    ],
    ignore_index=True
)

candidate_ids = (
    candidates_df["entity_id"]
    .astype(str)
    .to_numpy()
)

candidate_names = (
    candidates_df["name_norm"]
    .fillna("")
    .to_numpy()
)

candidate_addresses = (
    candidates_df["address_norm"]
    .fillna("")
    .to_numpy()
)

candidate_countries = (
    candidates_df["country_norm"]
    .fillna("")
    .to_numpy()
)


# ============================================================
# LOOKUP TABLES
# ============================================================

print("Creating lookup tables...")

id_to_index = {
    entity_id: i
    for i, entity_id
    in enumerate(candidate_ids)
}


# ============================================================
# BLOCKING INDEXES
# ============================================================

print("Building blocking indexes...")

name6_index = defaultdict(list)
name4_index = defaultdict(list)
firstword_index = defaultdict(list)
address8_index = defaultdict(list)
country_index = defaultdict(list)


for i in range(len(candidate_ids)):

    name = candidate_names[i]
    address = candidate_addresses[i]
    country = candidate_countries[i]

    if name:

        name6_index[
            name[:6]
        ].append(i)

        name4_index[
            name[:4]
        ].append(i)

        first_word = name.split(" ", 1)[0]

        firstword_index[
            first_word
        ].append(i)

    if address:

        address8_index[
            address[:8]
        ].append(i)

    if country:

        country_index[
            country
        ].append(i)


print("Indexes ready!")


# ============================================================
# MODEL
# ============================================================

print("\nLoading LightGBM model...")

model = Booster(
    model_file=MODEL_FILE
)

print("Model loaded!")


# ============================================================
# FAST CANDIDATE GENERATION
# ============================================================

def get_candidates(
    name,
    address,
    country
):

    pool = set()

    # -------------------------
    # NAME BLOCKS
    # -------------------------

    if name:

        pool.update(
            name6_index.get(
                name[:6],
                ()
            )
        )

        pool.update(
            name4_index.get(
                name[:4],
                ()
            )
        )

        first_word = name.split(
            " ",
            1
        )[0]

        pool.update(
            firstword_index.get(
                first_word,
                ()
            )
        )


    # -------------------------
    # ADDRESS BLOCK
    # -------------------------

    if address:

        pool.update(
            address8_index.get(
                address[:8],
                ()
            )
        )


    if not pool:

        return []


    # --------------------------------------------------------
    # COUNTRY PREFERENCE
    # --------------------------------------------------------

    if country:

        same_country = [
            i for i in pool
            if candidate_countries[i] == country
        ]

        if same_country:

            pool = same_country


    # --------------------------------------------------------
    # RAPIDFUZZ C-OPTIMIZED RANKING
    # --------------------------------------------------------

    pool_list = list(pool)

    choices = {
        i: candidate_names[i]
        for i in pool_list
    }

    matches = process.extract(
        name,
        choices,
        scorer=fuzz.ratio,
        limit=MAX_CANDIDATES
    )

    return [
        match[2]
        for match in matches
    ]


# ============================================================
# FAST FEATURE CALCULATION
# ============================================================

def build_features(
    source_name,
    source_address,
    source_country,
    candidate_indices
):

    n = len(candidate_indices)

    if n == 0:

        return None


    names = [
        candidate_names[i]
        for i in candidate_indices
    ]

    addresses = [
        candidate_addresses[i]
        for i in candidate_indices
    ]

    countries = [
        candidate_countries[i]
        for i in candidate_indices
    ]


    # --------------------------------------------------------
    # NAME FEATURES
    # --------------------------------------------------------

    name_ratio = np.array([
        x[1] / 100.0
        for x in process.extract(
            source_name,
            names,
            scorer=fuzz.ratio,
            limit=n
        )
    ])


    name_token = np.array([
        x[1] / 100.0
        for x in process.extract(
            source_name,
            names,
            scorer=fuzz.token_set_ratio,
            limit=n
        )
    ])


    name_wratio = np.array([
        x[1] / 100.0
        for x in process.extract(
            source_name,
            names,
            scorer=fuzz.WRatio,
            limit=n
        )
    ])


    exact_name = np.array([
        int(
            source_name != ""
            and source_name == x
        )
        for x in names
    ])


    # --------------------------------------------------------
    # ADDRESS FEATURES
    # --------------------------------------------------------

    address_ratio = np.array([
        x[1] / 100.0
        for x in process.extract(
            source_address,
            addresses,
            scorer=fuzz.ratio,
            limit=n
        )
    ])


    address_token = np.array([
        x[1] / 100.0
        for x in process.extract(
            source_address,
            addresses,
            scorer=fuzz.token_set_ratio,
            limit=n
        )
    ])


    address_wratio = np.array([
        x[1] / 100.0
        for x in process.extract(
            source_address,
            addresses,
            scorer=fuzz.WRatio,
            limit=n
        )
    ])


    exact_address = np.array([
        int(
            source_address != ""
            and source_address == x
        )
        for x in addresses
    ])


    # --------------------------------------------------------
    # JACCARD
    # --------------------------------------------------------

    source_name_tokens = token_set(
        source_name
    )

    source_address_tokens = token_set(
        source_address
    )


    name_jaccard = []

    address_jaccard = []


    for name, address in zip(
        names,
        addresses
    ):

        nt = token_set(name)

        union = (
            source_name_tokens | nt
        )

        if union:

            name_jaccard.append(
                len(
                    source_name_tokens & nt
                ) / len(union)
            )

        else:

            name_jaccard.append(0.0)


        at = token_set(address)

        union = (
            source_address_tokens | at
        )

        if union:

            address_jaccard.append(
                len(
                    source_address_tokens & at
                ) / len(union)
            )

        else:

            address_jaccard.append(0.0)


    name_jaccard = np.array(
        name_jaccard
    )

    address_jaccard = np.array(
        address_jaccard
    )


    # --------------------------------------------------------
    # LENGTH RATIOS
    # --------------------------------------------------------

    source_name_len = len(
        source_name
    )

    source_address_len = len(
        source_address
    )


    name_length_ratio = np.array([

        min(
            source_name_len,
            len(x)
        )
        /
        max(
            source_name_len,
            len(x),
            1
        )

        for x in names
    ])


    address_length_ratio = np.array([

        min(
            source_address_len,
            len(x)
        )
        /
        max(
            source_address_len,
            len(x),
            1
        )

        for x in addresses
    ])


    # --------------------------------------------------------
    # COUNTRY
    # --------------------------------------------------------

    country_match = np.array([

        int(
            source_country != ""
            and source_country == x
        )

        for x in countries

    ])


    # --------------------------------------------------------
    # FINAL FEATURE MATRIX
    # --------------------------------------------------------

    return np.column_stack([

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

    ])


# ============================================================
# PROCESS
# ============================================================

print("\n" + "=" * 60)
print("FAST ML INFERENCE")
print("=" * 60)

os.makedirs(
    "output",
    exist_ok=True
)


# Remove previous outputs
for path in [
    OUTPUT_MATCHES,
    OUTPUT_CANDIDATES
]:

    if os.path.exists(path):

        os.remove(path)


total = len(s1)

start_time = time.time()

first_write = True


# ------------------------------------------------------------
# PROCESS IN BATCHES
# ------------------------------------------------------------

for batch_start in range(
    0,
    total,
    BATCH_SIZE
):

    batch_end = min(
        batch_start + BATCH_SIZE,
        total
    )

    batch = s1.iloc[
        batch_start:batch_end
    ]


    batch_results = []
    batch_candidates = []


    for source in batch.itertuples(
        index=False
    ):

        source_id = source.entity_id

        source_name = source.name_norm
        source_address = source.address_norm
        source_country = source.country_norm


        # ----------------------------------------------------
        # CANDIDATES
        # ----------------------------------------------------

        candidate_indices = get_candidates(
            source_name,
            source_address,
            source_country
        )


        candidate_ids_for_row = [
            candidate_ids[i]
            for i in candidate_indices
        ]


        batch_candidates.append({

            "source1_entity_id":
                source_id,

            "candidate_entity_ids":
                ",".join(
                    candidate_ids_for_row
                )
        })


        # ----------------------------------------------------
        # NO CANDIDATES
        # ----------------------------------------------------

        if not candidate_indices:

            batch_results.append({

                "source1_entity_id":
                    source_id,

                "matched_entity_ids":
                    ""
            })

            continue


        # ----------------------------------------------------
        # FEATURES
        # ----------------------------------------------------

        X = build_features(

            source_name,
            source_address,
            source_country,

            candidate_indices

        )


        # ----------------------------------------------------
        # LIGHTGBM
        # ----------------------------------------------------

        probabilities = model.predict(
            X
        )


        # ----------------------------------------------------
        # THRESHOLD
        # ----------------------------------------------------

        matched = [

            (
                candidate_ids[idx],
                probabilities[pos]
            )

            for pos, idx
            in enumerate(candidate_indices)

            if probabilities[pos]
            >= THRESHOLD

        ]


        # ----------------------------------------------------
        # SORT BY MODEL PROBABILITY
        # ----------------------------------------------------

        matched.sort(
            key=lambda x: x[1],
            reverse=True
        )


        predicted_ids = [

            entity_id

            for entity_id, probability
            in matched

        ]


        batch_results.append({

            "source1_entity_id":
                source_id,

            "matched_entity_ids":
                ",".join(
                    predicted_ids
                )
        })


    # ========================================================
    # WRITE BATCH
    # ========================================================

    results_df = pd.DataFrame(
        batch_results
    )

    candidates_df_out = pd.DataFrame(
        batch_candidates
    )


    results_df.to_csv(
        OUTPUT_MATCHES,
        sep="\t",
        index=False,
        mode="w" if first_write else "a",
        header=first_write
    )


    candidates_df_out.to_csv(
        OUTPUT_CANDIDATES,
        sep="\t",
        index=False,
        mode="w" if first_write else "a",
        header=first_write
    )


    first_write = False


    # ========================================================
    # PROGRESS
    # ========================================================

    processed = batch_end

    elapsed = (
        time.time()
        - start_time
    )

    rate = (
        processed
        / elapsed
    )

    remaining = (
        total - processed
    ) / rate


    print(

        f"Processed "
        f"{processed:,}/"
        f"{total:,}"
        f" | {rate:.1f} rows/sec"
        f" | ETA "
        f"{remaining / 60:.1f} min"

    )


# ============================================================
# VALIDATION
# ============================================================

print("\n" + "=" * 60)
print("SUBMISSION VALIDATION")
print("=" * 60)


# Read output back
result_check = pd.read_csv(
    OUTPUT_MATCHES,
    sep="\t"
)


candidate_check = pd.read_csv(
    OUTPUT_CANDIDATES,
    sep="\t"
)


valid_ids = set(
    candidate_ids
)


invalid_ids = 0

for values in result_check[
    "matched_entity_ids"
].fillna(""):

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
    len(result_check)
)

print(
    "Candidate rows:",
    len(candidate_check)
)

print(
    "Duplicate Source 1 IDs:",
    result_check[
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
        result_check[
            "matched_entity_ids"
        ]
        .fillna("")
        .str.len()
        .gt(0)
        .sum()
    )
)


print("\n" + "=" * 60)
print("ML SUBMISSION READY")
print("=" * 60)

print(
    "Matching:",
    OUTPUT_MATCHES
)

print(
    "Candidates:",
    OUTPUT_CANDIDATES
)

print("\nDONE!")