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

TEST_BASE = "dataset/train"

SOURCE1 = f"{TEST_BASE}/train_source1.tsv"
SOURCE2 = f"{TEST_BASE}/train_source2.tsv"
SOURCE3 = f"{TEST_BASE}/train_source3.tsv"

MODEL_FILE = "match_model_v2.txt"

OUTPUT_MATCHES = "output/matching_results_dynamic_train.tsv"
OUTPUT_CANDIDATES = "output/candidate_pairs_dynamic_train.tsv"

# ------------------------------------------------------------
# TEST MODE
# ------------------------------------------------------------
# IMPORTANT:
# First run with 10,000.
# If it completes successfully, change to None for full data.
# ------------------------------------------------------------

TEST_ROWS = 1000

# Full run:
# TEST_ROWS = None


# ------------------------------------------------------------
# Candidate settings
# ------------------------------------------------------------

MAX_CANDIDATES = 50

# Number of candidates that actually go into LightGBM.
ML_CANDIDATES = 40

# Fallback threshold.
BASE_THRESHOLD = 0.75

BATCH_SIZE = 1000


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

print("=" * 70)
print("FAST DYNAMIC-THRESHOLD ML ENTITY RESOLUTION")
print("=" * 70)

print("\nLoading test data...")

s1 = pd.read_csv(SOURCE1, sep="\t")
s2 = pd.read_csv(SOURCE2, sep="\t")
s3 = pd.read_csv(SOURCE3, sep="\t")

print("Source 1:", len(s1))
print("Source 2:", len(s2))
print("Source 3:", len(s3))


# ============================================================
# TEST ROW LIMIT
# ============================================================

if TEST_ROWS is not None:
    s1 = s1.iloc[:TEST_ROWS].copy()

    print("\n" + "=" * 70)
    print(f"TEST MODE: {len(s1)} Source-1 rows")
    print("Change TEST_ROWS = None for the final full run.")
    print("=" * 70)
else:
    print("\nFULL DATA MODE")
    print("Source-1 rows:", len(s1))


# ============================================================
# NORMALIZATION
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

print("\nPreparing candidate data...")

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

print(
    "Total candidate records:",
    len(candidate_ids)
)


# ============================================================
# BLOCKING INDEXES
# ============================================================

print("\nBuilding high-recall blocking indexes...")

name6_index = defaultdict(list)
name4_index = defaultdict(list)
firstword_index = defaultdict(list)

address8_index = defaultdict(list)

for i in range(len(candidate_ids)):

    name = candidate_names[i]
    address = candidate_addresses[i]

    if name:

        name6_index[name[:6]].append(i)

        name4_index[name[:4]].append(i)

        first_word = name.split(" ", 1)[0]

        if first_word:
            firstword_index[first_word].append(i)

    if address:
        address8_index[address[:8]].append(i)


print(
    "Exact name/blocking keys:",
    len(name6_index)
)

print(
    "Name prefix keys:",
    len(name4_index)
)

print(
    "First-word keys:",
    len(firstword_index)
)

print(
    "Address keys:",
    len(address8_index)
)


# ============================================================
# LOAD MODEL
# ============================================================

print("\nLoading LightGBM model...")

model = Booster(
    model_file=MODEL_FILE
)

print("Model loaded!")


# ============================================================
# CANDIDATE GENERATION
# ============================================================

def get_candidates(
    source_name,
    source_address,
    source_country
):

    pool = set()

    # --------------------------------------------------------
    # NAME BLOCKING
    # --------------------------------------------------------

    if source_name:

        key6 = source_name[:6]

        pool.update(
            name6_index.get(key6, ())
        )

        key4 = source_name[:4]

        pool.update(
            name4_index.get(key4, ())
        )

        first_word = source_name.split(
            " ",
            1
        )[0]

        if first_word:

            pool.update(
                firstword_index.get(
                    first_word,
                    ()
                )
            )


    # --------------------------------------------------------
    # ADDRESS BLOCKING
    # --------------------------------------------------------

    if source_address:

        pool.update(
            address8_index.get(
                source_address[:8],
                ()
            )
        )


    if not pool:
        return []


    # --------------------------------------------------------
    # COUNTRY FILTER
    # --------------------------------------------------------

    if source_country:

        same_country = [
            i
            for i in pool
            if candidate_countries[i] == source_country
        ]

        if same_country:

            pool = set(same_country)


    if not pool:
        return []


    pool_list = list(pool)


    # --------------------------------------------------------
    # IMPORTANT OPTIMIZATION
    #
    # We ONLY run RapidFuzz on the blocked pool.
    # No cdist across the 9.9M records.
    # --------------------------------------------------------

    choices = [
        candidate_names[i]
        for i in pool_list
    ]


    # RapidFuzz version compatibility:
    # DO NOT use workers= here.
    matches = process.extract(
        source_name,
        choices,
        scorer=fuzz.ratio,
        limit=MAX_CANDIDATES
    )


    return [
        pool_list[m[2]]
        for m in matches
    ]


# ============================================================
# FAST FEATURE CALCULATION
# ============================================================

def build_features(
    source_name,
    source_address,
    source_country,
    indices
):

    n = len(indices)

    if n == 0:
        return None


    source_name_tokens = token_set(
        source_name
    )

    source_address_tokens = token_set(
        source_address
    )


    features = np.zeros(
        (n, 13),
        dtype=np.float32
    )


    for j, idx in enumerate(indices):

        candidate_name = candidate_names[idx]

        candidate_address = candidate_addresses[idx]

        candidate_country = candidate_countries[idx]


        # ----------------------------------------------------
        # NAME FEATURES
        # ----------------------------------------------------

        name_ratio = (
            fuzz.ratio(
                source_name,
                candidate_name
            ) / 100.0
        )

        name_token = (
            fuzz.token_set_ratio(
                source_name,
                candidate_name
            ) / 100.0
        )

        name_wratio = (
            fuzz.WRatio(
                source_name,
                candidate_name
            ) / 100.0
        )

        exact_name = int(
            source_name != ""
            and
            source_name == candidate_name
        )


        # ----------------------------------------------------
        # ADDRESS FEATURES
        # ----------------------------------------------------

        address_ratio = (
            fuzz.ratio(
                source_address,
                candidate_address
            ) / 100.0
        )

        address_token = (
            fuzz.token_set_ratio(
                source_address,
                candidate_address
            ) / 100.0
        )

        address_wratio = (
            fuzz.WRatio(
                source_address,
                candidate_address
            ) / 100.0
        )

        exact_address = int(
            source_address != ""
            and
            source_address == candidate_address
        )


        # ----------------------------------------------------
        # JACCARD FEATURES
        # ----------------------------------------------------

        candidate_name_tokens = token_set(
            candidate_name
        )

        name_union = (
            source_name_tokens
            |
            candidate_name_tokens
        )

        if name_union:

            name_jaccard = (
                len(
                    source_name_tokens
                    &
                    candidate_name_tokens
                )
                /
                len(name_union)
            )

        else:

            name_jaccard = 0.0


        candidate_address_tokens = token_set(
            candidate_address
        )

        address_union = (
            source_address_tokens
            |
            candidate_address_tokens
        )

        if address_union:

            address_jaccard = (
                len(
                    source_address_tokens
                    &
                    candidate_address_tokens
                )
                /
                len(address_union)
            )

        else:

            address_jaccard = 0.0


        # ----------------------------------------------------
        # LENGTH FEATURES
        # ----------------------------------------------------

        source_name_len = len(
            source_name
        )

        candidate_name_len = len(
            candidate_name
        )

        name_length_ratio = (
            min(
                source_name_len,
                candidate_name_len
            )
            /
            max(
                source_name_len,
                candidate_name_len,
                1
            )
        )


        source_address_len = len(
            source_address
        )

        candidate_address_len = len(
            candidate_address
        )

        address_length_ratio = (
            min(
                source_address_len,
                candidate_address_len
            )
            /
            max(
                source_address_len,
                candidate_address_len,
                1
            )
        )


        # ----------------------------------------------------
        # COUNTRY
        # ----------------------------------------------------

        country_match = int(
            source_country != ""
            and
            source_country == candidate_country
        )


        # ----------------------------------------------------
        # STORE FEATURES
        # ----------------------------------------------------

        features[j] = [

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


    return features


# ============================================================
# DYNAMIC THRESHOLD
# ============================================================

def calculate_dynamic_threshold(
    probabilities
):

    if len(probabilities) == 0:

        return BASE_THRESHOLD


    sorted_probs = np.sort(
        probabilities
    )[::-1]


    top = float(
        sorted_probs[0]
    )


    second = (
        float(sorted_probs[1])
        if len(sorted_probs) > 1
        else 0.0
    )


    margin = top - second


    # --------------------------------------------------------
    # HIGH CONFIDENCE
    #
    # If the model is very confident and clearly separates
    # the best candidates, allow slightly more recall.
    # --------------------------------------------------------

    if top >= 0.90 and margin >= 0.10:

        return 0.70


    if top >= 0.85 and margin >= 0.08:

        return 0.72


    if top >= 0.80 and margin >= 0.06:

        return 0.74


    # --------------------------------------------------------
    # NORMAL CASE
    # --------------------------------------------------------

    if top >= 0.75:

        return 0.75


    # --------------------------------------------------------
    # WEAK MODEL CONFIDENCE
    #
    # Don't flood the output with weak candidates.
    # --------------------------------------------------------

    return 0.78


# ============================================================
# INFERENCE
# ============================================================

print("\n" + "=" * 70)
print("FAST DYNAMIC-THRESHOLD ML INFERENCE")
print("=" * 70)

print(
    "Candidate limit:",
    MAX_CANDIDATES
)

print(
    "ML candidates:",
    ML_CANDIDATES
)

print(
    "Fallback threshold:",
    BASE_THRESHOLD
)

print(
    "Rows:",
    len(s1)
)

print("=" * 70)


os.makedirs(
    "output",
    exist_ok=True
)


# Remove old dynamic outputs

for path in [
    OUTPUT_MATCHES,
    OUTPUT_CANDIDATES
]:

    if os.path.exists(path):

        try:
            os.remove(path)

        except PermissionError:

            print(
                "\nERROR:"
            )

            print(
                "Close matching_results_dynamic.tsv "
                "or candidate_pairs_dynamic.tsv "
                "before running."
            )

            raise


total = len(s1)

start_time = time.time()

first_write = True


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
        # BLOCKING
        # ----------------------------------------------------

        candidate_indices = get_candidates(
            source_name,
            source_address,
            source_country
        )


        # Keep only top MAX_CANDIDATES
        candidate_indices = candidate_indices[
            :MAX_CANDIDATES
        ]


        candidate_ids_for_row = [
            candidate_ids[i]
            for i in candidate_indices
        ]


        # Save candidate list

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
        # ML SHORTLIST
        #
        # Candidate generation may return up to 50.
        # ML only sees 40.
        # ----------------------------------------------------

        ml_indices = candidate_indices[
            :ML_CANDIDATES
        ]


        # ----------------------------------------------------
        # FEATURE BUILDING
        #
        # IMPORTANT:
        # This is now only 40 candidates.
        # ----------------------------------------------------

        X = build_features(

            source_name,

            source_address,

            source_country,

            ml_indices
        )


        if X is None:

            batch_results.append({

                "source1_entity_id":
                    source_id,

                "matched_entity_ids":
                    ""
            })

            continue


        # ----------------------------------------------------
        # LIGHTGBM
        # ----------------------------------------------------

        probabilities = model.predict(
            X
        )


        probabilities = np.asarray(
            probabilities,
            dtype=np.float32
        )


        # ----------------------------------------------------
        # DYNAMIC THRESHOLD
        # ----------------------------------------------------

        dynamic_threshold = (
            calculate_dynamic_threshold(
                probabilities
            )
        )


        # ----------------------------------------------------
        # MATCHES
        # ----------------------------------------------------

        matched = []

        for idx, prob in zip(
            ml_indices,
            probabilities
        ):

            if float(prob) >= dynamic_threshold:

                matched.append(
                    (
                        candidate_ids[idx],
                        float(prob)
                    )
                )


        # Highest probability first

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

    candidates_out_df = pd.DataFrame(
        batch_candidates
    )


    results_df.to_csv(

        OUTPUT_MATCHES,

        sep="\t",

        index=False,

        mode=(
            "w"
            if first_write
            else "a"
        ),

        header=first_write
    )


    candidates_out_df.to_csv(

        OUTPUT_CANDIDATES,

        sep="\t",

        index=False,

        mode=(
            "w"
            if first_write
            else "a"
        ),

        header=first_write
    )


    first_write = False


    # ========================================================
    # PROGRESS
    # ========================================================

    processed = batch_end

    elapsed = (
        time.time()
        -
        start_time
    )

    rate = (
        processed
        /
        max(elapsed, 0.001)
    )

    remaining = (
        total
        -
        processed
    )

    eta_seconds = (
        remaining
        /
        max(rate, 0.001)
    )


    print(

        f"Processed "
        f"{processed:,}/"
        f"{total:,}"
        f" | "
        f"{rate:.1f} rows/sec"
        f" | ETA "
        f"{eta_seconds / 60:.1f} min"

    )


# ============================================================
# VALIDATION
# ============================================================

print("\n" + "=" * 70)
print("FINAL VALIDATION")
print("=" * 70)


result_check = pd.read_csv(

    OUTPUT_MATCHES,

    sep="\t",

    keep_default_na=False
)


candidate_check = pd.read_csv(

    OUTPUT_CANDIDATES,

    sep="\t",

    keep_default_na=False
)


valid_ids = set(

    s2.entity_id.astype(str)
)

valid_ids.update(

    s3.entity_id.astype(str)
)


invalid_ids = 0

matches_outside_candidates = 0


for matched_values, candidate_values in zip(

    result_check[
        "matched_entity_ids"
    ],

    candidate_check[
        "candidate_entity_ids"
    ]

):

    matched_set = set(

        x.strip()

        for x in matched_values.split(",")

        if x.strip()
    )


    candidate_set = set(

        x.strip()

        for x in candidate_values.split(",")

        if x.strip()
    )


    invalid_ids += sum(

        entity_id not in valid_ids

        for entity_id in matched_set
    )


    matches_outside_candidates += sum(

        entity_id not in candidate_set

        for entity_id in matched_set
    )


print(
    "Expected Source-1 rows:",
    len(s1)
)

print(
    "Matching rows:",
    len(result_check)
)

print(
    "Candidate rows:",
    len(candidate_check)
)

print(
    "Duplicate Source-1 IDs:",
    result_check[
        "source1_entity_id"
    ].duplicated().sum()
)

print(
    "Invalid predicted IDs:",
    invalid_ids
)

print(
    "Matches outside candidate set:",
    matches_outside_candidates
)

print(
    "Rows containing matches:",
    result_check[
        "matched_entity_ids"
    ]
    .fillna("")
    .str.len()
    .gt(0)
    .sum()
)


print("\n" + "=" * 70)
print("FILES READY")
print("=" * 70)

print(
    OUTPUT_MATCHES
)

print(
    OUTPUT_CANDIDATES
)

print("\nDONE!")