import os
import re
import time
from collections import defaultdict

import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from lightgbm import Booster


# ============================================================
# FINAL FAST SUBMISSION
# ============================================================

TEST_DIR = "dataset/test"

S1_FILE = os.path.join(TEST_DIR, "test_source1.tsv")
S2_FILE = os.path.join(TEST_DIR, "test_source2.tsv")
S3_FILE = os.path.join(TEST_DIR, "test_source3.tsv")

MODEL_FILE = "v7_lgbm_final.txt"

OUTPUT_DIR = "output"

MATCH_FILE = os.path.join(
    OUTPUT_DIR,
    "matching_results.tsv"
)

CANDIDATE_FILE = os.path.join(
    OUTPUT_DIR,
    "candidate_pairs.tsv"
)

# ============================================================
# FULL RUN
# ============================================================

# 0 = FULL TEST DATA
TEST_ROWS = 0

# Maximum candidates written per row
MAX_CANDIDATES = 100

# Maximum candidates sent to LightGBM
MODEL_CANDIDATES = 40

# Maximum posting size for first-word fallback
MAX_FIRST_WORD_POSTING = 300

# Threshold used by the already validated V7 model
THRESHOLD = 0.50


# ============================================================
# NORMALIZATION
# ============================================================

def normalize(value):

    if pd.isna(value):
        return ""

    value = str(value).lower()

    value = re.sub(
        r"[^a-z0-9\s]",
        " ",
        value
    )

    value = re.sub(
        r"\s+",
        " ",
        value
    ).strip()

    return value


def first_word(text):

    if not text:
        return ""

    return text.split()[0]


def address_number(text):

    if not text:
        return ""

    m = re.search(
        r"\b\d{1,6}\b",
        text
    )

    if m:
        return m.group(0)

    return ""


def token_set(text):

    if not text:
        return set()

    return set(text.split())


def jaccard(a, b):

    if not a or not b:
        return 0.0

    union = a | b

    if not union:
        return 0.0

    return len(a & b) / len(union)


# ============================================================
# FEATURES
# ============================================================

def build_features(
    source_name,
    source_address,
    source_country,
    candidate_names,
    candidate_addresses,
    candidate_countries
):

    source_name_tokens = token_set(source_name)
    source_address_tokens = token_set(source_address)

    source_first = first_word(source_name)

    source_name_len = len(source_name)
    source_address_len = len(source_address)

    source_name_count = len(source_name_tokens)
    source_address_count = len(source_address_tokens)

    source_number = address_number(
        source_address
    )

    features = []

    for cname, caddress, ccountry in zip(
        candidate_names,
        candidate_addresses,
        candidate_countries
    ):

        cname_tokens = token_set(cname)
        caddress_tokens = token_set(caddress)

        cname_first = first_word(cname)

        candidate_number = address_number(
            caddress
        )

        name_ratio = (
            fuzz.ratio(
                source_name,
                cname
            ) / 100.0
        )

        name_token_ratio = (
            fuzz.token_set_ratio(
                source_name,
                cname
            ) / 100.0
        )

        address_ratio = (
            fuzz.ratio(
                source_address,
                caddress
            ) / 100.0
        )

        address_token_ratio = (
            fuzz.token_set_ratio(
                source_address,
                caddress
            ) / 100.0
        )

        name_jaccard = jaccard(
            source_name_tokens,
            cname_tokens
        )

        address_jaccard = jaccard(
            source_address_tokens,
            caddress_tokens
        )

        feature_map = {

            "address_token_ratio":
                address_token_ratio,

            "address_jaccard":
                address_jaccard,

            "address_ratio":
                address_ratio,

            "name_token_ratio":
                name_token_ratio,

            "name_ratio":
                name_ratio,

            "address_number_match":
                int(
                    source_number != ""
                    and
                    candidate_number != ""
                    and
                    source_number == candidate_number
                ),

            "name_jaccard":
                name_jaccard,

            "name_prefix3":
                int(
                    len(source_name) >= 3
                    and
                    len(cname) >= 3
                    and
                    source_name[:3] == cname[:3]
                ),

            "name_length_diff":
                abs(
                    source_name_len -
                    len(cname)
                ),

            "address_length_diff":
                abs(
                    source_address_len -
                    len(caddress)
                ),

            "name_last_word":
                int(
                    source_name.split()[-1]
                    ==
                    cname.split()[-1]
                    if source_name and cname
                    else False
                ),

            "name_token_count_diff":
                abs(
                    source_name_count -
                    len(cname_tokens)
                ),

            "address_token_count_diff":
                abs(
                    source_address_count -
                    len(caddress_tokens)
                ),

            "address_prefix5":
                int(
                    len(source_address) >= 5
                    and
                    len(caddress) >= 5
                    and
                    source_address[:5]
                    ==
                    caddress[:5]
                ),

            "address_suffix5":
                int(
                    len(source_address) >= 5
                    and
                    len(caddress) >= 5
                    and
                    source_address[-5:]
                    ==
                    caddress[-5:]
                ),

            "name_prefix5":
                int(
                    len(source_name) >= 5
                    and
                    len(cname) >= 5
                    and
                    source_name[:5]
                    ==
                    cname[:5]
                ),

            "name_suffix3":
                int(
                    len(source_name) >= 3
                    and
                    len(cname) >= 3
                    and
                    source_name[-3:]
                    ==
                    cname[-3:]
                ),

            "name_first_word":
                int(
                    source_first != ""
                    and
                    source_first == cname_first
                ),

            "name_exact":
                int(
                    source_name != ""
                    and
                    source_name == cname
                ),

            "address_exact":
                int(
                    source_address != ""
                    and
                    source_address == caddress
                ),

            "country_exact":
                int(
                    source_country != ""
                    and
                    source_country == ccountry
                )
        }

        features.append(
            feature_map
        )

    return features


# ============================================================
# START
# ============================================================

print("=" * 70)
print("FINAL FAST V7 SUBMISSION")
print("=" * 70)

start_total = time.time()


# ============================================================
# LOAD DATA
# ============================================================

print("\nLoading datasets...")

s1 = pd.read_csv(
    S1_FILE,
    sep="\t"
)

s2 = pd.read_csv(
    S2_FILE,
    sep="\t"
)

s3 = pd.read_csv(
    S3_FILE,
    sep="\t"
)

print("Source 1:", len(s1))
print("Source 2:", len(s2))
print("Source 3:", len(s3))


if TEST_ROWS > 0:

    s1 = s1.iloc[
        :TEST_ROWS
    ].copy()

    print(
        "\nTEST MODE:",
        len(s1)
    )

else:

    print(
        "\nFULL TEST MODE:",
        len(s1)
    )


# ============================================================
# NORMALIZE
# ============================================================

print("\nNormalizing...")

for df in (
    s1,
    s2,
    s3
):

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
# PREPARE CANDIDATES
# ============================================================

print("\nPreparing candidate data...")

candidates = pd.concat(
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
    candidates["entity_id"]
    .astype(str)
    .to_numpy()
)

candidate_names = (
    candidates["name_norm"]
    .fillna("")
    .astype(str)
    .to_numpy()
)

candidate_addresses = (
    candidates["address_norm"]
    .fillna("")
    .astype(str)
    .to_numpy()
)

candidate_countries = (
    candidates["country_norm"]
    .fillna("")
    .astype(str)
    .to_numpy()
)

N = len(candidate_ids)

print(
    "Candidate records:",
    N
)


# ============================================================
# FAST INDEXES ONLY
#
# IMPORTANT:
# NO TOKEN INDEXES.
# NO RARE TOKEN INDEXES.
# NO COUNTRY+PREFIX INDEXES.
#
# This is the speed fix.
# ============================================================

print(
    "\nBuilding FAST indexes..."
)

name_exact = defaultdict(list)

address_exact = defaultdict(list)

first_word_index = defaultdict(list)

name_prefix5_index = defaultdict(list)

address_number_index = defaultdict(list)


# ------------------------------------------------------------
# BUILD INDEXES
# ------------------------------------------------------------

index_start = time.time()

for i in range(N):

    name = candidate_names[i]

    address = candidate_addresses[i]

    # Exact name
    if name:

        values = name_exact[name]

        if len(values) < MAX_CANDIDATES:

            values.append(i)

    # Exact address
    if address:

        values = address_exact[address]

        if len(values) < MAX_CANDIDATES:

            values.append(i)

    # First word
    fw = first_word(name)

    if fw:

        values = first_word_index[fw]

        if len(values) < MAX_FIRST_WORD_POSTING:

            values.append(i)

    # Name prefix 5
    if len(name) >= 5:

        key = name[:5]

        values = name_prefix5_index[key]

        if len(values) < MAX_CANDIDATES:

            values.append(i)

    # Address number
    number = address_number(address)

    if number:

        values = address_number_index[number]

        if len(values) < MAX_CANDIDATES:

            values.append(i)


print(
    "Exact-name keys:",
    len(name_exact)
)

print(
    "Exact-address keys:",
    len(address_exact)
)

print(
    "First-word keys:",
    len(first_word_index)
)

print(
    "Name-prefix keys:",
    len(name_prefix5_index)
)

print(
    "Address-number keys:",
    len(address_number_index)
)

print(
    "Index time:",
    round(
        time.time() - index_start,
        1
    ),
    "seconds"
)


# ============================================================
# LOAD LIGHTGBM
# ============================================================

print(
    "\nLoading V7 LightGBM model..."
)

model = Booster(
    model_file=MODEL_FILE
)

model_features = model.feature_name()

print(
    "Model loaded."
)

print(
    "Features:",
    len(model_features)
)


# ============================================================
# OUTPUT
# ============================================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

if os.path.exists(MATCH_FILE):

    os.remove(
        MATCH_FILE
    )

if os.path.exists(CANDIDATE_FILE):

    os.remove(
        CANDIDATE_FILE
    )


# ============================================================
# INFERENCE
# ============================================================

print()
print("=" * 70)
print("FAST INFERENCE")
print("=" * 70)

total = len(s1)

processed = 0

start = time.time()

match_header = True

candidate_header = True


# ============================================================
# PROCESS ROWS
# ============================================================

for row in s1.itertuples(
    index=False
):

    source_id = str(
        row.entity_id
    )

    source_name = row.name_norm

    source_address = row.address_norm

    source_country = row.country_norm


    # --------------------------------------------------------
    # CANDIDATE GENERATION
    # --------------------------------------------------------

    pool = set()


    # Exact name
    if source_name:

        pool.update(
            name_exact.get(
                source_name,
                []
            )
        )


    # Exact address
    if source_address:

        pool.update(
            address_exact.get(
                source_address,
                []
            )
        )


    # If exact signals are weak, use cheap fallback blocks
    if len(pool) < 5:

        fw = first_word(
            source_name
        )

        if fw:

            pool.update(
                first_word_index.get(
                    fw,
                    []
                )
            )


    if len(pool) < 5 and len(source_name) >= 5:

        pool.update(
            name_prefix5_index.get(
                source_name[:5],
                []
            )
        )


    # Address number fallback
    if len(pool) < 5:

        number = address_number(
            source_address
        )

        if number:

            pool.update(
                address_number_index.get(
                    number,
                    []
                )
            )


    # --------------------------------------------------------
    # KEEP ONLY S2/S3
    # --------------------------------------------------------

    pool = [
        x
        for x in pool
        if (
            candidate_ids[x].startswith("S2-")
            or
            candidate_ids[x].startswith("S3-")
        )
    ]


    # --------------------------------------------------------
    # HARD LIMIT
    # --------------------------------------------------------

    if len(pool) > MAX_CANDIDATES:

        pool = pool[
            :MAX_CANDIDATES
        ]


    # --------------------------------------------------------
    # EMPTY
    # --------------------------------------------------------

    if not pool:

        matches_text = ""

        candidate_text = ""


    else:

        # ----------------------------------------------------
        # EXACT MATCHES
        #
        # These don't need LightGBM.
        # ----------------------------------------------------

        exact_matches = []

        non_exact = []

        for idx in pool:

            exact_name = (
                source_name != ""
                and
                source_name
                ==
                candidate_names[idx]
            )

            exact_address = (
                source_address != ""
                and
                source_address
                ==
                candidate_addresses[idx]
            )

            if exact_name or exact_address:

                exact_matches.append(
                    candidate_ids[idx]
                )

            else:

                non_exact.append(idx)


        # ----------------------------------------------------
        # LIGHTGBM ONLY FOR AMBIGUOUS / FUZZY CANDIDATES
        # ----------------------------------------------------

        ranked_matches = []


        if non_exact:

            # Cheap pre-ranking
            quick_scores = []

            for idx in non_exact:

                nr = fuzz.ratio(
                    source_name,
                    candidate_names[idx]
                )

                ar = fuzz.ratio(
                    source_address,
                    candidate_addresses[idx]
                )

                score = (
                    0.55 * nr
                    +
                    0.45 * ar
                )

                quick_scores.append(
                    (
                        score,
                        idx
                    )
                )


            quick_scores.sort(
                reverse=True
            )


            shortlist = [
                idx
                for _, idx
                in quick_scores[
                    :MODEL_CANDIDATES
                ]
            ]


            # Feature creation
            feature_maps = build_features(
                source_name,
                source_address,
                source_country,

                [
                    candidate_names[i]
                    for i in shortlist
                ],

                [
                    candidate_addresses[i]
                    for i in shortlist
                ],

                [
                    candidate_countries[i]
                    for i in shortlist
                ]
            )


            X = np.asarray(
                [
                    [
                        fmap.get(
                            feature,
                            0.0
                        )

                        for feature in model_features

                    ]

                    for fmap in feature_maps
                ],

                dtype=np.float32
            )


            if len(shortlist):

                scores = model.predict(
                    X
                )

                ranked = sorted(
                    zip(
                        shortlist,
                        scores
                    ),

                    key=lambda x: x[1],

                    reverse=True
                )


                for idx, score in ranked:

                    if float(score) >= THRESHOLD:

                        ranked_matches.append(
                            (
                                candidate_ids[idx],
                                float(score)
                            )
                        )


        # ----------------------------------------------------
        # FINAL MATCH LIST
        # ----------------------------------------------------

        final_matches = []


        # Exact matches first
        for entity_id in exact_matches:

            if entity_id not in final_matches:

                final_matches.append(
                    entity_id
                )


        # ML matches
        for entity_id, score in ranked_matches:

            if entity_id not in final_matches:

                final_matches.append(
                    entity_id
                )


        # Limit
        final_matches = final_matches[
            :MAX_CANDIDATES
        ]


        matches_text = ",".join(
            final_matches
        )


        candidate_text = ",".join(
            candidate_ids[idx]
            for idx in pool
        )


    # --------------------------------------------------------
    # WRITE
    # --------------------------------------------------------

    with open(
        MATCH_FILE,
        "a",
        encoding="utf-8"
    ) as f:

        if match_header:

            f.write(
                "source1_entity_id\tmatched_entity_ids\n"
            )

            match_header = False

        f.write(
            source_id
            + "\t"
            + matches_text
            + "\n"
        )


    with open(
        CANDIDATE_FILE,
        "a",
        encoding="utf-8"
    ) as f:

        if candidate_header:

            f.write(
                "source1_entity_id\tcandidate_entity_ids\n"
            )

            candidate_header = False

        f.write(
            source_id
            + "\t"
            + candidate_text
            + "\n"
        )


    processed += 1


    # --------------------------------------------------------
    # PROGRESS
    # --------------------------------------------------------

    if processed % 5000 == 0:

        elapsed = (
            time.time()
            - start
        )

        rate = (
            processed
            /
            max(
                elapsed,
                0.001
            )
        )

        remaining = (
            total
            -
            processed
        )

        eta = (
            remaining
            /
            max(
                rate,
                0.001
            )
        )

        print(
            f"Processed "
            f"{processed:,}/"
            f"{total:,}"
            f" | "
            f"{rate:.1f} rows/sec"
            f" | ETA "
            f"{eta / 60:.1f} min"
        )


# ============================================================
# FINAL CHECK
# ============================================================

print()
print("=" * 70)
print("FINAL OUTPUT CHECK")
print("=" * 70)

matches = pd.read_csv(
    MATCH_FILE,
    sep="\t",
    dtype=str,
    keep_default_na=False
)

candidates_out = pd.read_csv(
    CANDIDATE_FILE,
    sep="\t",
    dtype=str,
    keep_default_na=False
)

print(
    "Matching rows:",
    len(matches)
)

print(
    "Candidate rows:",
    len(candidates_out)
)

print(
    "Expected rows:",
    len(s1)
)

print(
    "Duplicate Source-1 IDs:",
    matches[
        "source1_entity_id"
    ].duplicated().sum()
)

print(
    "Empty match rows:",
    (
        matches[
            "matched_entity_ids"
        ]
        == ""
    ).sum()
)

print(
    "Empty candidate rows:",
    (
        candidates_out[
            "candidate_entity_ids"
        ]
        == ""
    ).sum()
)

print()
print(
    "Saved:",
    MATCH_FILE
)

print(
    "Saved:",
    CANDIDATE_FILE
)

print(
    "\nTotal runtime:",
    round(
        time.time()
        -
        start_total,
        1
    ),
    "seconds"
)

print()
print("=" * 70)
print("FINAL FAST SUBMISSION COMPLETE")
print("=" * 70)