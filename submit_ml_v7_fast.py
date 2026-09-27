import os
import re
import time
from collections import defaultdict

import numpy as np
import pandas as pd

from rapidfuzz import fuzz, process
from lightgbm import Booster


# ============================================================
# V7 FAST FINAL SUBMISSION
# ============================================================

TEST_DIR = "dataset/test"

S1_FILE = os.path.join(
    TEST_DIR,
    "test_source1.tsv"
)

S2_FILE = os.path.join(
    TEST_DIR,
    "test_source2.tsv"
)

S3_FILE = os.path.join(
    TEST_DIR,
    "test_source3.tsv"
)

# ------------------------------------------------------------
# SAME MODEL THAT GAVE 93.13% VALIDATION RECALL@1
# ------------------------------------------------------------

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
# CONFIGURATION
# ============================================================

# FIRST RUN:
# Keep this at 10000.
#
# If 10000 rows completes quickly and correctly,
# change this to 0 for the full test set.
#
TEST_ROWS = 10000

MAX_CANDIDATES = 100

MODEL_CANDIDATES = 40

MAX_NAME_POSTING = 300

MAX_ADDRESS_POSTING = 300

MAX_TOKEN_POSTING = 300

THRESHOLD = 0.50

# Write output in batches instead of once per row.
WRITE_BATCH = 1000


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


def token_set(text):

    if not text:
        return set()

    return set(
        text.split()
    )


def useful_tokens(text):

    return {
        x
        for x in token_set(text)
        if len(x) >= 3
    }


def first_word(text):

    if not text:
        return ""

    return text.split()[0]


def last_word(text):

    if not text:
        return ""

    return text.split()[-1]


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


def jaccard(a, b):

    if not a or not b:
        return 0.0

    union = a | b

    if not union:
        return 0.0

    return len(a & b) / len(union)


# ============================================================
# FEATURE MATRIX
# ============================================================

def build_feature_matrix(
    source_name,
    source_address,
    source_country,
    candidate_names,
    candidate_addresses,
    candidate_countries,
    model_features
):

    source_name_tokens = token_set(
        source_name
    )

    source_address_tokens = token_set(
        source_address
    )

    source_name_first = first_word(
        source_name
    )

    source_name_last = last_word(
        source_name
    )

    source_name_len = len(
        source_name
    )

    source_address_len = len(
        source_address
    )

    source_name_token_count = len(
        source_name_tokens
    )

    source_address_token_count = len(
        source_address_tokens
    )

    source_number = address_number(
        source_address
    )

    rows = []

    for cname, caddress, ccountry in zip(
        candidate_names,
        candidate_addresses,
        candidate_countries
    ):

        cname_tokens = token_set(
            cname
        )

        caddress_tokens = token_set(
            caddress
        )

        cname_first = first_word(
            cname
        )

        cname_last = last_word(
            cname
        )

        cname_len = len(
            cname
        )

        caddress_len = len(
            caddress
        )

        cname_token_count = len(
            cname_tokens
        )

        caddress_token_count = len(
            caddress_tokens
        )

        candidate_number = address_number(
            caddress
        )

        # ----------------------------------------------------
        # FUZZY FEATURES
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # JACCARD
        # ----------------------------------------------------

        name_jaccard = jaccard(
            source_name_tokens,
            cname_tokens
        )

        address_jaccard = jaccard(
            source_address_tokens,
            caddress_tokens
        )

        # ----------------------------------------------------
        # EXACT
        # ----------------------------------------------------

        name_exact = int(
            source_name != ""
            and
            source_name == cname
        )

        address_exact = int(
            source_address != ""
            and
            source_address == caddress
        )

        country_exact = int(
            source_country != ""
            and
            source_country == ccountry
        )

        # ----------------------------------------------------
        # PREFIX / SUFFIX
        # ----------------------------------------------------

        name_prefix3 = int(
            len(source_name) >= 3
            and
            len(cname) >= 3
            and
            source_name[:3] == cname[:3]
        )

        name_prefix5 = int(
            len(source_name) >= 5
            and
            len(cname) >= 5
            and
            source_name[:5] == cname[:5]
        )

        name_suffix3 = int(
            len(source_name) >= 3
            and
            len(cname) >= 3
            and
            source_name[-3:] == cname[-3:]
        )

        address_prefix5 = int(
            len(source_address) >= 5
            and
            len(caddress) >= 5
            and
            source_address[:5] == caddress[:5]
        )

        address_suffix5 = int(
            len(source_address) >= 5
            and
            len(caddress) >= 5
            and
            source_address[-5:] == caddress[-5:]
        )

        # ----------------------------------------------------
        # WORD FEATURES
        # ----------------------------------------------------

        name_first_word = int(
            source_name_first != ""
            and
            source_name_first == cname_first
        )

        name_last_word = int(
            source_name_last != ""
            and
            source_name_last == cname_last
        )

        # ----------------------------------------------------
        # LENGTH FEATURES
        # ----------------------------------------------------

        name_length_diff = abs(
            source_name_len -
            cname_len
        )

        address_length_diff = abs(
            source_address_len -
            caddress_len
        )

        name_token_count_diff = abs(
            source_name_token_count -
            cname_token_count
        )

        address_token_count_diff = abs(
            source_address_token_count -
            caddress_token_count
        )

        # ----------------------------------------------------
        # ADDRESS NUMBER
        # ----------------------------------------------------

        address_number_match = int(
            source_number != ""
            and
            candidate_number != ""
            and
            source_number == candidate_number
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
                address_number_match,

            "name_jaccard":
                name_jaccard,

            "name_prefix3":
                name_prefix3,

            "name_length_diff":
                name_length_diff,

            "address_length_diff":
                address_length_diff,

            "name_last_word":
                name_last_word,

            "name_token_count_diff":
                name_token_count_diff,

            "address_token_count_diff":
                address_token_count_diff,

            "address_prefix5":
                address_prefix5,

            "address_suffix5":
                address_suffix5,

            "name_prefix5":
                name_prefix5,

            "name_suffix3":
                name_suffix3,

            "name_first_word":
                name_first_word,

            "name_exact":
                name_exact,

            "address_exact":
                address_exact,

            "country_exact":
                country_exact
        }

        rows.append(
            [
                feature_map.get(
                    feature,
                    0.0
                )
                for feature in model_features
            ]
        )

    if not rows:

        return np.empty(
            (0, len(model_features)),
            dtype=np.float32
        )

    return np.asarray(
        rows,
        dtype=np.float32
    )


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("V7 FAST FINAL SUBMISSION")
print("=" * 70)

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

print(
    "Source 1:",
    len(s1)
)

print(
    "Source 2:",
    len(s2)
)

print(
    "Source 3:",
    len(s3)
)


# ============================================================
# TEST LIMIT
# ============================================================

if TEST_ROWS > 0:

    s1 = s1.iloc[
        :TEST_ROWS
    ].copy()

    print()
    print("=" * 70)
    print(
        "TEST MODE:",
        len(s1),
        "Source-1 rows"
    )
    print("=" * 70)

else:

    print()
    print("=" * 70)
    print("FULL TEST MODE")
    print(
        "Source-1 rows:",
        len(s1)
    )
    print("=" * 70)


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
# CANDIDATE DATA
# ============================================================

print(
    "\nPreparing candidate data..."
)

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

N = len(
    candidate_ids
)

print(
    "Total candidate records:",
    N
)


# ============================================================
# BLOCKING INDEXES
# ============================================================

print(
    "\nBuilding blocking indexes..."
)

name_exact = defaultdict(list)

address_exact = defaultdict(list)

name_prefix3 = defaultdict(list)

name_prefix5 = defaultdict(list)

address_prefix5 = defaultdict(list)

address_suffix5 = defaultdict(list)

first_word_index = defaultdict(list)

last_word_index = defaultdict(list)

country_index = defaultdict(list)

name_token_index = defaultdict(list)

address_token_index = defaultdict(list)


# ------------------------------------------------------------
# BUILD INDEXES
# ------------------------------------------------------------

index_start = time.time()

for i in range(N):

    name = candidate_names[i]

    address = candidate_addresses[i]

    country = candidate_countries[i]

    # Exact name
    if name:

        name_exact[name].append(
            i
        )

    # Exact address
    if address:

        address_exact[address].append(
            i
        )

    # Name prefixes
    if len(name) >= 3:

        name_prefix3[
            name[:3]
        ].append(i)

    if len(name) >= 5:

        name_prefix5[
            name[:5]
        ].append(i)

    # Address prefix/suffix
    if len(address) >= 5:

        address_prefix5[
            address[:5]
        ].append(i)

        address_suffix5[
            address[-5:]
        ].append(i)

    # Words
    fw = first_word(
        name
    )

    lw = last_word(
        name
    )

    if fw:

        first_word_index[
            fw
        ].append(i)

    if lw:

        last_word_index[
            lw
        ].append(i)

    # Country
    if country:

        country_index[
            country
        ].append(i)

    # Name tokens
    for token in useful_tokens(name):

        posting = name_token_index[
            token
        ]

        if len(posting) < MAX_TOKEN_POSTING:

            posting.append(i)

    # Address tokens
    for token in useful_tokens(address):

        posting = address_token_index[
            token
        ]

        if len(posting) < MAX_TOKEN_POSTING:

            posting.append(i)


print(
    "Exact name keys:",
    len(name_exact)
)

print(
    "Exact address keys:",
    len(address_exact)
)

print(
    "Name token keys:",
    len(name_token_index)
)

print(
    "Address token keys:",
    len(address_token_index)
)

print(
    "Index build time:",
    f"{time.time() - index_start:.1f}s"
)


# ============================================================
# LOAD MODEL
# ============================================================

print(
    "\nLoading V7 LightGBM model..."
)

model = Booster(
    model_file=MODEL_FILE
)

model_features = model.feature_name()

print(
    "Model loaded!"
)

print(
    "Model features:",
    model_features
)


# ============================================================
# GENERATE CANDIDATES
# ============================================================

def generate_candidates(
    source_name,
    source_address,
    source_country
):

    pool = set()

    # --------------------------------------------------------
    # EXACT NAME
    # --------------------------------------------------------

    if source_name:

        pool.update(
            name_exact.get(
                source_name,
                []
            )
        )

    # --------------------------------------------------------
    # EXACT ADDRESS
    # --------------------------------------------------------

    if source_address:

        pool.update(
            address_exact.get(
                source_address,
                []
            )
        )

    # --------------------------------------------------------
    # NAME TOKENS
    # --------------------------------------------------------

    name_tokens = list(
        useful_tokens(
            source_name
        )
    )

    name_tokens.sort(
        key=lambda x:
        len(
            name_token_index.get(
                x,
                []
            )
        )
    )

    for token in name_tokens[:5]:

        pool.update(
            name_token_index.get(
                token,
                []
            )
        )

        if len(pool) >= 500:

            break

    # --------------------------------------------------------
    # ADDRESS TOKENS
    # --------------------------------------------------------

    address_tokens = list(
        useful_tokens(
            source_address
        )
    )

    address_tokens.sort(
        key=lambda x:
        len(
            address_token_index.get(
                x,
                []
            )
        )
    )

    for token in address_tokens[:5]:

        pool.update(
            address_token_index.get(
                token,
                []
            )
        )

        if len(pool) >= 500:

            break

    # --------------------------------------------------------
    # PREFIX BLOCKS
    # --------------------------------------------------------

    if len(source_name) >= 3:

        pool.update(
            name_prefix3.get(
                source_name[:3],
                []
            )
        )

    if len(source_name) >= 5:

        pool.update(
            name_prefix5.get(
                source_name[:5],
                []
            )
        )

    if len(source_address) >= 5:

        pool.update(
            address_prefix5.get(
                source_address[:5],
                []
            )
        )

        pool.update(
            address_suffix5.get(
                source_address[-5:],
                []
            )
        )

    # --------------------------------------------------------
    # WORD BLOCKS
    # --------------------------------------------------------

    fw = first_word(
        source_name
    )

    if fw:

        values = first_word_index.get(
            fw,
            []
        )

        if len(values) <= MAX_NAME_POSTING:

            pool.update(
                values
            )

    lw = last_word(
        source_name
    )

    if lw:

        values = last_word_index.get(
            lw,
            []
        )

        if len(values) <= MAX_NAME_POSTING:

            pool.update(
                values
            )

    # --------------------------------------------------------
    # COUNTRY FALLBACK
    # --------------------------------------------------------

    if len(pool) < 10 and source_country:

        values = country_index.get(
            source_country,
            []
        )

        if len(values) <= 5000:

            pool.update(
                values
            )

    return pool


# ============================================================
# OUTPUT SETUP
# ============================================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

if os.path.exists(
    MATCH_FILE
):

    os.remove(
        MATCH_FILE
    )

if os.path.exists(
    CANDIDATE_FILE
):

    os.remove(
        CANDIDATE_FILE
)


# ============================================================
# INFERENCE
# ============================================================

print()
print("=" * 70)
print("V7 FAST INFERENCE")
print("=" * 70)

print(
    "Rows:",
    len(s1)
)

print(
    "Maximum candidates:",
    MAX_CANDIDATES
)

print(
    "ML candidates:",
    MODEL_CANDIDATES
)

print(
    "Threshold:",
    THRESHOLD
)

print(
    "Output batch size:",
    WRITE_BATCH
)

print("=" * 70)


start_time = time.time()

total = len(s1)

processed = 0

match_buffer = []

candidate_buffer = []


# ============================================================
# FAST INFERENCE LOOP
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
    # CANDIDATES
    # --------------------------------------------------------

    pool = generate_candidates(
        source_name,
        source_address,
        source_country
    )


    # --------------------------------------------------------
    # SOURCE 2 / SOURCE 3 ONLY
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
    # FAST QUICK RANKING
    #
    # OLD:
    #   Python loop + fuzz.ratio() for every candidate
    #
    # NEW:
    #   RapidFuzz C implementation + parallel workers
    # --------------------------------------------------------

    if len(pool) > MODEL_CANDIDATES:

        pool_array = np.asarray(
            pool,
            dtype=np.int32
        )

        pool_names = [
            candidate_names[i]
            for i in pool_array
        ]

        pool_addresses = [
            candidate_addresses[i]
            for i in pool_array
        ]

        name_scores = process.cdist(
            [source_name],
            pool_names,
            scorer=fuzz.ratio,
            workers=-1
        )[0]

        address_scores = process.cdist(
            [source_address],
            pool_addresses,
            scorer=fuzz.ratio,
            workers=-1
        )[0]

        quick_scores = (
            0.55 * name_scores
            +
            0.45 * address_scores
        )

        # Same ordering principle as the old
        # score-descending / index-descending sort.
        order = np.lexsort(
            (
                -pool_array,
                -quick_scores
            )
        )

        top_order = order[
            :MODEL_CANDIDATES
        ]

        shortlist = [
            int(pool_array[i])
            for i in top_order
        ]

    else:

        shortlist = pool


    # --------------------------------------------------------
    # ML FEATURES
    # --------------------------------------------------------

    candidate_names_short = [
        candidate_names[i]
        for i in shortlist
    ]

    candidate_addresses_short = [
        candidate_addresses[i]
        for i in shortlist
    ]

    candidate_countries_short = [
        candidate_countries[i]
        for i in shortlist
    ]

    X = build_feature_matrix(
        source_name,
        source_address,
        source_country,
        candidate_names_short,
        candidate_addresses_short,
        candidate_countries_short,
        model_features
    )


    # --------------------------------------------------------
    # LIGHTGBM PREDICTION
    # --------------------------------------------------------

    if len(shortlist) > 0:

        scores = model.predict(
            X
        )

        scores = np.asarray(
            scores
        )

        # Descending score.
        # Stable sort preserves candidate order
        # when scores are tied.
        order = np.argsort(
            -scores,
            kind="stable"
        )

        ranked = [
            (
                shortlist[i],
                float(scores[i])
            )
            for i in order
        ]

    else:

        ranked = []


    # --------------------------------------------------------
    # FINAL MATCHES
    # --------------------------------------------------------

    matches = []

    for idx, score in ranked:

        if score >= THRESHOLD:

            matches.append(
                (
                    candidate_ids[idx],
                    score
                )
            )


    # --------------------------------------------------------
    # EXACT MATCH SAFETY
    # --------------------------------------------------------

    exact_ids = set()

    for idx in shortlist:

        if (
            source_name != ""
            and
            source_name ==
            candidate_names[idx]
        ):

            exact_ids.add(
                candidate_ids[idx]
            )

        if (
            source_address != ""
            and
            source_address ==
            candidate_addresses[idx]
        ):

            exact_ids.add(
                candidate_ids[idx]
            )


    matched_existing = {
        x[0]
        for x in matches
    }

    for entity_id in exact_ids:

        if entity_id not in matched_existing:

            matches.append(
                (
                    entity_id,
                    1.0
                )
            )


    matches.sort(
        key=lambda x: x[1],
        reverse=True
    )


    # --------------------------------------------------------
    # FINAL 100 CANDIDATES
    # --------------------------------------------------------

    candidate_ranked = [
        idx
        for idx, _ in ranked
    ]

    seen = set(
        candidate_ranked
    )

    for idx in pool:

        if len(candidate_ranked) >= MAX_CANDIDATES:

            break

        if idx not in seen:

            candidate_ranked.append(
                idx
            )

            seen.add(
                idx
            )


    candidate_ids_output = [
        candidate_ids[idx]
        for idx in candidate_ranked[
            :MAX_CANDIDATES
        ]
    ]


    # --------------------------------------------------------
    # BUFFER OUTPUT
    # --------------------------------------------------------

    match_buffer.append(
        {
            "source1_entity_id":
                source_id,

            "matched_entity_ids":
                ",".join(
                    x[0]
                    for x in matches
                )
        }
    )

    candidate_buffer.append(
        {
            "source1_entity_id":
                source_id,

            "candidate_entity_ids":
                ",".join(
                    candidate_ids_output
                )
        }
    )


    processed += 1


    # --------------------------------------------------------
    # WRITE EVERY 1000 ROWS
    # --------------------------------------------------------

    if (
        len(match_buffer)
        >= WRITE_BATCH
    ):

        pd.DataFrame(
            match_buffer
        ).to_csv(
            MATCH_FILE,
            sep="\t",
            index=False,
            mode="a",
            header=(
                not os.path.exists(
                    MATCH_FILE
                )
            )
        )

        pd.DataFrame(
            candidate_buffer
        ).to_csv(
            CANDIDATE_FILE,
            sep="\t",
            index=False,
            mode="a",
            header=(
                not os.path.exists(
                    CANDIDATE_FILE
                )
            )
        )

        match_buffer.clear()

        candidate_buffer.clear()


    # --------------------------------------------------------
    # PROGRESS
    # --------------------------------------------------------

    if processed % 500 == 0:

        elapsed = (
            time.time()
            -
            start_time
        )

        rate = (
            processed /
            max(
                elapsed,
                0.001
            )
        )

        remaining = (
            total -
            processed
        )

        eta_seconds = (
            remaining /
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
            f"{eta_seconds / 60:.1f} min"
        )


# ============================================================
# FLUSH REMAINING OUTPUT
# ============================================================

if match_buffer:

    pd.DataFrame(
        match_buffer
    ).to_csv(
        MATCH_FILE,
        sep="\t",
        index=False,
        mode="a",
        header=(
            not os.path.exists(
                MATCH_FILE
            )
        )
    )

if candidate_buffer:

    pd.DataFrame(
        candidate_buffer
    ).to_csv(
        CANDIDATE_FILE,
        sep="\t",
        index=False,
        mode="a",
        header=(
            not os.path.exists(
                CANDIDATE_FILE
            )
        )
    )


# ============================================================
# FINAL CHECK
# ============================================================

elapsed_total = (
    time.time()
    -
    start_time
)

print()
print("=" * 70)
print("FINAL OUTPUT CHECK")
print("=" * 70)

matches_out = pd.read_csv(
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
    "matching_results rows:",
    len(matches_out)
)

print(
    "candidate_pairs rows:",
    len(candidates_out)
)

print(
    "Expected Source-1 rows:",
    len(s1)
)

print(
    "Duplicate Source-1 IDs:",
    matches_out[
        "source1_entity_id"
    ].duplicated().sum()
)

print(
    "Empty candidate rows:",
    (
        candidates_out[
            "candidate_entity_ids"
        ] == ""
    ).sum()
)

print(
    "Total inference time:",
    f"{elapsed_total / 60:.2f} min"
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

print()
print("=" * 70)
print("V7 FAST SUBMISSION GENERATION COMPLETE")
print("=" * 70)