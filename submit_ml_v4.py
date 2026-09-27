import os
import gc
import time
from collections import defaultdict, Counter

import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from lightgbm import Booster


# ============================================================
# CONFIG
# ============================================================

TEST_DIR = "dataset/test"

S1_FILE = os.path.join(TEST_DIR, "test_source1.tsv")
S2_FILE = os.path.join(TEST_DIR, "test_source2.tsv")
S3_FILE = os.path.join(TEST_DIR, "test_source3.tsv")

MODEL_FILE = "match_model_v2.txt"

OUTPUT_DIR = r"C:\MLAmazonOutput"

MATCH_FILE = os.path.join(
    OUTPUT_DIR,
    "matching_results.tsv"
)

CAND_FILE = os.path.join(
    OUTPUT_DIR,
    "candidate_pairs.tsv"
)

BATCH_SIZE = 2000

BASE_THRESHOLD = 0.75

MAX_ML_CANDIDATES = 30

MAX_POSTING = 500

MAX_TOKENS_PER_FIELD = 3


# ============================================================
# NORMALIZATION
# ============================================================

def norm(x):
    if pd.isna(x):
        return ""

    x = str(x).lower()

    out = []
    prev_space = False

    for ch in x:

        if ch.isalnum():
            out.append(ch)
            prev_space = False

        else:

            if not prev_space:
                out.append(" ")
                prev_space = True

    return "".join(out).strip()


def tokens(text):

    if not text:
        return []

    return [
        x
        for x in text.split()
        if len(x) >= 3
    ]


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("FINAL FAST ML ENTITY RESOLUTION")
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

print("Source 1:", len(s1))
print("Source 2:", len(s2))
print("Source 3:", len(s3))


# ============================================================
# NORMALIZE
# ============================================================

print("\nNormalizing...")

for df in (s1, s2, s3):

    df["name_norm"] = (
        df["business_name"]
        .map(norm)
    )

    df["address_norm"] = (
        df["business_address"]
        .map(norm)
    )

    df["country_norm"] = (
        df["country"]
        .map(norm)
    )


# ============================================================
# COMBINE SOURCE 2 + SOURCE 3
# ============================================================

print("Combining candidate sources...")

cand = pd.concat(
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

cand_ids = (
    cand["entity_id"]
    .astype(str)
    .to_numpy()
)

cand_names = (
    cand["name_norm"]
    .to_numpy()
)

cand_addresses = (
    cand["address_norm"]
    .to_numpy()
)

cand_countries = (
    cand["country_norm"]
    .to_numpy()
)

N = len(cand)

print(
    "Combined candidates:",
    N
)


# ============================================================
# EXACT INDEXES
# ============================================================

print("\nBuilding exact indexes...")

exact_name = defaultdict(list)

exact_address = defaultdict(list)

for i in range(N):

    name = cand_names[i]

    address = cand_addresses[i]

    if name:
        exact_name[name].append(i)

    if address:
        exact_address[address].append(i)


print(
    "Exact names:",
    len(exact_name)
)

print(
    "Exact addresses:",
    len(exact_address)
)


# ============================================================
# TOKEN FREQUENCIES
# ============================================================

print("\nCounting blocking tokens...")

name_freq = Counter()

address_freq = Counter()


for name in cand_names:

    seen = set(
        tokens(name)
    )

    for token in seen:
        name_freq[token] += 1


for address in cand_addresses:

    seen = set(
        tokens(address)
    )

    for token in seen:
        address_freq[token] += 1


# ============================================================
# BOUNDED TOKEN INDEXES
# ============================================================

print(
    "Building bounded token indexes..."
)

name_index = defaultdict(list)

address_index = defaultdict(list)


for i in range(N):

    name_tokens = tokens(
        cand_names[i]
    )

    name_tokens = sorted(
        name_tokens,
        key=lambda x:
            name_freq[x]
    )[:MAX_TOKENS_PER_FIELD]

    for token in name_tokens:

        if name_freq[token] <= MAX_POSTING:

            name_index[token].append(i)


for i in range(N):

    address_tokens = tokens(
        cand_addresses[i]
    )

    address_tokens = sorted(
        address_tokens,
        key=lambda x:
            address_freq[x]
    )[:MAX_TOKENS_PER_FIELD]

    for token in address_tokens:

        if address_freq[token] <= MAX_POSTING:

            address_index[token].append(i)


print(
    "Name token keys:",
    len(name_index)
)

print(
    "Address token keys:",
    len(address_index)
)

print("Indexes ready!")


# ============================================================
# LOAD MODEL
# ============================================================

print("\nLoading LightGBM...")

model = Booster(
    model_file=MODEL_FILE
)

print("Model loaded!")


# ============================================================
# CANDIDATE GENERATION
# ============================================================

def generate_candidates(
    name,
    address,
    country
):

    candidates = set()


    # --------------------------------------------------------
    # EXACT NAME
    # --------------------------------------------------------

    if name:

        for idx in exact_name.get(
            name,
            []
        ):

            candidates.add(idx)


    # --------------------------------------------------------
    # EXACT ADDRESS
    # --------------------------------------------------------

    if address:

        for idx in exact_address.get(
            address,
            []
        ):

            candidates.add(idx)


    # --------------------------------------------------------
    # NAME TOKEN BLOCKING
    # --------------------------------------------------------

    name_tokens = tokens(name)

    name_tokens = sorted(
        name_tokens,
        key=lambda x:
            name_freq.get(
                x,
                10**9
            )
    )[:MAX_TOKENS_PER_FIELD]


    for token in name_tokens:

        for idx in name_index.get(
            token,
            []
        ):

            candidates.add(idx)


    # --------------------------------------------------------
    # ADDRESS TOKEN BLOCKING
    # --------------------------------------------------------

    address_tokens = tokens(address)

    address_tokens = sorted(
        address_tokens,
        key=lambda x:
            address_freq.get(
                x,
                10**9
            )
    )[:MAX_TOKENS_PER_FIELD]


    for token in address_tokens:

        for idx in address_index.get(
            token,
            []
        ):

            candidates.add(idx)


    # --------------------------------------------------------
    # COUNTRY FILTER
    # --------------------------------------------------------

    if country and candidates:

        same_country = [

            i

            for i in candidates

            if cand_countries[i] == country

        ]

        if same_country:

            candidates = set(
                same_country
            )


    if not candidates:

        return []


    # --------------------------------------------------------
    # CHEAP PRE-RANKING
    # --------------------------------------------------------

    scored = []

    source_name_tokens = set(
        name_tokens
    )

    source_address_tokens = set(
        address_tokens
    )


    for idx in candidates:

        candidate_name = cand_names[idx]

        candidate_address = (
            cand_addresses[idx]
        )

        candidate_name_tokens = set(
            tokens(candidate_name)
        )

        candidate_address_tokens = set(
            tokens(candidate_address)
        )


        name_overlap = len(
            source_name_tokens
            &
            candidate_name_tokens
        )

        address_overlap = len(
            source_address_tokens
            &
            candidate_address_tokens
        )


        exact_bonus = 0


        if (
            name
            and
            name == candidate_name
        ):

            exact_bonus += 100


        if (
            address
            and
            address == candidate_address
        ):

            exact_bonus += 100


        score = (
            exact_bonus
            +
            name_overlap * 10
            +
            address_overlap * 8
        )


        scored.append(
            (
                idx,
                score
            )
        )


    scored.sort(
        key=lambda x: x[1],
        reverse=True
    )


    return [
        x[0]

        for x in
        scored[:MAX_ML_CANDIDATES]
    ]


# ============================================================
# FEATURE GENERATION
# ============================================================

def make_features(
    source_name,
    source_address,
    source_country,
    indices
):

    rows = []


    source_name_tokens = set(
        tokens(source_name)
    )

    source_address_tokens = set(
        tokens(source_address)
    )


    for idx in indices:

        name = cand_names[idx]

        address = cand_addresses[idx]

        country = cand_countries[idx]


        # ----------------------------------------------------
        # NAME FEATURES
        # ----------------------------------------------------

        if source_name and name:

            name_ratio = (
                fuzz.ratio(
                    source_name,
                    name
                ) / 100.0
            )

            name_token = (
                fuzz.token_set_ratio(
                    source_name,
                    name
                ) / 100.0
            )

            name_wratio = (
                fuzz.WRatio(
                    source_name,
                    name
                ) / 100.0
            )

        else:

            name_ratio = 0.0
            name_token = 0.0
            name_wratio = 0.0


        exact_name = int(
            bool(source_name)
            and
            source_name == name
        )


        # ----------------------------------------------------
        # ADDRESS FEATURES
        # ----------------------------------------------------

        if source_address and address:

            address_ratio = (
                fuzz.ratio(
                    source_address,
                    address
                ) / 100.0
            )

            address_token = (
                fuzz.token_set_ratio(
                    source_address,
                    address
                ) / 100.0
            )

            address_wratio = (
                fuzz.WRatio(
                    source_address,
                    address
                ) / 100.0
            )

        else:

            address_ratio = 0.0
            address_token = 0.0
            address_wratio = 0.0


        exact_address = int(
            bool(source_address)
            and
            source_address == address
        )


        # ----------------------------------------------------
        # TOKEN FEATURES
        # ----------------------------------------------------

        candidate_name_tokens = set(
            tokens(name)
        )

        candidate_address_tokens = set(
            tokens(address)
        )


        name_union = (
            source_name_tokens
            |
            candidate_name_tokens
        )

        address_union = (
            source_address_tokens
            |
            candidate_address_tokens
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

        name_len = (

            min(
                len(source_name),
                len(name)
            )

            /

            max(
                len(source_name),
                len(name),
                1
            )

        )


        address_len = (

            min(
                len(source_address),
                len(address)
            )

            /

            max(
                len(source_address),
                len(address),
                1
            )

        )


        # ----------------------------------------------------
        # COUNTRY
        # ----------------------------------------------------

        country_match = int(
            bool(source_country)
            and
            source_country == country
        )


        rows.append(
            [
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

                name_len,
                address_len,

                country_match
            ]
        )


    return np.asarray(
        rows,
        dtype=np.float32
    )


# ============================================================
# RESUME
# ============================================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


start_row = 0


if os.path.exists(MATCH_FILE):

    print(
        "\nChecking existing matching output..."
    )


    with open(
        MATCH_FILE,
        "r",
        encoding="utf-8"
    ) as f:

        existing_rows = (
            sum(
                1
                for _
                in f
            )
            - 1
        )


    if (
        0
        <
        existing_rows
        <
        len(s1)
    ):

        start_row = existing_rows

        print(
            f"Existing matching rows: "
            f"{existing_rows:,}"
        )

        print(
            f"Resuming from row: "
            f"{start_row:,}"
        )


    elif existing_rows >= len(s1):

        print(
            "Matching file already contains "
            "the full dataset."
        )

        start_row = len(s1)


# ============================================================
# CANDIDATE FILE
# ============================================================

# IMPORTANT:
# Do NOT delete the candidate file if it already
# contains valid rows from a previous successful run.
#
# If this is the first run with this script, the old
# candidate file should be removed manually if it was
# generated by the previous broken format.

if os.path.exists(CAND_FILE):

    print(
        "\nExisting candidate_pairs.tsv found."
    )

else:

    print(
        "\nCreating candidate_pairs.tsv..."
    )


# ============================================================
# DETERMINE HEADERS
# ============================================================

match_header = (
    not os.path.exists(MATCH_FILE)
)

candidate_header = (
    not os.path.exists(CAND_FILE)
)


# ============================================================
# INFERENCE
# ============================================================

print()
print("=" * 70)
print("INFERENCE START")
print("=" * 70)

print(
    f"Starting row: "
    f"{start_row:,}"
)

print(
    f"Remaining: "
    f"{len(s1) - start_row:,}"
)

print(
    f"Maximum ML candidates: "
    f"{MAX_ML_CANDIDATES}"
)

print(
    f"Base threshold: "
    f"{BASE_THRESHOLD}"
)

print("=" * 70)


start_time = time.time()


for batch_start in range(
    start_row,
    len(s1),
    BATCH_SIZE
):

    batch_end = min(
        batch_start + BATCH_SIZE,
        len(s1)
    )


    matches = []

    candidate_rows = []


    # ========================================================
    # PROCESS BATCH
    # ========================================================

    for row in s1.iloc[
        batch_start:batch_end
    ].itertuples(
        index=False
    ):

        sid = row.entity_id

        source_name = row.name_norm

        source_address = row.address_norm

        source_country = row.country_norm


        # ----------------------------------------------------
        # CANDIDATES
        # ----------------------------------------------------

        indices = generate_candidates(
            source_name,
            source_address,
            source_country
        )


        candidate_ids = [
            cand_ids[i]
            for i in indices
        ]


        candidate_rows.append(
            {
                "source1_entity_id":
                    sid,

                "candidate_entity_ids":
                    ",".join(candidate_ids)
            }
        )


        # ----------------------------------------------------
        # NO CANDIDATES
        # ----------------------------------------------------

        if not indices:

            matches.append(
                {
                    "source1_entity_id":
                        sid,

                    "matched_entity_ids":
                        ""
                }
            )

            continue


        # ----------------------------------------------------
        # FEATURES
        # ----------------------------------------------------

        X = make_features(
            source_name,
            source_address,
            source_country,
            indices
        )


        # ----------------------------------------------------
        # LIGHTGBM
        # ----------------------------------------------------

        probabilities = model.predict(
            X
        )


        ranked = sorted(
            zip(
                indices,
                probabilities
            ),
            key=lambda x: x[1],
            reverse=True
        )


        accepted = []


        # ----------------------------------------------------
        # DYNAMIC THRESHOLD
        # ----------------------------------------------------

        for idx, probability in ranked:

            candidate_name = (
                cand_names[idx]
            )

            candidate_address = (
                cand_addresses[idx]
            )


            exact_name_match = (
                bool(source_name)
                and
                source_name ==
                candidate_name
            )


            exact_address_match = (
                bool(source_address)
                and
                source_address ==
                candidate_address
            )


            if (
                exact_name_match
                and
                exact_address_match
            ):

                threshold = 0.55


            elif (
                exact_name_match
                or
                exact_address_match
            ):

                threshold = 0.65


            else:

                threshold = BASE_THRESHOLD


            if probability >= threshold:

                accepted.append(
                    (
                        cand_ids[idx],
                        float(probability)
                    )
                )


        accepted.sort(
            key=lambda x: x[1],
            reverse=True
        )


        accepted_ids = list(
            dict.fromkeys(
                x[0]
                for x in accepted
            )
        )


        matches.append(
            {
                "source1_entity_id":
                    sid,

                "matched_entity_ids":
                    ",".join(
                        accepted_ids
                    )
            }
        )


    # ========================================================
    # SAFE CHECKPOINT WRITE
    # ========================================================

    match_df = pd.DataFrame(
        matches
    )

    candidate_df = pd.DataFrame(
        candidate_rows
    )


    match_tmp = (
        MATCH_FILE
        +
        ".tmp"
    )

    candidate_tmp = (
        CAND_FILE
        +
        ".tmp"
    )


    # --------------------------------------------------------
    # WRITE TEMPORARY FILES
    # --------------------------------------------------------

    match_df.to_csv(
        match_tmp,
        sep="\t",
        index=False,
        header=True,
        encoding="utf-8"
    )


    candidate_df.to_csv(
        candidate_tmp,
        sep="\t",
        index=False,
        header=True,
        encoding="utf-8"
    )


    # ========================================================
    # APPEND MATCH RESULTS SAFELY
    # ========================================================

    with open(
        MATCH_FILE,
        "a",
        encoding="utf-8",
        newline=""
    ) as out:

        with open(
            match_tmp,
            "r",
            encoding="utf-8"
        ) as src:

            # Skip temporary header.
            next(src)

            for line in src:

                out.write(line)


    # ========================================================
    # APPEND CANDIDATES SAFELY
    # ========================================================

    with open(
        CAND_FILE,
        "a",
        encoding="utf-8",
        newline=""
    ) as out:

        with open(
            candidate_tmp,
            "r",
            encoding="utf-8"
        ) as src:

            # Skip temporary header.
            next(src)

            for line in src:

                out.write(line)


    # ========================================================
    # DELETE TEMPORARY FILES
    # ========================================================

    try:

        os.remove(
            match_tmp
        )

    except OSError:

        pass


    try:

        os.remove(
            candidate_tmp
        )

    except OSError:

        pass


    # ========================================================
    # PROGRESS
    # ========================================================

    processed = batch_end

    elapsed = (
        time.time()
        -
        start_time
    )


    done = (
        processed
        -
        start_row
    )


    rate = (

        done
        /
        elapsed

        if elapsed > 0

        else 0

    )


    remaining = (
        len(s1)
        -
        processed
    )


    eta = (

        remaining
        /
        rate
        /
        60

        if rate > 0

        else 0

    )


    print(
        f"Processed "
        f"{processed:,}/"
        f"{len(s1):,}"
        f" | {rate:.1f} rows/sec"
        f" | ETA {eta:.1f} min"
    )


    # ========================================================
    # MEMORY CLEANUP
    # ========================================================

    del match_df

    del candidate_df

    del matches

    del candidate_rows

    gc.collect()


# ============================================================
# FINAL VALIDATION
# ============================================================

print()
print("=" * 70)
print("FINAL VALIDATION")
print("=" * 70)


with open(
    MATCH_FILE,
    "r",
    encoding="utf-8"
) as f:

    match_count = (
        sum(
            1
            for _
            in f
        )
        - 1
    )


with open(
    CAND_FILE,
    "r",
    encoding="utf-8"
) as f:

    candidate_count = (
        sum(
            1
            for _
            in f
        )
        - 1
    )


print(
    "Expected Source-1 rows:",
    len(s1)
)

print(
    "Matching rows:",
    match_count
)

print(
    "Candidate rows:",
    candidate_count
)


if match_count == len(s1):

    print(
        "MATCHING FILE: OK"
    )

else:

    print(
        "MATCHING FILE: INCOMPLETE"
    )


if candidate_count == len(s1):

    print(
        "CANDIDATE FILE: OK"
    )

else:

    print(
        "CANDIDATE FILE: INCOMPLETE"
    )


print()
print("=" * 70)
print("FILES READY")
print("=" * 70)

print(
    MATCH_FILE
)

print(
    CAND_FILE
)

print(
    "\nDONE!"
)