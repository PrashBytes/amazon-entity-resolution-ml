import pandas as pd
import re
import time
import unicodedata
from collections import defaultdict, Counter
from rapidfuzz.fuzz import ratio


# ============================================================
# V8 FAST HIGH-RECALL CANDIDATE GENERATION
# ============================================================

print("=" * 75)
print("V8 FAST HIGH-RECALL CANDIDATE GENERATION")
print("NO FULL TF-IDF SEARCH")
print("=" * 75)


# ============================================================
# SETTINGS
# ============================================================

TRAIN_BASE = "dataset/train"

SOURCE1_FILE = f"{TRAIN_BASE}/train_source1.tsv"
SOURCE2_FILE = f"{TRAIN_BASE}/train_source2.tsv"
SOURCE3_FILE = f"{TRAIN_BASE}/train_source3.tsv"

OUTPUT_FILE = "v8_fast_candidates.tsv"

# IMPORTANT:
# Keep this at 1000 while experimenting.
# Change to None ONLY for the final run.
TEST_ROWS = 1000

# Maximum candidates finally saved per Source-1 record
TOP_CANDIDATES = 100

# We can collect more than 100 before ranking.
# This helps recall.
MAX_BLOCK_CANDIDATES = 500

# Individual blocking limits
MAX_TOKEN_POSTINGS = 250
MAX_FIRSTWORD_POSTINGS = 500
MAX_PREFIX_POSTINGS = 500
MAX_ADDRESS_POSTINGS = 500

# Minimum token length
MIN_TOKEN_LENGTH = 2


# ============================================================
# NORMALIZATION
# ============================================================

def normalize_text(text):

    if pd.isna(text):
        return ""

    text = unicodedata.normalize(
        "NFKD",
        str(text)
    )

    text = (
        text
        .encode("ascii", "ignore")
        .decode("ascii")
    )

    text = text.lower()

    text = re.sub(
        r"[^a-z0-9\s]",
        " ",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    return text


def tokenize(text):

    if not text:
        return []

    return [
        token
        for token in text.split()
        if len(token) >= MIN_TOKEN_LENGTH
    ]


def sorted_name(text):

    tokens = tokenize(text)

    if not tokens:
        return ""

    return " ".join(
        sorted(tokens)
    )


def first_word(text):

    tokens = tokenize(text)

    if not tokens:
        return ""

    return tokens[0]


def last_word(text):

    tokens = tokenize(text)

    if not tokens:
        return ""

    return tokens[-1]


def address_numbers(text):

    if not text:
        return []

    return re.findall(
        r"\d+",
        text
    )


# ============================================================
# LOAD DATA
# ============================================================

start_total = time.time()

print()
print("Loading datasets...")

s1 = pd.read_csv(
    SOURCE1_FILE,
    sep="\t",
    dtype=str
)

s2 = pd.read_csv(
    SOURCE2_FILE,
    sep="\t",
    dtype=str
)

s3 = pd.read_csv(
    SOURCE3_FILE,
    sep="\t",
    dtype=str
)

print("Source 1:", len(s1))
print("Source 2:", len(s2))
print("Source 3:", len(s3))


# ============================================================
# TEST MODE
# ============================================================

if TEST_ROWS is not None:

    print()
    print("=" * 75)
    print(f"TEST MODE: {TEST_ROWS} Source-1 rows")
    print("=" * 75)
    print("Change TEST_ROWS = None only after testing.")

    s1 = s1.head(TEST_ROWS).copy()

else:

    print()
    print("=" * 75)
    print("FULL MODE")
    print("=" * 75)


# ============================================================
# NORMALIZATION
# ============================================================

print()
print("Normalizing...")

norm_start = time.time()


for df in [s1, s2, s3]:

    df["name_norm"] = (
        df["business_name"]
        .fillna("")
        .apply(normalize_text)
    )

    df["address_norm"] = (
        df["business_address"]
        .fillna("")
        .apply(normalize_text)
    )

    if "country" in df.columns:

        df["country_norm"] = (
            df["country"]
            .fillna("")
            .astype(str)
            .str.lower()
            .str.strip()
        )

    else:

        df["country_norm"] = ""


print(
    f"Normalization time: "
    f"{time.time() - norm_start:.1f}s"
)


# ============================================================
# PREPARE CANDIDATE DATA
# ============================================================

print()
print("Preparing candidate data...")

candidates = pd.concat(
    [
        s2,
        s3
    ],
    ignore_index=True
)

# Remember which source each candidate came from.
candidate_sources = (
    ["source2"] * len(s2)
    +
    ["source3"] * len(s3)
)

candidates["candidate_source"] = candidate_sources

print(
    "Total candidate records:",
    len(candidates)
)


# ============================================================
# PRECOMPUTE CANDIDATE FEATURES
# ============================================================

print()
print("Preparing candidate features...")

feature_start = time.time()

candidates["name_tokens"] = (
    candidates["name_norm"]
    .apply(tokenize)
)

candidates["address_tokens"] = (
    candidates["address_norm"]
    .apply(tokenize)
)

candidates["sorted_name"] = (
    candidates["name_norm"]
    .apply(sorted_name)
)

candidates["first_word"] = (
    candidates["name_norm"]
    .apply(first_word)
)

candidates["last_word"] = (
    candidates["name_norm"]
    .apply(last_word)
)

candidates["name_prefix3"] = (
    candidates["name_norm"]
    .str[:3]
)

candidates["name_prefix4"] = (
    candidates["name_norm"]
    .str[:4]
)

candidates["name_prefix5"] = (
    candidates["name_norm"]
    .str[:5]
)

candidates["name_suffix4"] = (
    candidates["name_norm"]
    .str[-4:]
)

candidates["name_suffix5"] = (
    candidates["name_norm"]
    .str[-5:]
)

candidates["address_numbers"] = (
    candidates["address_norm"]
    .apply(address_numbers)
)


# ============================================================
# BUILD BLOCKING INDEX
# ============================================================

print()
print("Building optimized blocking indexes...")

index_start = time.time()


def add_to_index(index, key, idx):

    if not key:
        return

    index[key].append(idx)


exact_name_index = defaultdict(list)
sorted_name_index = defaultdict(list)

first_word_index = defaultdict(list)
last_word_index = defaultdict(list)

prefix3_index = defaultdict(list)
prefix4_index = defaultdict(list)
prefix5_index = defaultdict(list)

suffix4_index = defaultdict(list)
suffix5_index = defaultdict(list)

country_first_index = defaultdict(list)
country_last_index = defaultdict(list)

country_prefix3_index = defaultdict(list)
country_prefix4_index = defaultdict(list)

address_number_index = defaultdict(list)

name_token_index = defaultdict(list)
address_token_index = defaultdict(list)

name_bigram_index = defaultdict(list)


# ============================================================
# TOKEN FREQUENCIES
# ============================================================

print("Calculating token frequencies...")

name_token_frequency = Counter()
address_token_frequency = Counter()

for tokens in candidates["name_tokens"]:

    for token in set(tokens):

        name_token_frequency[token] += 1


for tokens in candidates["address_tokens"]:

    for token in set(tokens):

        address_token_frequency[token] += 1


# ============================================================
# BUILD INDEXES
# ============================================================

for idx, row in candidates.iterrows():

    name = row["name_norm"]
    address = row["address_norm"]
    country = row["country_norm"]

    tokens = row["name_tokens"]

    # --------------------------------------------------------
    # Exact name
    # --------------------------------------------------------

    if name:

        add_to_index(
            exact_name_index,
            name,
            idx
        )


    # --------------------------------------------------------
    # Sorted name
    # --------------------------------------------------------

    sorted_nm = row["sorted_name"]

    if sorted_nm:

        add_to_index(
            sorted_name_index,
            sorted_nm,
            idx
        )


    # --------------------------------------------------------
    # First / last word
    # --------------------------------------------------------

    fw = row["first_word"]
    lw = row["last_word"]

    add_to_index(
        first_word_index,
        fw,
        idx
    )

    add_to_index(
        last_word_index,
        lw,
        idx
    )


    # --------------------------------------------------------
    # Prefix / suffix
    # --------------------------------------------------------

    p3 = row["name_prefix3"]
    p4 = row["name_prefix4"]
    p5 = row["name_prefix5"]

    s4 = row["name_suffix4"]
    s5 = row["name_suffix5"]

    add_to_index(
        prefix3_index,
        p3,
        idx
    )

    add_to_index(
        prefix4_index,
        p4,
        idx
    )

    add_to_index(
        prefix5_index,
        p5,
        idx
    )

    add_to_index(
        suffix4_index,
        s4,
        idx
    )

    add_to_index(
        suffix5_index,
        s5,
        idx
    )


    # --------------------------------------------------------
    # Country combinations
    # --------------------------------------------------------

    if country and fw:

        add_to_index(
            country_first_index,
            country + "|" + fw,
            idx
        )


    if country and lw:

        add_to_index(
            country_last_index,
            country + "|" + lw,
            idx
        )


    if country and p3:

        add_to_index(
            country_prefix3_index,
            country + "|" + p3,
            idx
        )


    if country and p4:

        add_to_index(
            country_prefix4_index,
            country + "|" + p4,
            idx
        )


    # --------------------------------------------------------
    # Address numbers
    # --------------------------------------------------------

    for number in set(
        row["address_numbers"]
    ):

        add_to_index(
            address_number_index,
            number,
            idx
        )


    # --------------------------------------------------------
    # Name tokens
    # --------------------------------------------------------

    for token in set(tokens):

        if (
            name_token_frequency[token]
            <= MAX_TOKEN_POSTINGS
        ):

            add_to_index(
                name_token_index,
                token,
                idx
            )


    # --------------------------------------------------------
    # Address tokens
    # --------------------------------------------------------

    for token in set(
        row["address_tokens"]
    ):

        if (
            address_token_frequency[token]
            <= MAX_TOKEN_POSTINGS
        ):

            add_to_index(
                address_token_index,
                token,
                idx
            )


    # --------------------------------------------------------
    # Name bigrams
    # --------------------------------------------------------

    unique_tokens = list(
        dict.fromkeys(tokens)
    )

    if len(unique_tokens) >= 2:

        # Only adjacent pairs.
        for a, b in zip(
            unique_tokens,
            unique_tokens[1:]
        ):

            if (
                name_token_frequency[a]
                <= MAX_TOKEN_POSTINGS
                and
                name_token_frequency[b]
                <= MAX_TOKEN_POSTINGS
            ):

                bigram = (
                    a + "|" + b
                )

                add_to_index(
                    name_bigram_index,
                    bigram,
                    idx
                )


print(
    "Exact names:",
    len(exact_name_index)
)

print(
    "Sorted names:",
    len(sorted_name_index)
)

print(
    "First words:",
    len(first_word_index)
)

print(
    "Last words:",
    len(last_word_index)
)

print(
    "Prefix indexes:",
    len(prefix4_index)
)

print(
    "Suffix indexes:",
    len(suffix5_index)
)

print(
    "Name token index:",
    len(name_token_index)
)

print(
    "Address token index:",
    len(address_token_index)
)

print(
    "Name bigram index:",
    len(name_bigram_index)
)

print(
    f"Index build time: "
    f"{time.time() - index_start:.1f}s"
)


# ============================================================
# CANDIDATE COLLECTION
# ============================================================

def collect_candidates(row):

    candidates_set = set()

    name = row["name_norm"]
    address = row["address_norm"]
    country = row["country_norm"]

    tokens = row["name_tokens"]

    # --------------------------------------------------------
    # EXACT NAME
    # --------------------------------------------------------

    if name:

        candidates_set.update(
            exact_name_index.get(
                name,
                []
            )
        )


    # --------------------------------------------------------
    # SORTED NAME
    # --------------------------------------------------------

    sorted_nm = sorted_name(name)

    if sorted_nm:

        candidates_set.update(
            sorted_name_index.get(
                sorted_nm,
                []
            )
        )


    # --------------------------------------------------------
    # PREFIXES
    # --------------------------------------------------------

    p3 = name[:3]
    p4 = name[:4]
    p5 = name[:5]

    for idx in prefix3_index.get(
        p3,
        []
    )[:MAX_PREFIX_POSTINGS]:

        candidates_set.add(idx)

    for idx in prefix4_index.get(
        p4,
        []
    )[:MAX_PREFIX_POSTINGS]:

        candidates_set.add(idx)

    for idx in prefix5_index.get(
        p5,
        []
    )[:MAX_PREFIX_POSTINGS]:

        candidates_set.add(idx)


    # --------------------------------------------------------
    # SUFFIXES
    # --------------------------------------------------------

    s4 = name[-4:]
    s5 = name[-5:]

    for idx in suffix4_index.get(
        s4,
        []
    )[:MAX_PREFIX_POSTINGS]:

        candidates_set.add(idx)

    for idx in suffix5_index.get(
        s5,
        []
    )[:MAX_PREFIX_POSTINGS]:

        candidates_set.add(idx)


    # --------------------------------------------------------
    # FIRST / LAST WORD
    # --------------------------------------------------------

    fw = first_word(name)
    lw = last_word(name)

    for idx in first_word_index.get(
        fw,
        []
    )[:MAX_FIRSTWORD_POSTINGS]:

        candidates_set.add(idx)


    for idx in last_word_index.get(
        lw,
        []
    )[:MAX_FIRSTWORD_POSTINGS]:

        candidates_set.add(idx)


    # --------------------------------------------------------
    # COUNTRY + WORD
    # --------------------------------------------------------

    if country and fw:

        key = country + "|" + fw

        candidates_set.update(
            country_first_index.get(
                key,
                []
            )[:MAX_FIRSTWORD_POSTINGS]
        )


    if country and lw:

        key = country + "|" + lw

        candidates_set.update(
            country_last_index.get(
                key,
                []
            )[:MAX_FIRSTWORD_POSTINGS]
        )


    # --------------------------------------------------------
    # COUNTRY + PREFIX
    # --------------------------------------------------------

    if country and p3:

        key = country + "|" + p3

        candidates_set.update(
            country_prefix3_index.get(
                key,
                []
            )[:MAX_PREFIX_POSTINGS]
        )


    if country and p4:

        key = country + "|" + p4

        candidates_set.update(
            country_prefix4_index.get(
                key,
                []
            )[:MAX_PREFIX_POSTINGS]
        )


    # --------------------------------------------------------
    # RARE NAME TOKENS
    # --------------------------------------------------------

    # Prefer rarest tokens first.
    name_tokens_sorted = sorted(
        set(tokens),
        key=lambda x:
        name_token_frequency.get(
            x,
            10**9
        )
    )

    for token in name_tokens_sorted[:5]:

        postings = name_token_index.get(
            token,
            []
        )

        candidates_set.update(
            postings[:MAX_TOKEN_POSTINGS]
        )


    # --------------------------------------------------------
    # NAME BIGRAMS
    # --------------------------------------------------------

    unique_tokens = list(
        dict.fromkeys(tokens)
    )

    if len(unique_tokens) >= 2:

        for a, b in zip(
            unique_tokens,
            unique_tokens[1:]
        ):

            key = a + "|" + b

            postings = name_bigram_index.get(
                key,
                []
            )

            candidates_set.update(
                postings[:MAX_TOKEN_POSTINGS]
            )


    # --------------------------------------------------------
    # ADDRESS TOKENS
    # --------------------------------------------------------

    address_tokens = tokenize(
        address
    )

    address_tokens_sorted = sorted(
        set(address_tokens),
        key=lambda x:
        address_token_frequency.get(
            x,
            10**9
        )
    )

    for token in address_tokens_sorted[:5]:

        postings = address_token_index.get(
            token,
            []
        )

        candidates_set.update(
            postings[:MAX_ADDRESS_POSTINGS]
        )


    # --------------------------------------------------------
    # ADDRESS NUMBERS
    # --------------------------------------------------------

    numbers = address_numbers(
        address
    )

    for number in set(numbers):

        postings = address_number_index.get(
            number,
            []
        )

        candidates_set.update(
            postings[:MAX_ADDRESS_POSTINGS]
        )


    # --------------------------------------------------------
    # LIMIT BEFORE EXPENSIVE RANKING
    # --------------------------------------------------------

    if len(candidates_set) <= MAX_BLOCK_CANDIDATES:

        return candidates_set


    # If there are too many candidates,
    # retain candidates that have stronger cheap signals.

    scored = []

    source_tokens = set(tokens)
    source_address_tokens = set(
        address_tokens
    )

    for idx in candidates_set:

        candidate = candidates.iloc[idx]

        score = 0

        candidate_name = candidate[
            "name_norm"
        ]

        candidate_tokens = set(
            candidate["name_tokens"]
        )

        # Exact name
        if name and candidate_name == name:

            score += 100


        # Same first word
        if (
            fw
            and candidate["first_word"]
            == fw
        ):

            score += 15


        # Same last word
        if (
            lw
            and candidate["last_word"]
            == lw
        ):

            score += 10


        # Prefix
        if (
            p4
            and candidate["name_prefix4"]
            == p4
        ):

            score += 12


        # Suffix
        if (
            s4
            and candidate["name_suffix4"]
            == s4
        ):

            score += 8


        # Country
        if (
            country
            and candidate["country_norm"]
            == country
        ):

            score += 8


        # Token overlap
        score += (
            5
            *
            len(
                source_tokens
                &
                candidate_tokens
            )
        )


        # Address overlap
        candidate_address_tokens = set(
            candidate["address_tokens"]
        )

        score += (
            3
            *
            len(
                source_address_tokens
                &
                candidate_address_tokens
            )
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

    return {
        idx
        for idx, score
        in scored[
            :MAX_BLOCK_CANDIDATES
        ]
    }


# ============================================================
# FAST RANKING
# ============================================================

def rank_candidates(
    row,
    candidate_indices
):

    results = []

    source_name = row[
        "name_norm"
    ]

    source_address = row[
        "address_norm"
    ]

    source_country = row[
        "country_norm"
    ]

    source_name_tokens = set(
        row["name_tokens"]
    )

    source_address_tokens = set(
        tokenize(source_address)
    )

    source_first = first_word(
        source_name
    )

    source_last = last_word(
        source_name
    )


    for idx in candidate_indices:

        candidate = candidates.iloc[
            idx
        ]

        candidate_name = candidate[
            "name_norm"
        ]

        candidate_address = candidate[
            "address_norm"
        ]

        candidate_country = candidate[
            "country_norm"
        ]


        # ----------------------------------------------------
        # FUZZY NAME
        # ----------------------------------------------------

        name_ratio = ratio(
            source_name,
            candidate_name
        )


        # ----------------------------------------------------
        # FUZZY ADDRESS
        # ----------------------------------------------------

        address_ratio = ratio(
            source_address,
            candidate_address
        )


        # ----------------------------------------------------
        # NAME TOKEN OVERLAP
        # ----------------------------------------------------

        candidate_name_tokens = set(
            candidate["name_tokens"]
        )

        if source_name_tokens:

            name_overlap_count = len(
                source_name_tokens
                &
                candidate_name_tokens
            )

            name_overlap = (
                name_overlap_count
                /
                max(
                    1,
                    len(source_name_tokens)
                )
            )

            candidate_name_coverage = (
                name_overlap_count
                /
                max(
                    1,
                    len(candidate_name_tokens)
                )
            )

        else:

            name_overlap = 0
            candidate_name_coverage = 0


        # ----------------------------------------------------
        # ADDRESS TOKEN OVERLAP
        # ----------------------------------------------------

        candidate_address_tokens = set(
            candidate["address_tokens"]
        )

        if source_address_tokens:

            address_overlap_count = len(
                source_address_tokens
                &
                candidate_address_tokens
            )

            address_overlap = (
                address_overlap_count
                /
                max(
                    1,
                    len(source_address_tokens)
                )
            )

            address_coverage = (
                address_overlap_count
                /
                max(
                    1,
                    len(candidate_address_tokens)
                )
            )

        else:

            address_overlap = 0
            address_coverage = 0


        # ----------------------------------------------------
        # EXACT / STRUCTURAL BONUSES
        # ----------------------------------------------------

        exact_name_bonus = (
            25
            if (
                source_name
                and
                source_name
                ==
                candidate_name
            )
            else 0
        )


        sorted_name_bonus = (
            15
            if (
                sorted_name(source_name)
                and
                sorted_name(source_name)
                ==
                candidate["sorted_name"]
            )
            else 0
        )


        first_word_bonus = (
            6
            if (
                source_first
                and
                source_first
                ==
                candidate["first_word"]
            )
            else 0
        )


        last_word_bonus = (
            5
            if (
                source_last
                and
                source_last
                ==
                candidate["last_word"]
            )
            else 0
        )


        prefix_bonus = (
            5
            if (
                source_name[:4]
                and
                source_name[:4]
                ==
                candidate["name_prefix4"]
            )
            else 0
        )


        suffix_bonus = (
            4
            if (
                source_name[-4:]
                and
                source_name[-4:]
                ==
                candidate["name_suffix4"]
            )
            else 0
        )


        country_bonus = (
            6
            if (
                source_country
                and
                candidate_country
                and
                source_country
                ==
                candidate_country
            )
            else 0
        )


        # ----------------------------------------------------
        # FINAL SCORE
        # ----------------------------------------------------

        final_score = (

            # Main fuzzy similarity
            0.30 * name_ratio

            +

            0.16 * address_ratio

            +

            # Token similarity
            12.0 * name_overlap

            +

            7.0 * candidate_name_coverage

            +

            8.0 * address_overlap

            +

            4.0 * address_coverage

            +

            # Structural signals
            exact_name_bonus

            +

            sorted_name_bonus

            +

            first_word_bonus

            +

            last_word_bonus

            +

            prefix_bonus

            +

            suffix_bonus

            +

            country_bonus
        )


        results.append(
            (
                idx,
                final_score
            )
        )


    results.sort(
        key=lambda x: x[1],
        reverse=True
    )

    return results


# ============================================================
# PROCESS SOURCE 1
# ============================================================

print()
print("=" * 75)
print("GENERATING V8 FAST CANDIDATES")
print("=" * 75)

process_start = time.time()

all_rows = []


for i, (_, row) in enumerate(
    s1.iterrows()
):

    # --------------------------------------------------------
    # COLLECT HIGH-RECALL CANDIDATES
    # --------------------------------------------------------

    candidate_indices = (
        collect_candidates(row)
    )


    # --------------------------------------------------------
    # RANK ONLY BLOCKED CANDIDATES
    # --------------------------------------------------------

    ranked = rank_candidates(
        row,
        candidate_indices
    )


    # --------------------------------------------------------
    # TOP 100
    # --------------------------------------------------------

    ranked = ranked[
        :TOP_CANDIDATES
    ]


    # --------------------------------------------------------
    # ENTITY IDS
    # --------------------------------------------------------

    entity_ids = []

    for idx, score in ranked:

        entity_id = candidates.iloc[
            idx
        ]["entity_id"]

        entity_ids.append(
            str(entity_id)
        )


    # --------------------------------------------------------
    # SAVE ROW
    # --------------------------------------------------------

    all_rows.append({

        "source1_entity_id":
            str(row["entity_id"]),

        "candidate_entity_ids":
            ",".join(entity_ids)
    })


    # --------------------------------------------------------
    # PROGRESS
    # --------------------------------------------------------

    if (
        (i + 1) % 100
        == 0
    ):

        elapsed = (
            time.time()
            -
            process_start
        )

        rate = (
            (i + 1)
            /
            max(
                elapsed,
                0.001
            )
        )

        remaining = (
            len(s1)
            -
            (i + 1)
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
            f"{i + 1:,}/"
            f"{len(s1):,} "
            f"| "
            f"{rate:.1f} rows/sec "
            f"| ETA "
            f"{eta:.1f}s"
        )


# ============================================================
# SAVE
# ============================================================

results = pd.DataFrame(
    all_rows
)

results.to_csv(
    OUTPUT_FILE,
    sep="\t",
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

candidate_counts = (
    results[
        "candidate_entity_ids"
    ]
    .apply(
        lambda x:
        len(x.split(","))
        if x
        else 0
    )
)


print()
print("=" * 75)
print("V8 FAST RESULTS")
print("=" * 75)

print(
    "Source 1 rows:",
    len(s1)
)

print(
    "Output rows:",
    len(results)
)

print(
    "Average candidates:",
    f"{candidate_counts.mean():.2f}"
)

print(
    "Maximum candidates:",
    candidate_counts.max()
)

print(
    "Rows with candidates:",
    (
        candidate_counts > 0
    ).sum()
)

print(
    "Total runtime:",
    f"{time.time() - start_total:.1f}s"
)

print()
print(
    "Saved:",
    OUTPUT_FILE
)

print()
print("=" * 75)
print("DONE!")
print("=" * 75)