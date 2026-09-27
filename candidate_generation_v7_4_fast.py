import pandas as pd
import re
import unicodedata
import time
from collections import defaultdict, Counter


# ============================================================
# CONFIGURATION
# ============================================================

# KEEP THIS AT 1000 FOR TESTING
# Change to None ONLY after recall is confirmed.
TEST_ROWS = 1000

TOP_CANDIDATES = 100

# Ignore extremely common tokens.
MAX_TOKEN_FREQUENCY = 5000

# Prevent one blocking key from flooding candidates.
MAX_FROM_BLOCK = 300

OUTPUT_FILE = "v7_4_fast_candidates.tsv"

# Targeted recovery indexes
ENABLE_TARGETED_RECOVERY = True
RECOVERY_MAX_BLOCK = 300


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

    return text.split()


def get_first_word(text):

    tokens = tokenize(text)

    if tokens:
        return tokens[0]

    return ""


def get_address_number(text):

    if not text:
        return ""

    match = re.search(
        r"\b\d{1,6}\b",
        text
    )

    if match:
        return match.group(0)

    return ""


def sorted_token_key(text):

    tokens = tokenize(text)

    if not tokens:
        return ""

    return " ".join(
        sorted(set(tokens))
    )


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("V7.4 FAST HIGH-RECALL CANDIDATE GENERATION")
print("=" * 70)

total_start = time.time()

print("\nLoading datasets...")

s1 = pd.read_csv(
    "dataset/train/train_source1.tsv",
    sep="\t"
)

s2 = pd.read_csv(
    "dataset/train/train_source2.tsv",
    sep="\t"
)

s3 = pd.read_csv(
    "dataset/train/train_source3.tsv",
    sep="\t"
)

print("Source 1:", len(s1))
print("Source 2:", len(s2))
print("Source 3:", len(s3))


# ============================================================
# TEST MODE
# ============================================================

if TEST_ROWS is not None:

    s1 = s1.head(TEST_ROWS).copy()

    print()
    print("=" * 70)
    print(
        f"TEST MODE: {TEST_ROWS} Source-1 rows"
    )
    print("=" * 70)

    print(
        "Change TEST_ROWS = None "
        "only after testing."
    )


# ============================================================
# NORMALIZATION
# ============================================================

print("\nNormalizing...")

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

    df["country_norm"] = (
        df["country"]
        .fillna("")
        .astype(str)
        .str.lower()
        .str.strip()
    )

    df["name_tokens"] = (
        df["name_norm"]
        .apply(tokenize)
    )

    df["address_tokens"] = (
        df["address_norm"]
        .apply(tokenize)
    )

    df["name_sorted"] = (
        df["name_norm"]
        .apply(sorted_token_key)
    )

    df["first_word"] = (
        df["name_norm"]
        .apply(get_first_word)
    )

    df["address_number"] = (
        df["address_norm"]
        .apply(get_address_number)
    )

    # Recovery features

    df["name_last_word"] = df["name_norm"].apply(
        lambda x: x.split()[-1] if x else ""
    )

    df["name_prefix4"] = (
        df["name_norm"]
        .str[:4]
    )

    df["name_suffix3"] = df["name_norm"].apply(
        lambda x: x[-3:] if len(x) >= 3 else x
    )

    df["address_last_word"] = df["address_norm"].apply(
        lambda x: x.split()[-1] if x else ""
    )


# ============================================================
# COMBINE SOURCE 2 + SOURCE 3
# ============================================================

print("\nPreparing candidate data...")

s2["candidate_source"] = "source2"
s3["candidate_source"] = "source3"

candidates = pd.concat(
    [s2, s3],
    ignore_index=True
)

print(
    "Total candidate records:",
    len(candidates)
)


# ============================================================
# CONVERT TO ARRAYS
# ============================================================

candidate_ids = (
    candidates["entity_id"]
    .astype(str)
    .to_numpy()
)

candidate_names = (
    candidates["name_norm"]
    .fillna("")
    .to_numpy()
)

candidate_addresses = (
    candidates["address_norm"]
    .fillna("")
    .to_numpy()
)

candidate_countries = (
    candidates["country_norm"]
    .fillna("")
    .to_numpy()
)

candidate_name_tokens = (
    candidates["name_tokens"]
    .to_numpy()
)

candidate_address_tokens = (
    candidates["address_tokens"]
    .to_numpy()
)

candidate_name_sorted = (
    candidates["name_sorted"]
    .fillna("")
    .to_numpy()
)

candidate_first_words = (
    candidates["first_word"]
    .fillna("")
    .to_numpy()
)

candidate_address_numbers = (
    candidates["address_number"]
    .fillna("")
    .to_numpy()
)

candidate_name_prefix4 = (
    candidates["name_prefix4"]
    .fillna("")
    .to_numpy()
)

candidate_name_last_words = (
    candidates["name_last_word"]
    .fillna("")
    .to_numpy()
)

candidate_name_suffix3 = (
    candidates["name_suffix3"]
    .fillna("")
    .to_numpy()
)

candidate_address_last_words = (
    candidates["address_last_word"]
    .fillna("")
    .to_numpy()
)


# ============================================================
# EXACT NAME INDEX
# ============================================================

print("\nBuilding exact-name index...")

exact_name_index = defaultdict(list)

for idx, value in enumerate(candidate_names):

    if value:
        exact_name_index[value].append(idx)


print(
    "Exact name keys:",
    len(exact_name_index)
)


# ============================================================
# SORTED NAME INDEX
# ============================================================

print("Building sorted-name index...")

sorted_name_index = defaultdict(list)

for idx, value in enumerate(candidate_name_sorted):

    if value:
        sorted_name_index[value].append(idx)


print(
    "Sorted-name keys:",
    len(sorted_name_index)
)


# ============================================================
# FIRST WORD INDEX
# ============================================================

print("Building first-word index...")

first_word_index = defaultdict(list)

for idx, value in enumerate(candidate_first_words):

    if value:
        first_word_index[value].append(idx)


print(
    "First-word keys:",
    len(first_word_index)
)


# ============================================================
# ADDRESS NUMBER INDEX
# ============================================================

print("Building address-number index...")

address_number_index = defaultdict(list)

for idx, value in enumerate(candidate_address_numbers):

    if value:
        address_number_index[value].append(idx)


print(
    "Address-number keys:",
    len(address_number_index)
)


# ============================================================
# TARGETED RECOVERY INDEXES
# ============================================================

print("\nBuilding targeted recovery indexes...")

name_prefix4_index = defaultdict(list)
name_last_word_index = defaultdict(list)
name_suffix3_index = defaultdict(list)
address_last_word_index = defaultdict(list)
country_suffix3_index = defaultdict(list)
country_address_number_index = defaultdict(list)


for idx in range(len(candidate_ids)):

    name_prefix4 = candidate_name_prefix4[idx]

    if name_prefix4:
        name_prefix4_index[name_prefix4].append(idx)


    name_last_word = candidate_name_last_words[idx]

    if name_last_word:
        name_last_word_index[name_last_word].append(idx)


    name_suffix3 = candidate_name_suffix3[idx]

    if name_suffix3:
        name_suffix3_index[name_suffix3].append(idx)


    address_last_word = candidate_address_last_words[idx]

    if address_last_word:
        address_last_word_index[
            address_last_word
        ].append(idx)


    country = candidate_countries[idx]

    if country and name_suffix3:

        country_suffix3_index[
            country + "|" + name_suffix3
        ].append(idx)


    address_number = candidate_address_numbers[idx]

    if country and address_number:

        country_address_number_index[
            country + "|" + address_number
        ].append(idx)


print(
    "  Name prefix4 keys:",
    len(name_prefix4_index)
)

print(
    "  Name last-word keys:",
    len(name_last_word_index)
)

print(
    "  Name suffix3 keys:",
    len(name_suffix3_index)
)

print(
    "  Address last-word keys:",
    len(address_last_word_index)
)

print(
    "  Country+suffix3 keys:",
    len(country_suffix3_index)
)

print(
    "  Country+address-number keys:",
    len(country_address_number_index)
)


# ============================================================
# TOKEN FREQUENCIES
# ============================================================

print("\nCalculating token frequencies...")

name_frequency = Counter()
address_frequency = Counter()


for tokens in candidate_name_tokens:

    for token in set(tokens):

        if len(token) >= 2:
            name_frequency[token] += 1


for tokens in candidate_address_tokens:

    for token in set(tokens):

        if len(token) >= 2:
            address_frequency[token] += 1


print(
    "Unique name tokens:",
    len(name_frequency)
)

print(
    "Unique address tokens:",
    len(address_frequency)
)


# ============================================================
# RARE NAME TOKEN INDEX
# ============================================================

print("\nBuilding rare name-token index...")

name_token_index = defaultdict(list)


for idx, tokens in enumerate(candidate_name_tokens):

    for token in set(tokens):

        if len(token) < 2:
            continue

        frequency = name_frequency[token]

        if frequency <= MAX_TOKEN_FREQUENCY:

            name_token_index[token].append(idx)


print(
    "Usable name tokens:",
    len(name_token_index)
)


# ============================================================
# RARE ADDRESS TOKEN INDEX
# ============================================================

print("Building rare address-token index...")

address_token_index = defaultdict(list)


for idx, tokens in enumerate(candidate_address_tokens):

    for token in set(tokens):

        if len(token) < 2:
            continue

        frequency = address_frequency[token]

        if frequency <= MAX_TOKEN_FREQUENCY:

            address_token_index[token].append(idx)


print(
    "Usable address tokens:",
    len(address_token_index)
)


# ============================================================
# COUNTRY + FIRST WORD INDEX
# ============================================================

print("Building country + first-word index...")

country_first_word_index = defaultdict(list)


for idx in range(len(candidate_ids)):

    country = candidate_countries[idx]
    first_word = candidate_first_words[idx]

    if country and first_word:

        country_first_word_index[
            country + "|" + first_word
        ].append(idx)


print(
    "Country+first-word keys:",
    len(country_first_word_index)
)


# ============================================================
# COUNTRY + PREFIX INDEX
# ============================================================
#
# IMPORTANT:
# We intentionally REMOVED the old country+prefix index.
#
# That index was the main bottleneck when processing
# ~10.3 million candidate records.
#
# We keep:
#
#   country + first word
#   country + suffix3
#   country + address number
#
# as targeted recovery blocks.
# ============================================================

print(
    "Skipping country + prefix index "
    "(V7.4 speed fix)..."
)


# ============================================================
# BLOCK HELPER
# ============================================================

def add_scored_block(
    scores,
    values,
    block_score,
    max_items=MAX_FROM_BLOCK
):

    if not values:
        return

    if len(values) > max_items:

        values = values[:max_items]

    for idx in values:

        scores[idx] = (
            scores.get(idx, 0)
            + block_score
        )


# ============================================================
# CANDIDATE COLLECTION
# ============================================================

def collect_candidates(i):

    """
    Collect candidates using cheap blocking.

    IMPORTANT:
    No RapidFuzz is used here.

    The LightGBM ranker will later calculate
    expensive fuzzy matching features.

    This keeps candidate generation much faster.
    """

    scores = {}

    source_name = s1_names[i]

    source_country = s1_countries[i]

    source_tokens = s1_name_tokens[i]

    source_address_tokens = (
        s1_address_tokens[i]
    )

    source_first_word = (
        s1_first_words[i]
    )

    source_address_number = (
        s1_address_numbers[i]
    )

    source_sorted_name = (
        s1_name_sorted[i]
    )

    source_name_prefix4 = (
        s1_name_prefix4[i]
    )

    source_name_last_word = (
        s1_name_last_words[i]
    )

    source_name_suffix3 = (
        s1_name_suffix3[i]
    )

    source_address_last_word = (
        s1_address_last_words[i]
    )


    # ========================================================
    # 1. EXACT NORMALIZED NAME
    # ========================================================

    if source_name:

        add_scored_block(
            scores,
            exact_name_index.get(
                source_name,
                ()
            ),
            100
        )


    # ========================================================
    # 2. SAME TOKENS, DIFFERENT ORDER
    # ========================================================

    if source_sorted_name:

        add_scored_block(
            scores,
            sorted_name_index.get(
                source_sorted_name,
                ()
            ),
            80
        )


    # ========================================================
    # 3. COUNTRY + FIRST WORD
    # ========================================================

    if (
        source_country
        and source_first_word
    ):

        key = (
            source_country
            + "|"
            + source_first_word
        )

        add_scored_block(
            scores,
            country_first_word_index.get(
                key,
                ()
            ),
            70
        )


    # ========================================================
    # 4. COUNTRY + ADDRESS NUMBER
    # ========================================================

    if (
        source_country
        and source_address_number
    ):

        key = (
            source_country
            + "|"
            + source_address_number
        )

        add_scored_block(
            scores,
            country_address_number_index.get(
                key,
                ()
            ),
            65,
            RECOVERY_MAX_BLOCK
        )


    # ========================================================
    # 5. RARE NAME TOKENS
    # ========================================================

    useful_name_tokens = []

    for token in set(source_tokens):

        if len(token) < 2:
            continue

        freq = name_frequency.get(
            token,
            0
        )

        if (
            0 < freq
            <= MAX_TOKEN_FREQUENCY
        ):

            useful_name_tokens.append(
                (freq, token)
            )


    # Rare tokens first
    useful_name_tokens.sort(
        key=lambda x: x[0]
    )


    for _, token in useful_name_tokens[:5]:

        add_scored_block(
            scores,
            name_token_index.get(
                token,
                ()
            ),
            40
        )


    # ========================================================
    # 6. ADDRESS NUMBER WITHOUT COUNTRY
    # ========================================================

    if source_address_number:

        add_scored_block(
            scores,
            address_number_index.get(
                source_address_number,
                ()
            ),
            50
        )


    # ========================================================
    # 7. RARE ADDRESS TOKENS
    # ========================================================

    useful_address_tokens = []

    for token in set(
        source_address_tokens
    ):

        if len(token) < 2:
            continue

        freq = address_frequency.get(
            token,
            0
        )

        if (
            0 < freq
            <= MAX_TOKEN_FREQUENCY
        ):

            useful_address_tokens.append(
                (freq, token)
            )


    useful_address_tokens.sort(
        key=lambda x: x[0]
    )


    for _, token in useful_address_tokens[:5]:

        add_scored_block(
            scores,
            address_token_index.get(
                token,
                ()
            ),
            30
        )


    # ========================================================
    # 8. TARGETED RECOVERY
    # ========================================================

    if ENABLE_TARGETED_RECOVERY:


        # Name prefix 4
        if source_name_prefix4:

            add_scored_block(
                scores,
                name_prefix4_index.get(
                    source_name_prefix4,
                    ()
                ),
                25,
                RECOVERY_MAX_BLOCK
            )


        # Name last word
        if source_name_last_word:

            add_scored_block(
                scores,
                name_last_word_index.get(
                    source_name_last_word,
                    ()
                ),
                20,
                RECOVERY_MAX_BLOCK
            )


        # Name suffix 3
        if source_name_suffix3:

            add_scored_block(
                scores,
                name_suffix3_index.get(
                    source_name_suffix3,
                    ()
                ),
                20,
                RECOVERY_MAX_BLOCK
            )


            # Country + suffix 3
            if source_country:

                key = (
                    source_country
                    + "|"
                    + source_name_suffix3
                )

                add_scored_block(
                    scores,
                    country_suffix3_index.get(
                        key,
                        ()
                    ),
                    35,
                    RECOVERY_MAX_BLOCK
                )


        # Address last word
        if source_address_last_word:

            add_scored_block(
                scores,
                address_last_word_index.get(
                    source_address_last_word,
                    ()
                ),
                15,
                RECOVERY_MAX_BLOCK
            )


    # ========================================================
    # COUNTRY BONUS
    # ========================================================

    if source_country:

        for idx in scores:

            if (
                candidate_countries[idx]
                == source_country
            ):

                scores[idx] += 8


    # ========================================================
    # FINAL CHEAP BLOCK SCORE
    # ========================================================

    ranked = sorted(
        scores.items(),
        key=lambda x: (
            -x[1],
            x[0]
        )
    )


    return [
        idx
        for idx, _ in ranked[
            :TOP_CANDIDATES
        ]
    ]


# ============================================================
# SOURCE 1 ARRAYS
# ============================================================

s1_ids = (
    s1["entity_id"]
    .astype(str)
    .to_numpy()
)

s1_names = (
    s1["name_norm"]
    .fillna("")
    .to_numpy()
)

s1_addresses = (
    s1["address_norm"]
    .fillna("")
    .to_numpy()
)

s1_countries = (
    s1["country_norm"]
    .fillna("")
    .to_numpy()
)

s1_name_tokens = (
    s1["name_tokens"]
    .to_numpy()
)

s1_address_tokens = (
    s1["address_tokens"]
    .to_numpy()
)

s1_name_sorted = (
    s1["name_sorted"]
    .fillna("")
    .to_numpy()
)

s1_first_words = (
    s1["first_word"]
    .fillna("")
    .to_numpy()
)

s1_address_numbers = (
    s1["address_number"]
    .fillna("")
    .to_numpy()
)

s1_name_prefix4 = (
    s1["name_prefix4"]
    .fillna("")
    .to_numpy()
)

s1_name_last_words = (
    s1["name_last_word"]
    .fillna("")
    .to_numpy()
)

s1_name_suffix3 = (
    s1["name_suffix3"]
    .fillna("")
    .to_numpy()
)

s1_address_last_words = (
    s1["address_last_word"]
    .fillna("")
    .to_numpy()
)


# ============================================================
# GENERATE CANDIDATES
# ============================================================

print()
print("=" * 70)
print("GENERATING V7.4 FAST CANDIDATES")
print("=" * 70)

all_rows = []

process_start = time.time()

total = len(s1_ids)


for i in range(total):

    candidate_indices = (
        collect_candidates(i)
    )


    candidate_entity_ids = [
        candidate_ids[idx]
        for idx in candidate_indices
    ]


    all_rows.append({

        "source1_entity_id":
            s1_ids[i],

        "candidate_entity_ids":
            ",".join(
                candidate_entity_ids
            )
    })


    # ========================================================
    # PROGRESS
    # ========================================================

    if (
        (i + 1) % 100 == 0
        or i + 1 == total
    ):

        elapsed = (
            time.time()
            - process_start
        )

        rate = (
            (i + 1)
            / max(elapsed, 1e-9)
        )

        remaining = (
            (total - i - 1)
            / max(rate, 1e-9)
        )

        print(
            f"Processed "
            f"{i + 1:,}/{total:,}"
            f" | {rate:.1f} rows/sec"
            f" | ETA {remaining:.1f}s"
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

print()
print("=" * 70)
print("V7.4 FAST RESULTS")
print("=" * 70)

print(
    "Source 1 rows:",
    len(s1)
)

print(
    "Output rows:",
    len(results)
)


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


print(
    "Average candidates:",
    round(
        candidate_counts.mean(),
        2
    )
)

print(
    "Maximum candidates:",
    candidate_counts.max()
)

print(
    "Rows with candidates:",
    int(
        (
            candidate_counts > 0
        ).sum()
    )
)

print(
    "Empty candidate rows:",
    int(
        (
            candidate_counts == 0
        ).sum()
    )
)

print(
    "Processing time:",
    round(
        time.time()
        - process_start,
        1
    ),
    "s"
)

print(
    "Total runtime:",
    round(
        time.time()
        - total_start,
        1
    ),
    "s"
)

print()

print(
    "Saved:",
    OUTPUT_FILE
)

print()
print("=" * 70)
print("DONE!")
print("=" * 70)