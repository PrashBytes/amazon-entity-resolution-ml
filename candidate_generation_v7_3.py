import pandas as pd
import re
import unicodedata
from collections import defaultdict, Counter
from rapidfuzz.fuzz import ratio, token_set_ratio, WRatio


# ============================================================
# CONFIGURATION
# ============================================================

# IMPORTANT:
# Keep this at 1000 while testing.
# Change to None ONLY after we confirm the recall is better.
TEST_ROWS = 1000

# Keep more candidates than V6 so we can measure recall properly.
TOP_CANDIDATES = 100

# Ignore extremely common tokens because they create huge
# candidate lists and don't help much.
MAX_TOKEN_FREQUENCY = 5000

# Prevent one blocking key from flooding the candidate set.
MAX_FROM_BLOCK = 300

OUTPUT_FILE = "v7_3_candidates.tsv"

# V7.3: targeted recovery indexes. These are intentionally small/cheap.
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
print("V7.3 TARGETED RECOVERY + MULTI-SIGNAL RANKING")
print("=" * 70)

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

    # V7.3 recovery features
    df["name_last_word"] = df["name_norm"].apply(
        lambda x: x.split()[-1] if x else ""
    )
    df["name_prefix4"] = df["name_norm"].str[:4]
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
# EXACT NAME INDEX
# ============================================================

print("\nBuilding exact-name index...")

exact_name_index = defaultdict(list)

for idx, value in enumerate(
    candidates["name_norm"]
):

    if value:
        exact_name_index[value].append(idx)


print(
    "Exact name keys:",
    len(exact_name_index)
)


# ============================================================
# SORTED TOKEN NAME INDEX
# ============================================================

print("Building sorted-name index...")

sorted_name_index = defaultdict(list)

for idx, value in enumerate(
    candidates["name_sorted"]
):

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

for idx, value in enumerate(
    candidates["first_word"]
):

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

for idx, value in enumerate(
    candidates["address_number"]
):

    if value:
        address_number_index[value].append(idx)


print(
    "Address-number keys:",
    len(address_number_index)
)


# ============================================================
# V7.3 TARGETED RECOVERY INDEXES
# ============================================================

print("\nBuilding targeted recovery indexes...")

name_prefix4_index = defaultdict(list)
name_last_word_index = defaultdict(list)
name_suffix3_index = defaultdict(list)
address_last_word_index = defaultdict(list)
country_suffix3_index = defaultdict(list)
country_address_number_index = defaultdict(list)

for idx, row in candidates.iterrows():

    name_prefix4 = row["name_prefix4"]
    if name_prefix4:
        name_prefix4_index[name_prefix4].append(idx)

    name_last_word = row["name_last_word"]
    if name_last_word:
        name_last_word_index[name_last_word].append(idx)

    name_suffix3 = row["name_suffix3"]
    if name_suffix3:
        name_suffix3_index[name_suffix3].append(idx)

    address_last_word = row["address_last_word"]
    if address_last_word:
        address_last_word_index[address_last_word].append(idx)

    country = row["country_norm"]
    if country and name_suffix3:
        country_suffix3_index[country + "|" + name_suffix3].append(idx)

    address_number = row["address_number"]
    if country and address_number:
        country_address_number_index[
            country + "|" + address_number
        ].append(idx)

print("  Name prefix4 keys:", len(name_prefix4_index))
print("  Name last-word keys:", len(name_last_word_index))
print("  Name suffix3 keys:", len(name_suffix3_index))
print("  Address last-word keys:", len(address_last_word_index))
print("  Country+suffix3 keys:", len(country_suffix3_index))
print("  Country+address-number keys:", len(country_address_number_index))


# ============================================================
# TOKEN FREQUENCIES
# ============================================================

print("\nCalculating token frequencies...")

name_frequency = Counter()
address_frequency = Counter()


for tokens in candidates["name_tokens"]:

    for token in set(tokens):

        if len(token) >= 2:
            name_frequency[token] += 1


for tokens in candidates["address_tokens"]:

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
# NAME TOKEN INDEX
# ============================================================

print("\nBuilding rare name-token index...")

name_token_index = defaultdict(list)

for idx, tokens in enumerate(
    candidates["name_tokens"]
):

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
# ADDRESS TOKEN INDEX
# ============================================================

print("Building rare address-token index...")

address_token_index = defaultdict(list)

for idx, tokens in enumerate(
    candidates["address_tokens"]
):

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

for idx, row in candidates.iterrows():

    country = row["country_norm"]
    first_word = row["first_word"]

    if country and first_word:

        key = (
            country
            + "|"
            + first_word
        )

        country_first_word_index[key].append(
            idx
        )


print(
    "Country+first-word keys:",
    len(country_first_word_index)
)


# ============================================================
# COUNTRY + PREFIX INDEX
# ============================================================

print("Building country + prefix index...")

country_prefix_index = defaultdict(list)

for idx, row in candidates.iterrows():

    country = row["country_norm"]
    name = row["name_norm"]

    if not country or not name:
        continue

    prefixes = set()

    if len(name) >= 3:
        prefixes.add(
            name[:3]
        )

    if len(name) >= 5:
        prefixes.add(
            name[:5]
        )

    for prefix in prefixes:

        key = (
            country
            + "|"
            + prefix
        )

        country_prefix_index[key].append(
            idx
        )


print(
    "Country+prefix keys:",
    len(country_prefix_index)
)


# ============================================================
# BLOCK HELPER
# ============================================================

def add_block(
    target_set,
    values,
    max_items=MAX_FROM_BLOCK
):

    if not values:
        return

    if len(values) <= max_items:

        target_set.update(values)

    else:

        target_set.update(
            values[:max_items]
        )


# ============================================================
# CANDIDATE COLLECTION
# ============================================================

def collect_candidates(row):

    candidate_indices = set()

    source_name = row["name_norm"]
    source_country = row["country_norm"]
    source_tokens = row["name_tokens"]
    source_address_tokens = row["address_tokens"]
    source_first_word = row["first_word"]
    source_address_number = row["address_number"]
    source_sorted_name = row["name_sorted"]
    source_name_prefix4 = row["name_prefix4"]
    source_name_last_word = row["name_last_word"]
    source_name_suffix3 = row["name_suffix3"]
    source_address_last_word = row["address_last_word"]


    # --------------------------------------------------------
    # 1. EXACT NORMALIZED NAME
    # --------------------------------------------------------

    if source_name:

        add_block(
            candidate_indices,
            exact_name_index.get(
                source_name,
                []
            )
        )


    # --------------------------------------------------------
    # 2. SORTED NAME TOKENS
    # --------------------------------------------------------

    if source_sorted_name:

        add_block(
            candidate_indices,
            sorted_name_index.get(
                source_sorted_name,
                []
            )
        )


    # --------------------------------------------------------
    # 3. COUNTRY + FIRST WORD
    # --------------------------------------------------------

    if (
        source_country
        and source_first_word
    ):

        key = (
            source_country
            + "|"
            + source_first_word
        )

        add_block(
            candidate_indices,
            country_first_word_index.get(
                key,
                []
            )
        )


    # --------------------------------------------------------
    # 4. COUNTRY + NAME PREFIX
    # --------------------------------------------------------

    if source_country and source_name:

        prefixes = set()

        if len(source_name) >= 3:
            prefixes.add(
                source_name[:3]
            )

        if len(source_name) >= 5:
            prefixes.add(
                source_name[:5]
            )

        for prefix in prefixes:

            key = (
                source_country
                + "|"
                + prefix
            )

            add_block(
                candidate_indices,
                country_prefix_index.get(
                    key,
                    []
                )
            )


    # --------------------------------------------------------
    # 5. RARE NAME TOKENS
    # --------------------------------------------------------

    useful_name_tokens = []

    for token in set(source_tokens):

        if len(token) < 2:
            continue

        frequency = name_frequency.get(
            token,
            0
        )

        if (
            frequency > 0
            and frequency <= MAX_TOKEN_FREQUENCY
        ):

            useful_name_tokens.append(
                (
                    frequency,
                    token
                )
            )


    # Rare tokens first.
    useful_name_tokens.sort(
        key=lambda x: x[0]
    )


    for _, token in useful_name_tokens[:5]:

        add_block(
            candidate_indices,
            name_token_index.get(
                token,
                []
            )
        )


    # --------------------------------------------------------
    # 6. ADDRESS NUMBER
    # --------------------------------------------------------

    if source_address_number:

        add_block(
            candidate_indices,
            address_number_index.get(
                source_address_number,
                []
            )
        )


    # --------------------------------------------------------
    # 7. RARE ADDRESS TOKENS
    # --------------------------------------------------------

    useful_address_tokens = []

    for token in set(
        source_address_tokens
    ):

        if len(token) < 2:
            continue

        frequency = address_frequency.get(
            token,
            0
        )

        if (
            frequency > 0
            and frequency <= MAX_TOKEN_FREQUENCY
        ):

            useful_address_tokens.append(
                (
                    frequency,
                    token
                )
            )


    useful_address_tokens.sort(
        key=lambda x: x[0]
    )


    for _, token in useful_address_tokens[:5]:

        add_block(
            candidate_indices,
            address_token_index.get(
                token,
                []
            )
        )


    # --------------------------------------------------------
    # 8. V7.3 TARGETED RECOVERY BLOCKS
    # --------------------------------------------------------
    # These are deliberately used in addition to V7, not instead of it.
    # They target cases where punctuation/order/one token differs.
    if ENABLE_TARGETED_RECOVERY:

        if source_name_prefix4:
            add_block(
                candidate_indices,
                name_prefix4_index.get(source_name_prefix4, []),
                RECOVERY_MAX_BLOCK
            )

        if source_name_last_word:
            add_block(
                candidate_indices,
                name_last_word_index.get(source_name_last_word, []),
                RECOVERY_MAX_BLOCK
            )

        if source_name_suffix3:
            add_block(
                candidate_indices,
                name_suffix3_index.get(source_name_suffix3, []),
                RECOVERY_MAX_BLOCK
            )

        if source_address_last_word:
            add_block(
                candidate_indices,
                address_last_word_index.get(source_address_last_word, []),
                RECOVERY_MAX_BLOCK
            )

        if source_country and source_name_suffix3:
            key = source_country + "|" + source_name_suffix3
            add_block(
                candidate_indices,
                country_suffix3_index.get(key, []),
                RECOVERY_MAX_BLOCK
            )

        if source_country and source_address_number:
            key = source_country + "|" + source_address_number
            add_block(
                candidate_indices,
                country_address_number_index.get(key, []),
                RECOVERY_MAX_BLOCK
            )

    return candidate_indices


# ============================================================
# RANK CANDIDATES
# ============================================================

def rank_candidates(
    row,
    candidate_indices
):

    results = []

    source_name = row["name_norm"]
    source_address = row["address_norm"]
    source_country = row["country_norm"]
    source_first_word = row["first_word"]
    source_address_number = row["address_number"]
    source_name_prefix4 = row["name_prefix4"]
    source_name_suffix3 = row["name_suffix3"]

    source_name_tokens = set(
        row["name_tokens"]
    )

    source_address_tokens = set(
        row["address_tokens"]
    )


    for idx in candidate_indices:

        candidate = candidates.iloc[idx]

        candidate_name = candidate["name_norm"]
        candidate_address = candidate["address_norm"]
        candidate_country = candidate["country_norm"]
        candidate_first_word = candidate["first_word"]
        candidate_address_number = candidate["address_number"]
        candidate_name_prefix4 = candidate["name_prefix4"]
        candidate_name_suffix3 = candidate["name_suffix3"]


        # ----------------------------------------------------
        # NAME SCORES
        # ----------------------------------------------------

        name_ratio = ratio(
            source_name,
            candidate_name
        )

        name_token_score = token_set_ratio(
            source_name,
            candidate_name
        )

        name_wratio = WRatio(
            source_name,
            candidate_name
        )


        # ----------------------------------------------------
        # ADDRESS SCORES
        # ----------------------------------------------------

        address_ratio = ratio(
            source_address,
            candidate_address
        )

        address_token_score = token_set_ratio(
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

            common_name_tokens = (
                source_name_tokens
                &
                candidate_name_tokens
            )

            name_overlap = (
                len(common_name_tokens)
                /
                max(
                    1,
                    len(source_name_tokens)
                )
            )

        else:

            name_overlap = 0


        # ----------------------------------------------------
        # ADDRESS TOKEN OVERLAP
        # ----------------------------------------------------

        candidate_address_tokens = set(
            candidate["address_tokens"]
        )

        if source_address_tokens:

            common_address_tokens = (
                source_address_tokens
                &
                candidate_address_tokens
            )

            address_overlap = (
                len(common_address_tokens)
                /
                max(
                    1,
                    len(source_address_tokens)
                )
            )

        else:

            address_overlap = 0


        # ----------------------------------------------------
        # V7.3 STRUCTURAL MATCH FEATURES
        # ----------------------------------------------------

        first_word_bonus = (
            2.5
            if source_first_word
            and candidate_first_word
            and source_first_word == candidate_first_word
            else 0
        )

        address_number_bonus = (
            4.0
            if source_address_number
            and candidate_address_number
            and source_address_number == candidate_address_number
            else 0
        )

        prefix4_bonus = (
            1.5
            if source_name_prefix4
            and candidate_name_prefix4
            and source_name_prefix4 == candidate_name_prefix4
            else 0
        )

        suffix3_bonus = (
            1.5
            if source_name_suffix3
            and candidate_name_suffix3
            and source_name_suffix3 == candidate_name_suffix3
            else 0
        )

        # Jaccard rewards candidates containing the same distinctive
        # name tokens without letting long names dominate.
        if source_name_tokens or candidate_name_tokens:
            name_jaccard = (
                len(source_name_tokens & candidate_name_tokens)
                / max(1, len(source_name_tokens | candidate_name_tokens))
            )
        else:
            name_jaccard = 0

        # ----------------------------------------------------
        # COUNTRY BONUS
        # ----------------------------------------------------

        country_bonus = 0

        if (
            source_country
            and candidate_country
            and source_country
            == candidate_country
        ):

            country_bonus = 5


        # ----------------------------------------------------
        # FINAL RANKING SCORE
        # ----------------------------------------------------

        final_score = (

            0.30 * name_ratio

            + 0.25 * name_token_score

            + 0.15 * name_wratio

            + 0.15 * address_ratio

            + 0.10 * address_token_score

            + 5.0 * name_overlap

            + 3.0 * address_overlap

            + 3.0 * name_jaccard

            + first_word_bonus

            + address_number_bonus

            + prefix4_bonus

            + suffix3_bonus

            + country_bonus

        )


        results.append(
            (
                idx,
                final_score,
                name_ratio,
                name_token_score,
                address_ratio
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
print("=" * 70)
print("GENERATING V7.3 CANDIDATES")
print("=" * 70)

all_rows = []


for i, (_, row) in enumerate(
    s1.iterrows()
):

    # --------------------------------------------------------
    # COLLECT ALL BLOCKING CANDIDATES
    # --------------------------------------------------------

    candidate_indices = collect_candidates(
        row
    )


    # --------------------------------------------------------
    # RANK
    # --------------------------------------------------------

    ranked = rank_candidates(
        row,
        candidate_indices
    )


    # --------------------------------------------------------
    # KEEP TOP 100
    # --------------------------------------------------------

    ranked = ranked[
        :TOP_CANDIDATES
    ]


    # --------------------------------------------------------
    # CONVERT TO ENTITY IDS
    # --------------------------------------------------------

    candidate_entity_ids = []

    for (
        idx,
        score,
        name_ratio,
        name_token_score,
        address_ratio
    ) in ranked:

        entity_id = candidates.iloc[
            idx
        ]["entity_id"]

        candidate_entity_ids.append(
            str(entity_id)
        )


    # --------------------------------------------------------
    # SAVE ONE ROW PER SOURCE-1 RECORD
    # --------------------------------------------------------

    all_rows.append({

        "source1_entity_id":
            row["entity_id"],

        "candidate_entity_ids":
            ",".join(
                candidate_entity_ids
            )

    })


    # --------------------------------------------------------
    # PROGRESS
    # --------------------------------------------------------

    if (
        (i + 1) % 100
        == 0
    ):

        print(
            f"Processed "
            f"{i + 1}/"
            f"{len(s1)}"
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
print("V7.3 RESULTS")
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

print()
print(
    "Saved:",
    OUTPUT_FILE
)

print()
print("=" * 70)
print("DONE!")
print("=" * 70)