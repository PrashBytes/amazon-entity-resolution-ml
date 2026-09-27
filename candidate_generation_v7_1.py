import pandas as pd
import re
import unicodedata
import time
from collections import defaultdict
from rapidfuzz.fuzz import ratio


# ============================================================
# V7.1 FAST HIGH-RECALL CANDIDATE GENERATION
#
# Based on working V7 approach.
#
# IMPORTANT:
# - NO TF-IDF
# - NO token-frequency calculation
# - NO global fuzzy search
# - NO LightGBM
# - Only cheap dictionary/set blocking
# - Test on 1000 rows first
# ============================================================


print("=" * 75)
print("V7.1 FAST MULTI-BLOCKING CANDIDATE GENERATION")
print("=" * 75)


# ============================================================
# SETTINGS
# ============================================================

TEST_ROWS = 1000

# Change to None ONLY after V7.1 is tested successfully.
# TEST_ROWS = None

MAX_CANDIDATES = 100

# Maximum number of candidates pulled from an individual block.
PER_BLOCK_LIMIT = 40

# Number of final candidates after fuzzy ranking.
FINAL_LIMIT = 100


# ============================================================
# FILES
# ============================================================

SOURCE1 = "dataset/train/train_source1.tsv"
SOURCE2 = "dataset/train/train_source2.tsv"
SOURCE3 = "dataset/train/train_source3.tsv"

OUTPUT_FILE = "v7_1_candidates.csv"


# ============================================================
# NORMALIZATION
# ============================================================

def normalize_text(text):

    if pd.isna(text):
        return ""

    text = unicodedata.normalize("NFKD", str(text))

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


# ============================================================
# NAME HELPERS
# ============================================================

def first_word(text):

    if not text:
        return ""

    return text.split()[0]


def last_word(text):

    if not text:
        return ""

    return text.split()[-1]


def sorted_name(text):

    if not text:
        return ""

    chars = [
        c for c in text
        if c.isalnum()
    ]

    return "".join(sorted(chars))


def name_signature(text):

    if not text:
        return ""

    words = text.split()

    # Keep first character of each word.
    # Example:
    # "pizza hut restaurant"
    # -> "phr"
    return "".join(
        word[0]
        for word in words
        if word
    )


# ============================================================
# ADDRESS HELPERS
# ============================================================

def address_number(text):

    if not text:
        return ""

    match = re.search(
        r"\b\d+[a-z]?\b",
        text
    )

    if match:
        return match.group(0)

    return ""


def address_first_word(text):

    if not text:
        return ""

    return text.split()[0]


def address_last_word(text):

    if not text:
        return ""

    return text.split()[-1]


# ============================================================
# LOAD DATA
# ============================================================

print("\nLoading datasets...")

start = time.time()

s1 = pd.read_csv(
    SOURCE1,
    sep="\t",
    dtype=str
)

s2 = pd.read_csv(
    SOURCE2,
    sep="\t",
    dtype=str
)

s3 = pd.read_csv(
    SOURCE3,
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

    print("\n" + "=" * 75)
    print(
        f"TEST MODE: {TEST_ROWS} Source-1 rows"
    )
    print("=" * 75)

    s1 = s1.head(TEST_ROWS).copy()

else:

    print("\n" + "=" * 75)
    print("FULL MODE")
    print("=" * 75)


# ============================================================
# NORMALIZATION
# ============================================================

print("\nNormalizing...")

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

    df["country_norm"] = (
        df["country"]
        .fillna("")
        .astype(str)
        .str.lower()
        .str.strip()
    )


print(
    f"Normalization time: "
    f"{time.time() - norm_start:.1f}s"
)


# ============================================================
# PREPARE CANDIDATE DATA
# ============================================================

print("\nPreparing candidate data...")

candidate_frames = [
    s2,
    s3
]

candidates = pd.concat(
    candidate_frames,
    ignore_index=True
)

print(
    "Total candidate records:",
    len(candidates)
)


# ============================================================
# BUILD FEATURES
# ============================================================

print("\nBuilding cheap blocking features...")

feature_start = time.time()


candidates["name_first6"] = (
    candidates["name_norm"].str[:6]
)

candidates["name_first5"] = (
    candidates["name_norm"].str[:5]
)

candidates["name_first4"] = (
    candidates["name_norm"].str[:4]
)

candidates["name_last5"] = (
    candidates["name_norm"].str[-5:]
)

candidates["name_first_word"] = (
    candidates["name_norm"]
    .apply(first_word)
)

candidates["name_last_word"] = (
    candidates["name_norm"]
    .apply(last_word)
)

candidates["name_signature"] = (
    candidates["name_norm"]
    .apply(name_signature)
)

candidates["name_sorted"] = (
    candidates["name_norm"]
    .apply(sorted_name)
)

candidates["address_number"] = (
    candidates["address_norm"]
    .apply(address_number)
)

candidates["address_first8"] = (
    candidates["address_norm"].str[:8]
)

candidates["address_first_word"] = (
    candidates["address_norm"]
    .apply(address_first_word)
)

candidates["address_last_word"] = (
    candidates["address_norm"]
    .apply(address_last_word)
)


print(
    f"Feature preparation time: "
    f"{time.time() - feature_start:.1f}s"
)


# ============================================================
# BUILD INDEX
# ============================================================

print("\nBuilding fast blocking indexes...")

index_start = time.time()


def add_to_index(index, key, entity_id):

    if not key:
        return

    index[key].add(entity_id)


def build_index(df, column):

    index = defaultdict(set)

    for entity_id, value in zip(
        df["entity_id"],
        df[column]
    ):

        if value:
            index[value].add(entity_id)

    return index


# ------------------------------------------------------------
# NAME INDEXES
# ------------------------------------------------------------

print("  Exact name...")

exact_name_index = build_index(
    candidates,
    "name_norm"
)

print(
    "    keys:",
    len(exact_name_index)
)


print("  Name first 6...")

name_first6_index = build_index(
    candidates,
    "name_first6"
)

print(
    "    keys:",
    len(name_first6_index)
)


print("  Name first 5...")

name_first5_index = build_index(
    candidates,
    "name_first5"
)

print(
    "    keys:",
    len(name_first5_index)
)


print("  Name first 4...")

name_first4_index = build_index(
    candidates,
    "name_first4"
)

print(
    "    keys:",
    len(name_first4_index)
)


print("  Name first word...")

name_first_word_index = build_index(
    candidates,
    "name_first_word"
)

print(
    "    keys:",
    len(name_first_word_index)
)


print("  Name last word...")

name_last_word_index = build_index(
    candidates,
    "name_last_word"
)

print(
    "    keys:",
    len(name_last_word_index)
)


print("  Name signature...")

name_signature_index = build_index(
    candidates,
    "name_signature"
)

print(
    "    keys:",
    len(name_signature_index)
)


print("  Sorted name...")

name_sorted_index = build_index(
    candidates,
    "name_sorted"
)

print(
    "    keys:",
    len(name_sorted_index)
)


# ------------------------------------------------------------
# ADDRESS INDEXES
# ------------------------------------------------------------

print("  Address number...")

address_number_index = build_index(
    candidates,
    "address_number"
)

print(
    "    keys:",
    len(address_number_index)
)


print("  Address prefix...")

address_first8_index = build_index(
    candidates,
    "address_first8"
)

print(
    "    keys:",
    len(address_first8_index)
)


print("  Address first word...")

address_first_word_index = build_index(
    candidates,
    "address_first_word"
)

print(
    "    keys:",
    len(address_first_word_index)
)


print("  Address last word...")

address_last_word_index = build_index(
    candidates,
    "address_last_word"
)

print(
    "    keys:",
    len(address_last_word_index)
)


# ------------------------------------------------------------
# COMBINED INDEXES
# ------------------------------------------------------------

print("  Country + name prefix...")

country_name_prefix_index = defaultdict(set)

for _, row in candidates.iterrows():

    country = row["country_norm"]
    prefix = row["name_first5"]

    if country and prefix:

        key = (
            country
            + "|"
            + prefix
        )

        country_name_prefix_index[key].add(
            row["entity_id"]
        )


print(
    "    keys:",
    len(country_name_prefix_index)
)


print("  Country + address number...")

country_address_number_index = defaultdict(set)

for _, row in candidates.iterrows():

    country = row["country_norm"]
    number = row["address_number"]

    if country and number:

        key = (
            country
            + "|"
            + number
        )

        country_address_number_index[key].add(
            row["entity_id"]
        )


print(
    "    keys:",
    len(country_address_number_index)
)


print(
    f"\nIndex construction time: "
    f"{time.time() - index_start:.1f}s"
)


# ============================================================
# LOOKUP TABLE
# ============================================================

print("\nBuilding candidate lookup...")

candidate_lookup = candidates.set_index(
    "entity_id"
).to_dict("index")


# ============================================================
# CANDIDATE COLLECTION
# ============================================================

def add_limited(
    destination,
    source_set,
    limit=PER_BLOCK_LIMIT
):

    if not source_set:
        return

    count = 0

    for entity_id in source_set:

        destination.add(entity_id)

        count += 1

        if count >= limit:
            break


def collect_candidates(row):

    collected = set()

    name = row["name_norm"]
    address = row["address_norm"]
    country = row["country_norm"]

    first6 = name[:6]
    first5 = name[:5]
    first4 = name[:4]
    last5 = name[-5:]

    fw = first_word(name)
    lw = last_word(name)

    sig = name_signature(name)
    sorted_n = sorted_name(name)

    addr_num = address_number(address)
    addr_prefix = address[:8]

    addr_fw = address_first_word(address)
    addr_lw = address_last_word(address)


    # ========================================================
    # 1. EXACT NAME
    # ========================================================

    add_limited(
        collected,
        exact_name_index.get(name, set()),
        100
    )


    # ========================================================
    # 2. STRONG NAME PREFIXES
    # ========================================================

    add_limited(
        collected,
        name_first6_index.get(first6, set())
    )

    add_limited(
        collected,
        name_first5_index.get(first5, set())
    )

    add_limited(
        collected,
        name_first4_index.get(first4, set())
    )


    # ========================================================
    # 3. NAME WORD BLOCKS
    # ========================================================

    add_limited(
        collected,
        name_first_word_index.get(fw, set())
    )

    add_limited(
        collected,
        name_last_word_index.get(lw, set())
    )


    # ========================================================
    # 4. NAME SIGNATURE
    # ========================================================

    add_limited(
        collected,
        name_signature_index.get(sig, set())
    )


    # ========================================================
    # 5. SORTED NAME
    # Useful for reordered names.
    # ========================================================

    add_limited(
        collected,
        name_sorted_index.get(sorted_n, set())
    )


    # ========================================================
    # 6. COUNTRY + NAME PREFIX
    # ========================================================

    if country and first5:

        key = (
            country
            + "|"
            + first5
        )

        add_limited(
            collected,
            country_name_prefix_index.get(
                key,
                set()
            )
        )


    # ========================================================
    # 7. ADDRESS NUMBER
    # ========================================================

    add_limited(
        collected,
        address_number_index.get(
            addr_num,
            set()
        )
    )


    # ========================================================
    # 8. ADDRESS PREFIX
    # ========================================================

    add_limited(
        collected,
        address_first8_index.get(
            addr_prefix,
            set()
        )
    )


    # ========================================================
    # 9. ADDRESS WORDS
    # ========================================================

    add_limited(
        collected,
        address_first_word_index.get(
            addr_fw,
            set()
        )
    )

    add_limited(
        collected,
        address_last_word_index.get(
            addr_lw,
            set()
        )
    )


    # ========================================================
    # 10. COUNTRY + ADDRESS NUMBER
    # ========================================================

    if country and addr_num:

        key = (
            country
            + "|"
            + addr_num
        )

        add_limited(
            collected,
            country_address_number_index.get(
                key,
                set()
            )
        )


    return collected


# ============================================================
# FAST RANKING
# ============================================================

def rank_candidates(
    row,
    candidate_ids
):

    source_name = row["name_norm"]
    source_address = row["address_norm"]
    source_country = row["country_norm"]

    ranked = []

    for entity_id in candidate_ids:

        candidate = candidate_lookup.get(
            entity_id
        )

        if candidate is None:
            continue


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
        # NAME SCORE
        # ----------------------------------------------------

        name_score = ratio(
            source_name,
            candidate_name
        )


        # ----------------------------------------------------
        # ADDRESS SCORE
        # ----------------------------------------------------

        address_score = ratio(
            source_address,
            candidate_address
        )


        # ----------------------------------------------------
        # COUNTRY BONUS
        # ----------------------------------------------------

        country_bonus = 0

        if (
            source_country
            and candidate_country
            and source_country == candidate_country
        ):
            country_bonus = 5


        # ----------------------------------------------------
        # EXACT NAME BONUS
        # ----------------------------------------------------

        exact_name_bonus = 0

        if (
            source_name
            and source_name == candidate_name
        ):
            exact_name_bonus = 20


        # ----------------------------------------------------
        # COMBINED SCORE
        # ----------------------------------------------------

        final_score = (
            0.60 * name_score
            +
            0.35 * address_score
            +
            country_bonus
            +
            exact_name_bonus
        )


        ranked.append(
            (
                entity_id,
                final_score
            )
        )


    ranked.sort(
        key=lambda x: x[1],
        reverse=True
    )


    return [
        entity_id
        for entity_id, score
        in ranked[:FINAL_LIMIT]
    ]


# ============================================================
# GENERATION
# ============================================================

print("\n" + "=" * 75)
print("GENERATING V7.1 CANDIDATES")
print("=" * 75)

start_time = time.time()

results = []

total_rows = len(s1)


for i, (_, row) in enumerate(
    s1.iterrows(),
    start=1
):

    candidate_ids = collect_candidates(
        row
    )


    ranked_ids = rank_candidates(
        row,
        candidate_ids
    )


    results.append({
        "source1_entity_id": row[
            "entity_id"
        ],

        "candidate_entity_ids": ",".join(
            ranked_ids
        )
    })


    if i % 100 == 0:

        elapsed = (
            time.time()
            - start_time
        )

        rate = (
            i / elapsed
            if elapsed > 0
            else 0
        )

        remaining = (
            total_rows - i
        )

        eta = (
            remaining / rate
            if rate > 0
            else 0
        )

        print(
            f"Processed {i:,}/{total_rows:,} "
            f"| {rate:.1f} rows/sec "
            f"| ETA {eta:.1f}s"
        )


# ============================================================
# SAVE
# ============================================================

print("\nSaving candidate file...")

output = pd.DataFrame(
    results
)

output.to_csv(
    OUTPUT_FILE,
    sep="\t",
    index=False
)


# ============================================================
# VALIDATION
# ============================================================

print("\n" + "=" * 75)
print("V7.1 VALIDATION")
print("=" * 75)

print(
    "Source 1 rows:",
    len(s1)
)

print(
    "Output rows:",
    len(output)
)

print(
    "Duplicate Source-1 IDs:",
    output[
        "source1_entity_id"
    ].duplicated().sum()
)

empty_count = (
    output[
        "candidate_entity_ids"
    ]
    .fillna("")
    .eq("")
    .sum()
)

print(
    "Rows with empty candidates:",
    empty_count
)


candidate_counts = (
    output[
        "candidate_entity_ids"
    ]
    .fillna("")
    .apply(
        lambda x:
        len([
            y
            for y in x.split(",")
            if y
        ])
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


# ============================================================
# DONE
# ============================================================

print("\n" + "=" * 75)
print("V7.1 COMPLETE")
print("=" * 75)

print(
    "Saved:",
    OUTPUT_FILE
)

print(
    f"Total time: "
    f"{time.time() - start_time:.1f}s"
)

print("=" * 75)