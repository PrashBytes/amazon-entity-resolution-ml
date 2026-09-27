import pandas as pd
import re
import unicodedata
import time
from collections import defaultdict, Counter


# ============================================================
# CONFIG
# ============================================================

TEST_ROWS = 1000

MAX_CANDIDATES = 100

SOURCE1 = "dataset/train/train_source1.tsv"
SOURCE2 = "dataset/train/train_source2.tsv"
SOURCE3 = "dataset/train/train_source3.tsv"

OUTPUT_FILE = "v7_2_candidates.tsv"


# ============================================================
# HEADER
# ============================================================

print("=" * 78)
print("V7.2 TARGETED HIGH-RECALL CANDIDATE GENERATION")
print("=" * 78)


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

    text = re.sub(r"\s+", " ", text).strip()

    return text


def first_word(text):

    if not text:
        return ""

    return text.split()[0]


def last_word(text):

    if not text:
        return ""

    return text.split()[-1]


def name_signature(text):

    if not text:
        return ""

    words = text.split()

    # Keep first character of each word.
    # This catches word-order changes.
    chars = sorted(
        w[0]
        for w in words
        if w
    )

    return "".join(chars[:10])


def sorted_name(text):

    if not text:
        return ""

    return " ".join(
        sorted(text.split())
    )


def address_number(text):

    if not text:
        return ""

    nums = re.findall(
        r"\d+",
        text
    )

    if not nums:
        return ""

    return nums[0]


# ============================================================
# LOAD DATA
# ============================================================

start = time.time()

print("\nLoading datasets...")

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

print(
    "Loading time:",
    round(time.time() - start, 1),
    "seconds"
)


# ============================================================
# TEST MODE
# ============================================================

print("\n" + "=" * 78)
print(f"TEST MODE: {TEST_ROWS} Source-1 rows")
print("=" * 78)

if TEST_ROWS is not None:

    s1 = s1.head(TEST_ROWS).copy()

print(
    "\nChange TEST_ROWS = None only after testing."
)


# ============================================================
# NORMALIZATION
# ============================================================

print("\nNormalizing...")

start = time.time()


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


print(
    "Normalization time:",
    round(time.time() - start, 1),
    "seconds"
)


# ============================================================
# FEATURE PREPARATION
# ============================================================

print("\nPreparing candidate data...")

candidates = pd.concat(
    [
        s2,
        s3
    ],
    ignore_index=True
)

print(
    "Total candidate records:",
    len(candidates)
)


print("\nCreating blocking features...")

start = time.time()


def prepare_features(df):

    df["name_first_6"] = (
        df["name_norm"].str[:6]
    )

    df["name_first_5"] = (
        df["name_norm"].str[:5]
    )

    df["name_first_4"] = (
        df["name_norm"].str[:4]
    )

    df["name_first_word"] = (
        df["name_norm"].map(first_word)
    )

    df["name_last_word"] = (
        df["name_norm"].map(last_word)
    )

    df["name_signature"] = (
        df["name_norm"].map(name_signature)
    )

    df["sorted_name"] = (
        df["name_norm"].map(sorted_name)
    )

    df["address_number"] = (
        df["address_norm"].map(address_number)
    )

    df["address_prefix"] = (
        df["address_norm"].str[:10]
    )

    df["address_first_word"] = (
        df["address_norm"].map(first_word)
    )

    df["address_last_word"] = (
        df["address_norm"].map(last_word)
    )

    df["country_name_prefix"] = (
        df["country_norm"]
        + "|"
        + df["name_norm"].str[:5]
    )

    df["country_first_word"] = (
        df["country_norm"]
        + "|"
        + df["name_first_word"]
    )

    df["country_address_number"] = (
        df["country_norm"]
        + "|"
        + df["address_number"]
    )

    return df


s1 = prepare_features(s1)
candidates = prepare_features(candidates)


print(
    "Feature preparation time:",
    round(time.time() - start, 1),
    "seconds"
)


# ============================================================
# INDEX BUILDER
# ============================================================

def build_index(df, column):

    index = defaultdict(list)

    for entity_id, key in zip(
        df["entity_id"].values,
        df[column].values
    ):

        if not key:
            continue

        index[key].append(entity_id)

    return index


# ============================================================
# BUILD INDEXES
# ============================================================

print("\n" + "=" * 78)
print("BUILDING V7.2 BLOCKING INDEXES")
print("=" * 78)


indexes = {}


index_columns = [

    (
        "exact_name",
        "name_norm"
    ),

    (
        "name_first_6",
        "name_first_6"
    ),

    (
        "name_first_5",
        "name_first_5"
    ),

    (
        "name_first_4",
        "name_first_4"
    ),

    (
        "name_first_word",
        "name_first_word"
    ),

    (
        "name_last_word",
        "name_last_word"
    ),

    (
        "name_signature",
        "name_signature"
    ),

    (
        "sorted_name",
        "sorted_name"
    ),

    (
        "address_number",
        "address_number"
    ),

    (
        "address_prefix",
        "address_prefix"
    ),

    (
        "address_first_word",
        "address_first_word"
    ),

    (
        "address_last_word",
        "address_last_word"
    ),

    (
        "country_name_prefix",
        "country_name_prefix"
    ),

    (
        "country_first_word",
        "country_first_word"
    ),

    (
        "country_address_number",
        "country_address_number"
    )
]


for index_name, column_name in index_columns:

    print(
        f"  Building {index_name}..."
    )

    idx = build_index(
        candidates,
        column_name
    )

    indexes[index_name] = idx

    print(
        f"    keys: {len(idx)}"
    )


# ============================================================
# TOKEN INDEXES
# ============================================================

print("\nCalculating token frequencies...")

name_frequency = Counter()
address_frequency = Counter()


for value in candidates["name_norm"]:

    for token in set(value.split()):

        if len(token) >= 3:

            name_frequency[token] += 1


for value in candidates["address_norm"]:

    for token in set(value.split()):

        if len(token) >= 3:

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
# RARE TOKEN INDEXES
# ============================================================

print("\nBuilding rare-token indexes...")


rare_name_index = defaultdict(list)

rare_address_index = defaultdict(list)


# A token appearing in too many records is not useful
NAME_TOKEN_LIMIT = 200
ADDRESS_TOKEN_LIMIT = 300


for entity_id, name in zip(
    candidates["entity_id"].values,
    candidates["name_norm"].values
):

    tokens = set(name.split())

    for token in tokens:

        if (
            len(token) >= 3
            and name_frequency[token] <= NAME_TOKEN_LIMIT
        ):

            rare_name_index[token].append(
                entity_id
            )


for entity_id, address in zip(
    candidates["entity_id"].values,
    candidates["address_norm"].values
):

    tokens = set(address.split())

    for token in tokens:

        if (
            len(token) >= 3
            and address_frequency[token] <= ADDRESS_TOKEN_LIMIT
        ):

            rare_address_index[token].append(
                entity_id
            )


indexes["rare_name_token"] = rare_name_index
indexes["rare_address_token"] = rare_address_index


print(
    "Rare name tokens:",
    len(rare_name_index)
)

print(
    "Rare address tokens:",
    len(rare_address_index)
)


# ============================================================
# GENERATE CANDIDATES
# ============================================================

print("\n" + "=" * 78)
print("GENERATING V7.2 CANDIDATES")
print("=" * 78)

print(
    "Rows:",
    len(s1)
)

print(
    "Maximum candidates per row:",
    MAX_CANDIDATES
)


# ============================================================
# CANDIDATE GENERATION
# ============================================================

def generate_candidates(row):

    scores = defaultdict(float)


    # --------------------------------------------------------
    # IMPORTANT:
    #
    # These names MUST exactly match the names in `indexes`.
    #
    # This fixes:
    # KeyError: 'name first6'
    # --------------------------------------------------------

    blocks = [

        (
            "exact_name",
            row["name_norm"],
            30
        ),

        (
            "name_first_6",
            row["name_first_6"],
            20
        ),

        (
            "name_first_5",
            row["name_first_5"],
            15
        ),

        (
            "name_first_4",
            row["name_first_4"],
            10
        ),

        (
            "name_first_word",
            row["name_first_word"],
            8
        ),

        (
            "name_last_word",
            row["name_last_word"],
            6
        ),

        (
            "name_signature",
            row["name_signature"],
            12
        ),

        (
            "sorted_name",
            row["sorted_name"],
            12
        ),

        (
            "address_number",
            row["address_number"],
            8
        ),

        (
            "address_prefix",
            row["address_prefix"],
            7
        ),

        (
            "address_first_word",
            row["address_first_word"],
            5
        ),

        (
            "address_last_word",
            row["address_last_word"],
            4
        ),

        (
            "country_name_prefix",
            row["country_name_prefix"],
            12
        ),

        (
            "country_first_word",
            row["country_first_word"],
            8
        ),

        (
            "country_address_number",
            row["country_address_number"],
            8
        )
    ]


    # --------------------------------------------------------
    # NORMAL BLOCKS
    # --------------------------------------------------------

    for index_name, key, weight in blocks:

        if not key:
            continue

        index = indexes[index_name]

        matches = index.get(
            key,
            []
        )

        # Don't let a huge block dominate.
        if len(matches) > 300:

            matches = matches[:300]


        for entity_id in matches:

            scores[entity_id] += weight


    # --------------------------------------------------------
    # RARE NAME TOKENS
    # --------------------------------------------------------

    name_tokens = set(
        x
        for x in row["name_norm"].split()
        if len(x) >= 3
    )


    for token in name_tokens:

        matches = rare_name_index.get(
            token,
            []
        )

        for entity_id in matches:

            scores[entity_id] += 4


    # --------------------------------------------------------
    # RARE ADDRESS TOKENS
    # --------------------------------------------------------

    address_tokens = set(
        x
        for x in row["address_norm"].split()
        if len(x) >= 3
    )


    for token in address_tokens:

        matches = rare_address_index.get(
            token,
            []
        )

        for entity_id in matches:

            scores[entity_id] += 2


    # --------------------------------------------------------
    # SORT
    # --------------------------------------------------------

    ranked = sorted(
        scores.items(),
        key=lambda x: x[1],
        reverse=True
    )


    return [
        entity_id
        for entity_id, score
        in ranked[:MAX_CANDIDATES]
    ]


# ============================================================
# PROCESS SOURCE 1
# ============================================================

results = []

start = time.time()


for i, (_, row) in enumerate(
    s1.iterrows()
):

    candidate_ids = generate_candidates(
        row
    )


    results.append({

        "source1_entity_id":
            row["entity_id"],

        "candidate_entity_ids":
            ",".join(candidate_ids)
    })


    if (i + 1) % 100 == 0:

        elapsed = time.time() - start

        rate = (
            (i + 1) / elapsed
            if elapsed > 0
            else 0
        )

        remaining = (
            len(s1) - (i + 1)
        )

        eta = (
            remaining / rate
            if rate > 0
            else 0
        )

        print(
            f"Processed {i + 1:,}/{len(s1):,} "
            f"| {rate:.1f} rows/sec "
            f"| ETA {eta:.1f}s"
        )


# ============================================================
# SAVE
# ============================================================

result_df = pd.DataFrame(
    results
)


result_df.to_csv(
    OUTPUT_FILE,
    sep="\t",
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

candidate_counts = (
    result_df["candidate_entity_ids"]
    .fillna("")
    .apply(
        lambda x:
        len(
            [
                y
                for y in x.split(",")
                if y
            ]
        )
    )
)


print("\n" + "=" * 78)
print("V7.2 RESULTS")
print("=" * 78)


print(
    "Source 1 rows:",
    len(result_df)
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
    (candidate_counts > 0).sum()
)


print(
    "Empty candidate rows:",
    (candidate_counts == 0).sum()
)


print(
    "Processing time:",
    round(
        time.time() - start,
        1
    ),
    "seconds"
)


print(
    "\nSaved:",
    OUTPUT_FILE
)


print("\n" + "=" * 78)
print("DONE!")
print("=" * 78)