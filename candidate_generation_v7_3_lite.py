import pandas as pd
import re
import time
from collections import defaultdict


# ============================================================
# CONFIGURATION
# ============================================================

TEST_ROWS = 1000
MAX_CANDIDATES = 100

S1_FILE = "dataset/train/train_source1.tsv"
S2_FILE = "dataset/train/train_source2.tsv"
S3_FILE = "dataset/train/train_source3.tsv"

OUTPUT_FILE = "v7_3_lite_candidates.tsv"


# ============================================================
# FAST NORMALIZATION
# ============================================================

def normalize_text_series(series):

    return (
        series.fillna("")
        .astype(str)
        .str.lower()
        .str.replace(r"[^a-z0-9\s]", " ", regex=True)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )


# ============================================================
# START
# ============================================================

START = time.time()

print("=" * 78)
print("V7.3-LITE FAST HIGH-RECALL CANDIDATE GENERATION")
print("=" * 78)

print("\nLoading datasets...")

s1 = pd.read_csv(S1_FILE, sep="\t", dtype=str)
s2 = pd.read_csv(S2_FILE, sep="\t", dtype=str)
s3 = pd.read_csv(S3_FILE, sep="\t", dtype=str)

print("Source 1:", len(s1))
print("Source 2:", len(s2))
print("Source 3:", len(s3))


# ============================================================
# TEST MODE
# ============================================================

if TEST_ROWS is not None:
    print("\n" + "=" * 78)
    print(f"TEST MODE: {TEST_ROWS} Source-1 rows")
    print("=" * 78)

    s1 = s1.head(TEST_ROWS).copy()

else:
    print("\nFULL MODE")


# ============================================================
# NORMALIZATION
# ============================================================

print("\nNormalizing...")

t = time.time()

for df in [s1, s2, s3]:

    df["name_norm"] = normalize_text_series(
        df["business_name"]
    )

    df["address_norm"] = normalize_text_series(
        df["business_address"]
    )

    df["country_norm"] = (
        df["country"]
        .fillna("")
        .astype(str)
        .str.lower()
        .str.strip()
    )

print(
    f"Normalization time: {time.time() - t:.1f}s"
)


# ============================================================
# FEATURE CREATION
# ============================================================

print("\nCreating cheap blocking features...")

t = time.time()


def prepare_features(df):

    name = df["name_norm"]
    address = df["address_norm"]
    country = df["country_norm"]

    df["name_first_6"] = name.str[:6]
    df["name_first_5"] = name.str[:5]
    df["name_first_4"] = name.str[:4]

    df["name_first_word"] = (
        name.str.split().str[0].fillna("")
    )

    df["name_last_word"] = (
        name.str.split().str[-1].fillna("")
    )

    df["name_suffix_3"] = name.str[-3:]

    df["address_first_word"] = (
        address.str.split().str[0].fillna("")
    )

    df["address_last_word"] = (
        address.str.split().str[-1].fillna("")
    )

    # First number occurring in address
    df["address_number"] = (
        address.str.extract(
            r"(\d+)",
            expand=False
        ).fillna("")
    )

    # Country combinations
    df["country_name_prefix"] = (
        country
        + "_"
        + name.str[:5]
    )

    df["country_first_word"] = (
        country
        + "_"
        + df["name_first_word"]
    )

    df["country_name_suffix"] = (
        country
        + "_"
        + df["name_suffix_3"]
    )

    df["country_address_number"] = (
        country
        + "_"
        + df["address_number"]
    )

    return df


s1 = prepare_features(s1)
s2 = prepare_features(s2)
s3 = prepare_features(s3)

print(
    f"Feature preparation time: {time.time() - t:.1f}s"
)


# ============================================================
# COMBINE CANDIDATE DATA
# ============================================================

print("\nPreparing candidate data...")

candidate_df = pd.concat(
    [
        s2[
            [
                "entity_id",
                "name_norm",
                "address_norm",
                "name_first_6",
                "name_first_5",
                "name_first_4",
                "name_first_word",
                "name_last_word",
                "name_suffix_3",
                "address_first_word",
                "address_last_word",
                "address_number",
                "country_name_prefix",
                "country_first_word",
                "country_name_suffix",
                "country_address_number"
            ]
        ],
        s3[
            [
                "entity_id",
                "name_norm",
                "address_norm",
                "name_first_6",
                "name_first_5",
                "name_first_4",
                "name_first_word",
                "name_last_word",
                "name_suffix_3",
                "address_first_word",
                "address_last_word",
                "address_number",
                "country_name_prefix",
                "country_first_word",
                "country_name_suffix",
                "country_address_number"
            ]
        ]
    ],
    ignore_index=True
)

print(
    "Total candidate records:",
    len(candidate_df)
)


# ============================================================
# INDEX BUILDER
# ============================================================

def build_index(df, column):

    index = defaultdict(list)

    values = df[column].values
    ids = df["entity_id"].values

    for key, entity_id in zip(values, ids):

        if key:
            index[key].append(entity_id)

    return index


# ============================================================
# BUILD ONLY CHEAP / HIGH-VALUE INDEXES
# ============================================================

print("\n" + "=" * 78)
print("BUILDING V7.3-LITE INDEXES")
print("=" * 78)

indexes = {}

index_columns = [

    # Strong name blocks
    ("exact_name", "name_norm"),
    ("name_first_6", "name_first_6"),
    ("name_first_5", "name_first_5"),
    ("name_first_4", "name_first_4"),

    # Word recovery
    ("name_first_word", "name_first_word"),
    ("name_last_word", "name_last_word"),

    # Suffix recovery
    ("name_suffix_3", "name_suffix_3"),

    # Address recovery
    ("address_number", "address_number"),
    ("address_first_word", "address_first_word"),
    ("address_last_word", "address_last_word"),

    # Country-aware recovery
    ("country_name_prefix", "country_name_prefix"),
    ("country_first_word", "country_first_word"),
    ("country_name_suffix", "country_name_suffix"),
    ("country_address_number", "country_address_number"),
]


for index_name, column in index_columns:

    print(f"  Building {index_name}...")

    t = time.time()

    indexes[index_name] = build_index(
        candidate_df,
        column
    )

    print(
        f"    keys: {len(indexes[index_name])}"
        f" | {time.time() - t:.1f}s"
    )


# ============================================================
# GENERATE CANDIDATES
# ============================================================

print("\n" + "=" * 78)
print("GENERATING V7.3-LITE CANDIDATES")
print("=" * 78)

print("Rows:", len(s1))
print("Maximum candidates:", MAX_CANDIDATES)


# ------------------------------------------------------------
# Block weights
# ------------------------------------------------------------

BLOCK_WEIGHTS = {

    # Very strong
    "exact_name": 20,

    # Strong
    "name_first_6": 10,
    "name_first_5": 8,
    "name_first_4": 6,

    # Moderate
    "name_first_word": 5,
    "name_last_word": 4,
    "name_suffix_3": 4,

    # Address
    "address_number": 7,
    "address_first_word": 3,
    "address_last_word": 3,

    # Country combinations
    "country_name_prefix": 9,
    "country_first_word": 6,
    "country_name_suffix": 5,
    "country_address_number": 8,
}


# ============================================================
# ROW PROCESSING
# ============================================================

all_results = []

total_rows = len(s1)

process_start = time.time()


for i, (_, row) in enumerate(s1.iterrows()):

    scores = defaultdict(int)

    # --------------------------------------------------------
    # BLOCK DEFINITIONS
    # --------------------------------------------------------

    blocks = [

        (
            "exact_name",
            row["name_norm"]
        ),

        (
            "name_first_6",
            row["name_first_6"]
        ),

        (
            "name_first_5",
            row["name_first_5"]
        ),

        (
            "name_first_4",
            row["name_first_4"]
        ),

        (
            "name_first_word",
            row["name_first_word"]
        ),

        (
            "name_last_word",
            row["name_last_word"]
        ),

        (
            "name_suffix_3",
            row["name_suffix_3"]
        ),

        (
            "address_number",
            row["address_number"]
        ),

        (
            "address_first_word",
            row["address_first_word"]
        ),

        (
            "address_last_word",
            row["address_last_word"]
        ),

        (
            "country_name_prefix",
            row["country_name_prefix"]
        ),

        (
            "country_first_word",
            row["country_first_word"]
        ),

        (
            "country_name_suffix",
            row["country_name_suffix"]
        ),

        (
            "country_address_number",
            row["country_address_number"]
        ),
    ]


    # --------------------------------------------------------
    # COLLECT + VOTE
    # --------------------------------------------------------

    for block_name, key in blocks:

        if not key:
            continue

        index = indexes[block_name]

        matches = index.get(key)

        if not matches:
            continue

        weight = BLOCK_WEIGHTS[block_name]

        # Avoid letting enormous common blocks dominate
        if len(matches) > 5000:
            weight = max(1, weight // 3)

        elif len(matches) > 1000:
            weight = max(1, weight // 2)

        for entity_id in matches:

            scores[entity_id] += weight


    # --------------------------------------------------------
    # RANK
    # --------------------------------------------------------

    if scores:

        ranked = sorted(
            scores.items(),
            key=lambda x: x[1],
            reverse=True
        )

        candidate_ids = [
            entity_id
            for entity_id, score
            in ranked[:MAX_CANDIDATES]
        ]

    else:

        candidate_ids = []


    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    all_results.append({

        "source1_entity_id":
            row["entity_id"],

        "candidate_entity_ids":
            ",".join(candidate_ids)

    })


    # --------------------------------------------------------
    # PROGRESS
    # --------------------------------------------------------

    if (i + 1) % 100 == 0:

        elapsed = time.time() - process_start

        rate = (i + 1) / elapsed

        remaining = total_rows - (i + 1)

        eta = remaining / rate if rate > 0 else 0

        print(
            f"Processed {i + 1:,}/{total_rows:,}"
            f" | {rate:.1f} rows/sec"
            f" | ETA {eta:.1f}s"
        )


# ============================================================
# SAVE
# ============================================================

results = pd.DataFrame(all_results)

results.to_csv(
    OUTPUT_FILE,
    sep="\t",
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

candidate_counts = (
    results["candidate_entity_ids"]
    .fillna("")
    .apply(
        lambda x:
        len(x.split(",")) if x else 0
    )
)

print("\n" + "=" * 78)
print("V7.3-LITE RESULTS")
print("=" * 78)

print(
    "Source 1 rows:",
    len(results)
)

print(
    "Average candidates:",
    round(candidate_counts.mean(), 2)
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
    round(time.time() - process_start, 1),
    "seconds"
)

print(
    "\nSaved:",
    OUTPUT_FILE
)

print(
    "Total runtime:",
    round(time.time() - START, 1),
    "seconds"
)

print("\n" + "=" * 78)
print("DONE!")
print("=" * 78)