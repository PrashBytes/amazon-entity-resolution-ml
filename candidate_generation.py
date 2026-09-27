import pandas as pd
import re
import unicodedata
from collections import defaultdict


# ============================================================
# 1. TEXT NORMALIZATION
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


# ============================================================
# 2. LOAD DATA
# ============================================================

print("Loading datasets...")

source1 = pd.read_csv(
    "dataset/train/train_source1.tsv",
    sep="\t"
)

source2 = pd.read_csv(
    "dataset/train/train_source2.tsv",
    sep="\t"
)

source3 = pd.read_csv(
    "dataset/train/train_source3.tsv",
    sep="\t"
)

print("Source 1:", len(source1))
print("Source 2:", len(source2))
print("Source 3:", len(source3))


# ============================================================
# 3. NORMALIZE DATA
# ============================================================

print("\nNormalizing Source 1...")

source1["name_norm"] = (
    source1["business_name"]
    .fillna("")
    .apply(normalize_text)
)

source1["address_norm"] = (
    source1["business_address"]
    .fillna("")
    .apply(normalize_text)
)

source1["country_norm"] = (
    source1["country"]
    .fillna("")
    .astype(str)
    .str.lower()
    .str.strip()
)


print("Normalizing Source 2...")

source2["name_norm"] = (
    source2["business_name"]
    .fillna("")
    .apply(normalize_text)
)

source2["address_norm"] = (
    source2["business_address"]
    .fillna("")
    .apply(normalize_text)
)

source2["country_norm"] = (
    source2["country"]
    .fillna("")
    .astype(str)
    .str.lower()
    .str.strip()
)


print("Normalizing Source 3...")

source3["name_norm"] = (
    source3["business_name"]
    .fillna("")
    .apply(normalize_text)
)

source3["address_norm"] = (
    source3["business_address"]
    .fillna("")
    .apply(normalize_text)
)

source3["country_norm"] = (
    source3["country"]
    .fillna("")
    .astype(str)
    .str.lower()
    .str.strip()
)


# ============================================================
# 4. CREATE BLOCKING KEYS
# ============================================================

def create_blocks(df):

    # VIEW 1
    # Country + first 6 characters of name

    df["name_block"] = (
        df["country_norm"]
        + "_"
        + df["name_norm"].str[:6]
    )

    # VIEW 2
    # Country + first 8 characters of address

    df["address_block"] = (
        df["country_norm"]
        + "_"
        + df["address_norm"].str[:8]
    )

    # VIEW 3
    # Country + beginning of name + beginning of address

    df["name_address_block"] = (
        df["country_norm"]
        + "_"
        + df["name_norm"].str[:4]
        + "_"
        + df["address_norm"].str[:6]
    )

    return df


print("\nCreating blocking keys...")

source1 = create_blocks(source1)
source2 = create_blocks(source2)
source3 = create_blocks(source3)


# ============================================================
# 5. BUILD FAST INDEX
# ============================================================

def build_index(df, column):

    index = defaultdict(list)

    for key, entity_id in zip(
        df[column].values,
        df["entity_id"].values
    ):
        index[key].append(entity_id)

    return index


print("\nBuilding Source 2 indexes...")

s2_name_index = build_index(
    source2,
    "name_block"
)

s2_address_index = build_index(
    source2,
    "address_block"
)

s2_name_address_index = build_index(
    source2,
    "name_address_block"
)


print("Building Source 3 indexes...")

s3_name_index = build_index(
    source3,
    "name_block"
)

s3_address_index = build_index(
    source3,
    "address_block"
)

s3_name_address_index = build_index(
    source3,
    "name_address_block"
)


print("Indexes ready!")


# ============================================================
# 6. CANDIDATE GENERATION
# ============================================================

def get_candidates(row):

    candidates = set()

    # --------------------------------------------------------
    # NAME VIEW
    # --------------------------------------------------------

    key = row["name_block"]

    candidates.update(
        s2_name_index.get(key, [])
    )

    candidates.update(
        s3_name_index.get(key, [])
    )


    # --------------------------------------------------------
    # ADDRESS VIEW
    # --------------------------------------------------------

    key = row["address_block"]

    candidates.update(
        s2_address_index.get(key, [])
    )

    candidates.update(
        s3_address_index.get(key, [])
    )


    # --------------------------------------------------------
    # NAME + ADDRESS VIEW
    # --------------------------------------------------------

    key = row["name_address_block"]

    candidates.update(
        s2_name_address_index.get(key, [])
    )

    candidates.update(
        s3_name_address_index.get(key, [])
    )


    return candidates


# ============================================================
# 7. TEST ON 10,000 SOURCE-1 RECORDS
# ============================================================

print("\n======================================")
print("TESTING MULTI-VIEW CANDIDATE GENERATION")
print("======================================")

sample = source1.head(10000)

candidate_counts = []

for i, (_, row) in enumerate(sample.iterrows()):

    candidates = get_candidates(row)

    candidate_counts.append(len(candidates))

    if (i + 1) % 1000 == 0:

        print(
            f"Processed {i + 1}/10000"
        )


# ============================================================
# 8. RESULTS
# ============================================================

print("\n======================================")
print("RESULTS")
print("======================================")

print(
    "Source 1 tested:",
    len(sample)
)

print(
    "Average candidates:",
    round(
        sum(candidate_counts) /
        len(candidate_counts),
        2
    )
)

print(
    "Median candidates:",
    sorted(candidate_counts)[
        len(candidate_counts) // 2
    ]
)

print(
    "Maximum candidates:",
    max(candidate_counts)
)

print(
    "Minimum candidates:",
    min(candidate_counts)
)

print("\nDONE!")