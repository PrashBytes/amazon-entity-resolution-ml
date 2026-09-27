import pandas as pd
import re
import unicodedata
from collections import defaultdict


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


# ============================================================
# LOAD DATA
# ============================================================

print("Loading datasets...")

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

ground_truth = pd.read_csv(
    "dataset/train/train_ground_truth.tsv",
    sep="\t"
)

print("Source 1:", len(s1))
print("Source 2:", len(s2))
print("Source 3:", len(s3))
print("Ground truth:", len(ground_truth))


# ============================================================
# NORMALIZE
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


# ============================================================
# SAME V2 BLOCKING
# ============================================================

def create_blocks(df):

    name = df["name_norm"]
    address = df["address_norm"]
    country = df["country_norm"]

    df["name_block"] = (
        country
        + "_"
        + name.str[:4]
        + "_"
        + name.str[-4:]
    )

    df["address_block"] = (
        country
        + "_"
        + address.str[:5]
        + "_"
        + address.str[-5:]
    )

    df["combined_block"] = (
        country
        + "_"
        + name.str[:3]
        + "_"
        + name.str[-3:]
        + "_"
        + address.str[:4]
    )

    return df


print("Creating blocking keys...")

s1 = create_blocks(s1)
s2 = create_blocks(s2)
s3 = create_blocks(s3)


# ============================================================
# BUILD INDEX
# ============================================================

def build_index(df, column):

    index = defaultdict(set)

    for key, entity_id in zip(
        df[column],
        df["entity_id"]
    ):

        if key:
            index[key].add(entity_id)

    return index


print("\nBuilding indexes...")

s2_name = build_index(s2, "name_block")
s2_address = build_index(s2, "address_block")
s2_combined = build_index(s2, "combined_block")

s3_name = build_index(s3, "name_block")
s3_address = build_index(s3, "address_block")
s3_combined = build_index(s3, "combined_block")


# ============================================================
# CANDIDATES
# ============================================================

def get_candidates(row):

    candidates = set()

    key = row["name_block"]

    candidates.update(
        s2_name.get(key, set())
    )

    candidates.update(
        s3_name.get(key, set())
    )

    key = row["address_block"]

    candidates.update(
        s2_address.get(key, set())
    )

    candidates.update(
        s3_address.get(key, set())
    )

    key = row["combined_block"]

    candidates.update(
        s2_combined.get(key, set())
    )

    candidates.update(
        s3_combined.get(key, set())
    )

    return candidates


# ============================================================
# GROUND TRUTH LOOKUP
# ============================================================

truth = {}

for _, row in ground_truth.iterrows():

    source1_id = row["source1_entity_id"]

    matched = str(
        row["matched_entity_ids"]
    )

    # Ground truth stores IDs separated by commas
    true_ids = set(
        x.strip()
        for x in matched.split(",")
        if x.strip()
    )

    truth[source1_id] = true_ids


# ============================================================
# TEST
# ============================================================

print("\n======================================")
print("EVALUATING V2 AGAINST GROUND TRUTH")
print("======================================")

# Test first 10,000 records
sample = s1.head(10000)

total_true_matches = 0
found_true_matches = 0

rows_with_truth = 0

candidate_counts = []


for i, (_, row) in enumerate(
    sample.iterrows()
):

    source1_id = row["entity_id"]

    true_matches = truth.get(
        source1_id,
        set()
    )

    if not true_matches:
        continue

    rows_with_truth += 1

    candidates = get_candidates(row)

    candidate_counts.append(
        len(candidates)
    )

    total_true_matches += len(true_matches)

    found = true_matches.intersection(
        candidates
    )

    found_true_matches += len(found)

    if (i + 1) % 1000 == 0:
        print(
            f"Processed {i + 1}/10000"
        )


# ============================================================
# FINAL RESULTS
# ============================================================

print("\n======================================")
print("V2 EVALUATION RESULTS")
print("======================================")

print(
    "Rows with ground truth:",
    rows_with_truth
)

print(
    "True matches:",
    total_true_matches
)

print(
    "True matches found:",
    found_true_matches
)

if total_true_matches > 0:

    recall = (
        found_true_matches
        / total_true_matches
    ) * 100

    print(
        f"V2 Recall: {recall:.2f}%"
    )

if candidate_counts:

    print(
        "Average candidates:",
        round(
            sum(candidate_counts)
            / len(candidate_counts),
            2
        )
    )

print("\nDONE!")