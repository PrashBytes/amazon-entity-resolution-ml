import pandas as pd
import re
import unicodedata
from collections import defaultdict, Counter


# ============================================================
# 1. NORMALIZATION
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
# 3. NORMALIZE
# ============================================================

print("\nNormalizing data...")

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
# 4. BASIC BLOCKING KEYS
# ============================================================

def create_blocks(df):

    name = df["name_norm"]
    address = df["address_norm"]
    country = df["country_norm"]

    # V1-style name block
    df["name_block"] = (
        country
        + "_"
        + name.str[:6]
    )

    # V2-style precise name block
    df["smart_name_block"] = (
        country
        + "_"
        + name.str[:4]
        + "_"
        + name.str[-4:]
    )

    # Address block
    df["address_block"] = (
        country
        + "_"
        + address.str[:5]
    )

    # Strong combined block
    df["combined_block"] = (
        country
        + "_"
        + name.str[:3]
        + "_"
        + address.str[:5]
    )

    return df


print("\nCreating blocking keys...")

s1 = create_blocks(s1)
s2 = create_blocks(s2)
s3 = create_blocks(s3)


# ============================================================
# 5. BUILD NORMAL BLOCK INDEXES
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


print("\nBuilding normal indexes...")

s2_name = build_index(s2, "name_block")
s2_smart_name = build_index(s2, "smart_name_block")
s2_address = build_index(s2, "address_block")
s2_combined = build_index(s2, "combined_block")

s3_name = build_index(s3, "name_block")
s3_smart_name = build_index(s3, "smart_name_block")
s3_address = build_index(s3, "address_block")
s3_combined = build_index(s3, "combined_block")

print("Normal indexes ready!")


# ============================================================
# 6. INFORMATIVE TOKEN EXTRACTION
# ============================================================

STOPWORDS = {
    "inc",
    "llc",
    "ltd",
    "limited",
    "private",
    "company",
    "corporation",
    "corp",
    "group",
    "center",
    "centre",
    "services",
    "service",
    "the",
    "and",
    "of",
    "for",
    "india",
    "usa",
    "us"
}


def get_tokens(name):

    tokens = name.split()

    useful = []

    for token in tokens:

        if len(token) < 4:
            continue

        if token in STOPWORDS:
            continue

        useful.append(token)

    return useful


# ============================================================
# 7. FIND TOKEN FREQUENCIES
# ============================================================

print("\nFinding informative tokens...")

token_counter = Counter()

# We use names only.
# This keeps the token index much smaller than indexing
# every word from every address.

for name in s2["name_norm"]:

    tokens = set(get_tokens(name))

    for token in tokens:
        token_counter[token] += 1


for name in s3["name_norm"]:

    tokens = set(get_tokens(name))

    for token in tokens:
        token_counter[token] += 1


# Rare enough to be useful, but not so rare that they
# only catch accidental records.

MAX_TOKEN_FREQUENCY = 3000

useful_tokens = {
    token
    for token, count in token_counter.items()
    if count <= MAX_TOKEN_FREQUENCY
}


print(
    "Total unique tokens:",
    len(token_counter)
)

print(
    "Informative tokens:",
    len(useful_tokens)
)


# ============================================================
# 8. BUILD TOKEN INDEX
# ============================================================

print("\nBuilding informative token index...")

s2_token_index = defaultdict(set)
s3_token_index = defaultdict(set)


for entity_id, name in zip(
    s2["entity_id"],
    s2["name_norm"]
):

    tokens = set(get_tokens(name))

    for token in tokens:

        if token in useful_tokens:

            s2_token_index[token].add(
                entity_id
            )


for entity_id, name in zip(
    s3["entity_id"],
    s3["name_norm"]
):

    tokens = set(get_tokens(name))

    for token in tokens:

        if token in useful_tokens:

            s3_token_index[token].add(
                entity_id
            )


print("Token indexes ready!")


# ============================================================
# 9. CANDIDATE GENERATION
# ============================================================

def get_candidates(row):

    candidates = set()


    # --------------------------------------------------------
    # VIEW 1
    # BROAD NAME BLOCK
    # --------------------------------------------------------

    key = row["name_block"]

    candidates.update(
        s2_name.get(key, set())
    )

    candidates.update(
        s3_name.get(key, set())
    )


    # --------------------------------------------------------
    # VIEW 2
    # SMART NAME BLOCK
    # --------------------------------------------------------

    key = row["smart_name_block"]

    candidates.update(
        s2_smart_name.get(key, set())
    )

    candidates.update(
        s3_smart_name.get(key, set())
    )


    # --------------------------------------------------------
    # VIEW 3
    # ADDRESS BLOCK
    # --------------------------------------------------------

    key = row["address_block"]

    candidates.update(
        s2_address.get(key, set())
    )

    candidates.update(
        s3_address.get(key, set())
    )


    # --------------------------------------------------------
    # VIEW 4
    # COMBINED BLOCK
    # --------------------------------------------------------

    key = row["combined_block"]

    candidates.update(
        s2_combined.get(key, set())
    )

    candidates.update(
        s3_combined.get(key, set())
    )


    # --------------------------------------------------------
    # VIEW 5
    # INFORMATIVE TOKEN BLOCK
    # --------------------------------------------------------

    tokens = get_tokens(
        row["name_norm"]
    )

    # Use at most the 2 rarest useful tokens.
    ranked_tokens = sorted(
        [
            token
            for token in tokens
            if token in useful_tokens
        ],
        key=lambda x: token_counter[x]
    )

    for token in ranked_tokens[:2]:

        candidates.update(
            s2_token_index.get(
                token,
                set()
            )
        )

        candidates.update(
            s3_token_index.get(
                token,
                set()
            )
        )


    return candidates


# ============================================================
# 10. LOAD GROUND TRUTH
# ============================================================

truth = {}

for _, row in ground_truth.iterrows():

    source1_id = row[
        "source1_entity_id"
    ]

    matched = str(
        row["matched_entity_ids"]
    )

    true_ids = set(
        x.strip()
        for x in matched.split(",")
        if x.strip()
    )

    truth[source1_id] = true_ids


# ============================================================
# 11. TEST V3
# ============================================================

print("\n======================================")
print("TESTING V3")
print("======================================")

sample = s1.head(10000)

total_true_matches = 0
found_true_matches = 0

candidate_counts = []

rows_with_truth = 0


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

    total_true_matches += len(
        true_matches
    )

    found_true_matches += len(
        true_matches.intersection(
            candidates
        )
    )

    if (i + 1) % 1000 == 0:

        print(
            f"Processed {i + 1}/10000"
        )


# ============================================================
# 12. FINAL RESULTS
# ============================================================

print("\n======================================")
print("V3 RESULTS")
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
        f"V3 Recall: {recall:.2f}%"
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

    sorted_counts = sorted(
        candidate_counts
    )

    print(
        "Median candidates:",
        sorted_counts[
            len(sorted_counts) // 2
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