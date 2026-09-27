import pandas as pd
import re
import unicodedata
from collections import defaultdict, Counter


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
# 3. NORMALIZATION
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
# 4. BLOCKING KEYS
# ============================================================

print("\nCreating blocking keys...")


def create_blocks(df):

    name = df["name_norm"]
    address = df["address_norm"]
    country = df["country_norm"]

    # Precise name block
    df["smart_name_block"] = (
        country
        + "_"
        + name.str[:4]
        + "_"
        + name.str[-4:]
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


s1 = create_blocks(s1)
s2 = create_blocks(s2)
s3 = create_blocks(s3)


# ============================================================
# 5. BUILD BLOCK INDEX
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


print("\nBuilding Source 2 indexes...")

s2_smart = build_index(
    s2,
    "smart_name_block"
)

s2_combined = build_index(
    s2,
    "combined_block"
)


print("Building Source 3 indexes...")

s3_smart = build_index(
    s3,
    "smart_name_block"
)

s3_combined = build_index(
    s3,
    "combined_block"
)


print("Normal indexes ready!")


# ============================================================
# 6. TOKENIZATION
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
    "india",
    "usa",
    "the",
    "and",
    "of",
    "for",
    "with",
    "international"
}


def get_useful_tokens(name):

    tokens = name.split()

    result = []

    for token in tokens:

        if len(token) < 4:
            continue

        if token in STOPWORDS:
            continue

        result.append(token)

    return result


# ============================================================
# 7. TOKEN FREQUENCIES
# ============================================================

print("\nCounting name tokens...")

token_counter = Counter()

for name in s2["name_norm"]:

    tokens = set(
        get_useful_tokens(name)
    )

    for token in tokens:
        token_counter[token] += 1


for name in s3["name_norm"]:

    tokens = set(
        get_useful_tokens(name)
    )

    for token in tokens:
        token_counter[token] += 1


print(
    "Total unique tokens:",
    len(token_counter)
)


# ============================================================
# 8. SELECT ONLY RARE/USEFUL TOKENS
# ============================================================

# A token appearing thousands of times is not useful
# for candidate generation.

MAX_TOKEN_FREQUENCY = 500

useful_tokens = {
    token
    for token, count in token_counter.items()
    if count <= MAX_TOKEN_FREQUENCY
}


print(
    "Useful tokens:",
    len(useful_tokens)
)


# ============================================================
# 9. CREATE TOKEN PAIRS
# ============================================================

def get_token_pairs(name):

    tokens = set(
        token
        for token in get_useful_tokens(name)
        if token in useful_tokens
    )

    tokens = sorted(tokens)

    pairs = []

    for i in range(len(tokens)):

        for j in range(
            i + 1,
            len(tokens)
        ):

            pairs.append(
                (
                    tokens[i],
                    tokens[j]
                )
            )

    return pairs


# ============================================================
# 10. BUILD TOKEN-PAIR INDEX
# ============================================================

print("\nBuilding token-pair indexes...")

s2_pair_index = defaultdict(set)
s3_pair_index = defaultdict(set)


for entity_id, name in zip(
    s2["entity_id"],
    s2["name_norm"]
):

    pairs = get_token_pairs(name)

    for pair in pairs:

        s2_pair_index[pair].add(
            entity_id
        )


for entity_id, name in zip(
    s3["entity_id"],
    s3["name_norm"]
):

    pairs = get_token_pairs(name)

    for pair in pairs:

        s3_pair_index[pair].add(
            entity_id
        )


print("Token-pair indexes ready!")


# ============================================================
# 11. GROUND TRUTH
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
# 12. CANDIDATE GENERATION
# ============================================================

def get_candidates(row):

    candidates = set()


    # --------------------------------------------------------
    # VIEW 1: SMART NAME BLOCK
    # --------------------------------------------------------

    key = row[
        "smart_name_block"
    ]

    candidates.update(
        s2_smart.get(
            key,
            set()
        )
    )

    candidates.update(
        s3_smart.get(
            key,
            set()
        )
    )


    # --------------------------------------------------------
    # VIEW 2: COMBINED NAME + ADDRESS
    # --------------------------------------------------------

    key = row[
        "combined_block"
    ]

    candidates.update(
        s2_combined.get(
            key,
            set()
        )
    )

    candidates.update(
        s3_combined.get(
            key,
            set()
        )
    )


    # --------------------------------------------------------
    # VIEW 3: TWO-TOKEN MATCH
    # --------------------------------------------------------

    pairs = get_token_pairs(
        row["name_norm"]
    )

    # Only use the strongest few pairs.
    #
    # Rarer tokens are more informative.

    ranked_pairs = sorted(
        pairs,
        key=lambda pair:
            token_counter[pair[0]]
            +
            token_counter[pair[1]]
    )

    for pair in ranked_pairs[:3]:

        candidates.update(
            s2_pair_index.get(
                pair,
                set()
            )
        )

        candidates.update(
            s3_pair_index.get(
                pair,
                set()
            )
        )


    return candidates


# ============================================================
# 13. TEST V4
# ============================================================

print("\n======================================")
print("TESTING V4")
print("======================================")

sample = s1.head(10000)

total_true_matches = 0
found_true_matches = 0

candidate_counts = []

rows_with_truth = 0


for i, (_, row) in enumerate(
    sample.iterrows()
):

    source1_id = row[
        "entity_id"
    ]

    true_matches = truth.get(
        source1_id,
        set()
    )

    if not true_matches:
        continue

    rows_with_truth += 1

    candidates = get_candidates(
        row
    )

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
# 14. RESULTS
# ============================================================

print("\n======================================")
print("V4 RESULTS")
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
        /
        total_true_matches
    ) * 100

    print(
        f"V4 Recall: {recall:.2f}%"
    )


if candidate_counts:

    average = (
        sum(candidate_counts)
        /
        len(candidate_counts)
    )

    sorted_counts = sorted(
        candidate_counts
    )

    median = sorted_counts[
        len(sorted_counts) // 2
    ]

    print(
        "Average candidates:",
        round(average, 2)
    )

    print(
        "Median candidates:",
        median
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