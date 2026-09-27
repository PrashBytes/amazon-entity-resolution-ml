import pandas as pd
import re
import unicodedata


# ============================================================
# 1. NORMALIZATION
# ============================================================

def normalize_text(text):
    if pd.isna(text):
        return ""

    text = unicodedata.normalize("NFKD", str(text))
    text = text.encode("ascii", "ignore").decode("ascii")
    text = text.lower()

    # Keep only letters and numbers
    text = re.sub(r"[^a-z0-9\s]", " ", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


# ============================================================
# 2. TOKEN EXTRACTION
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
    "co",
    "the",
    "and",
    "of",
    "for",
    "group",
    "center",
    "centre",
}


def get_tokens(text):

    if not text:
        return set()

    words = text.split()

    useful_words = set()

    for word in words:

        # Ignore very short words
        if len(word) < 5:
            continue

        # Ignore generic business words
        if word in STOPWORDS:
            continue

        useful_words.add(word)

    return useful_words


# ============================================================
# 3. LOAD DATA
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

ground_truth = pd.read_csv(
    "dataset/train/train_ground_truth.tsv",
    sep="\t"
)

print("Source 1:", len(source1))
print("Source 2:", len(source2))
print("Source 3:", len(source3))
print("Ground truth:", len(ground_truth))


# ============================================================
# 4. NORMALIZE ALL SOURCES
# ============================================================

for df in [source1, source2, source3]:

    df["name_norm"] = (
        df["business_name"]
        .apply(normalize_text)
    )

    df["address_norm"] = (
        df["business_address"]
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
# 5. CREATE EXISTING BLOCKING KEYS
# ============================================================

def create_keys(df):

    # -----------------------------
    # NAME BLOCK
    # -----------------------------

    df["name_block"] = (
        df["country_norm"]
        + "_"
        + df["name_norm"].str[:6]
    )

    # -----------------------------
    # ADDRESS BLOCK
    # -----------------------------

    df["address_block"] = (
        df["country_norm"]
        + "_"
        + df["address_norm"].str[:8]
    )

    # -----------------------------
    # NAME + ADDRESS BLOCK
    # -----------------------------

    df["name_address_block"] = (
        df["country_norm"]
        + "_"
        + df["name_norm"].str[:4]
        + "_"
        + df["address_norm"].str[:6]
    )

    # -----------------------------
    # TOKEN BLOCK
    # -----------------------------

    df["name_tokens"] = (
        df["name_norm"]
        .apply(get_tokens)
    )

    return df


source1 = create_keys(source1)
source2 = create_keys(source2)
source3 = create_keys(source3)


# ============================================================
# 6. BUILD NORMAL BLOCK INDEXES
# ============================================================

print("\nBuilding normal indexes...")


def build_index(df, column):

    return (
        df.groupby(column)["entity_id"]
        .apply(set)
        .to_dict()
    )


s2_name = build_index(
    source2,
    "name_block"
)

s2_address = build_index(
    source2,
    "address_block"
)

s2_name_address = build_index(
    source2,
    "name_address_block"
)


s3_name = build_index(
    source3,
    "name_block"
)

s3_address = build_index(
    source3,
    "address_block"
)

s3_name_address = build_index(
    source3,
    "name_address_block"
)


# ============================================================
# 7. BUILD TOKEN INDEX
# ============================================================

print("Building token indexes...")


def build_token_index(df):

    token_index = {}

    for _, row in df.iterrows():

        country = row["country_norm"]

        tokens = row["name_tokens"]

        for token in tokens:

            key = country + "_" + token

            if key not in token_index:
                token_index[key] = set()

            token_index[key].add(
                row["entity_id"]
            )

    return token_index


s2_token = build_token_index(source2)
s3_token = build_token_index(source3)


print(
    "Source 2 token blocks:",
    len(s2_token)
)

print(
    "Source 3 token blocks:",
    len(s3_token)
)


# ============================================================
# 8. GROUND TRUTH
# ============================================================

truth = {}

for _, row in ground_truth.iterrows():

    ids = str(
        row["matched_entity_ids"]
    )

    truth[
        row["source1_entity_id"]
    ] = set(
        x.strip()
        for x in ids.split(",")
        if x.strip()
    )


# ============================================================
# 9. TEST ON 1000 SOURCE-1 RECORDS
# ============================================================

sample = source1.head(1000)

print("\n======================================")
print("TESTING TOKEN BLOCKING")
print("======================================")


old_candidates_total = 0
token_candidates_total = 0
final_candidates_total = 0

old_recall = 0
token_recall = 0
final_recall = 0


# ============================================================
# 10. PROCESS EACH SOURCE-1 RECORD
# ============================================================

for _, row in sample.iterrows():

    entity_id = row["entity_id"]

    true_matches = truth.get(
        entity_id,
        set()
    )

    # --------------------------------------------------------
    # EXISTING NAME BLOCK
    # --------------------------------------------------------

    name_candidates = s2_name.get(
        row["name_block"],
        set()
    )

    name_candidates = (
        name_candidates
        | s3_name.get(
            row["name_block"],
            set()
        )
    )


    # --------------------------------------------------------
    # EXISTING ADDRESS BLOCK
    # --------------------------------------------------------

    address_candidates = s2_address.get(
        row["address_block"],
        set()
    )

    address_candidates = (
        address_candidates
        | s3_address.get(
            row["address_block"],
            set()
        )
    )


    # --------------------------------------------------------
    # EXISTING NAME + ADDRESS BLOCK
    # --------------------------------------------------------

    name_address_candidates = (
        s2_name_address.get(
            row["name_address_block"],
            set()
        )
    )

    name_address_candidates = (
        name_address_candidates
        | s3_name_address.get(
            row["name_address_block"],
            set()
        )
    )


    # --------------------------------------------------------
    # OLD MULTI-VIEW SYSTEM
    # --------------------------------------------------------

    old_candidates = (
        name_candidates
        | address_candidates
        | name_address_candidates
    )


    # --------------------------------------------------------
    # TOKEN BLOCK
    # --------------------------------------------------------

    token_candidates = set()

    country = row["country_norm"]

    for token in row["name_tokens"]:

        key = country + "_" + token

        token_candidates.update(
            s2_token.get(key, set())
        )

        token_candidates.update(
            s3_token.get(key, set())
        )


    # --------------------------------------------------------
    # FINAL MULTI-VIEW + TOKEN SYSTEM
    # --------------------------------------------------------

    final_candidates = (
        old_candidates
        | token_candidates
    )


    # --------------------------------------------------------
    # STATISTICS
    # --------------------------------------------------------

    old_candidates_total += len(
        old_candidates
    )

    token_candidates_total += len(
        token_candidates
    )

    final_candidates_total += len(
        final_candidates
    )


    # --------------------------------------------------------
    # RECALL
    # --------------------------------------------------------

    if true_matches & old_candidates:
        old_recall += 1

    if true_matches & token_candidates:
        token_recall += 1

    if true_matches & final_candidates:
        final_recall += 1


# ============================================================
# 11. RESULTS
# ============================================================

n = len(sample)


print("\n======================================")
print("FINAL RESULTS")
print("======================================")


print("\nOLD MULTI-VIEW")
print(
    "Average candidates:",
    round(
        old_candidates_total / n,
        2
    )
)

print(
    "Recall:",
    round(
        old_recall / n * 100,
        2
    ),
    "%"
)


print("\nTOKEN BLOCK ONLY")
print(
    "Average candidates:",
    round(
        token_candidates_total / n,
        2
    )
)

print(
    "Recall:",
    round(
        token_recall / n * 100,
        2
    ),
    "%"
)


print("\nMULTI-VIEW + TOKEN")
print(
    "Average candidates:",
    round(
        final_candidates_total / n,
        2
    )
)

print(
    "Recall:",
    round(
        final_recall / n * 100,
        2
    ),
    "%"
)


print("\n======================================")
print("DONE")
print("======================================")