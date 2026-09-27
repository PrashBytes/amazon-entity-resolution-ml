import pandas as pd
import re
import unicodedata
from collections import defaultdict
from rapidfuzz.fuzz import ratio


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

s1 = pd.read_csv("dataset/train/train_source1.tsv", sep="\t")
s2 = pd.read_csv("dataset/train/train_source2.tsv", sep="\t")
s3 = pd.read_csv("dataset/train/train_source3.tsv", sep="\t")

print("Source 1:", len(s1))
print("Source 2:", len(s2))
print("Source 3:", len(s3))


# Test only 1000 first
s1 = s1.head(1000).copy()


# ============================================================
# NORMALIZE
# ============================================================

print("\nNormalizing...")

for df in [s1, s2, s3]:

    df["name_norm"] = df["business_name"].fillna("").apply(
        normalize_text
    )

    df["address_norm"] = df["business_address"].fillna("").apply(
        normalize_text
    )

    df["country_norm"] = (
        df["country"]
        .fillna("")
        .astype(str)
        .str.lower()
        .str.strip()
    )


# ============================================================
# HIGH-RECALL BLOCKING
# ============================================================

print("\nCreating high-recall blocking keys...")


def make_keys(df):

    # Several blocking views instead of only one

    df["key1"] = (
        df["country_norm"]
        + "_"
        + df["name_norm"].str[:6]
    )

    df["key2"] = (
        df["country_norm"]
        + "_"
        + df["name_norm"].str[:4]
    )

    df["key3"] = (
        df["country_norm"]
        + "_"
        + df["name_norm"].str[-5:]
    )

    return df


s1 = make_keys(s1)
s2 = make_keys(s2)
s3 = make_keys(s3)


# ============================================================
# BUILD INDEX
# ============================================================

def build_index(df):

    index = defaultdict(set)

    for _, row in df.iterrows():

        for key_column in ["key1", "key2", "key3"]:

            key = row[key_column]

            if key:
                index[key].add(row["entity_id"])

    return index


print("Building indexes...")

s2_index = build_index(s2)
s3_index = build_index(s3)

print("Indexes ready!")


# ============================================================
# LOOKUPS
# ============================================================

s2_lookup = s2.set_index("entity_id").to_dict("index")
s3_lookup = s3.set_index("entity_id").to_dict("index")


# ============================================================
# RANK CANDIDATES
# ============================================================

def rank_candidates(row, candidate_ids, lookup):

    results = []

    source_name = row["name_norm"]
    source_address = row["address_norm"]

    for entity_id in candidate_ids:

        candidate = lookup.get(entity_id)

        if candidate is None:
            continue

        name_score = ratio(
            source_name,
            candidate["name_norm"]
        )

        address_score = ratio(
            source_address,
            candidate["address_norm"]
        )

        # Name is slightly more important
        final_score = (
            0.65 * name_score
            +
            0.35 * address_score
        )

        results.append(
            (
                entity_id,
                name_score,
                address_score,
                final_score
            )
        )

    results.sort(
        key=lambda x: x[3],
        reverse=True
    )

    return results


# ============================================================
# PROCESS
# ============================================================

print("\n======================================")
print("V6 HIGH-RECALL + RAPIDFUZZ")
print("======================================")


all_results = []


for i, (_, row) in enumerate(s1.iterrows()):

    # --------------------------------------------------------
    # COLLECT CANDIDATES FROM ALL 3 BLOCKING VIEWS
    # --------------------------------------------------------

    candidate_s2 = set()
    candidate_s3 = set()

    for key_column in ["key1", "key2", "key3"]:

        key = row[key_column]

        if key:

            candidate_s2.update(
                s2_index.get(key, set())
            )

            candidate_s3.update(
                s3_index.get(key, set())
            )


    # --------------------------------------------------------
    # RANK
    # --------------------------------------------------------

    ranked_s2 = rank_candidates(
        row,
        candidate_s2,
        s2_lookup
    )

    ranked_s3 = rank_candidates(
        row,
        candidate_s3,
        s3_lookup
    )


    # --------------------------------------------------------
    # KEEP TOP 50
    # --------------------------------------------------------

    for entity_id, name_score, address_score, final_score in ranked_s2[:50]:

        all_results.append({
            "source1_entity_id": row["entity_id"],
            "candidate_entity_id": entity_id,
            "candidate_source": "source2",
            "name_score": name_score,
            "address_score": address_score,
            "final_score": final_score
        })


    for entity_id, name_score, address_score, final_score in ranked_s3[:50]:

        all_results.append({
            "source1_entity_id": row["entity_id"],
            "candidate_entity_id": entity_id,
            "candidate_source": "source3",
            "name_score": name_score,
            "address_score": address_score,
            "final_score": final_score
        })


    if (i + 1) % 100 == 0:

        print(
            f"Processed {i + 1}/1000"
        )


# ============================================================
# SAVE
# ============================================================

results = pd.DataFrame(all_results)

results.to_csv(
    "v6_candidates.csv",
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

print("\n======================================")
print("V6 RESULTS")
print("======================================")

print(
    "Source 1 tested:",
    len(s1)
)

print(
    "Candidate pairs:",
    len(results)
)

print(
    "Average candidates:",
    round(
        len(results) / len(s1),
        2
    )
)

print(
    "Maximum candidates:",
    results.groupby(
        "source1_entity_id"
    ).size().max()
)

print(
    "\nSaved: v6_candidates.csv"
)

print("\nDONE!")