import pandas as pd
import re
import unicodedata
from collections import defaultdict
from rapidfuzz.fuzz import ratio


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

print("Source 1:", len(s1))
print("Source 2:", len(s2))
print("Source 3:", len(s3))


# ============================================================
# 3. TEST SAMPLE
# ============================================================

# IMPORTANT:
# We only test 1,000 records initially.
# We don't want to run millions of comparisons yet.

s1 = s1.head(1000).copy()


# ============================================================
# 4. NORMALIZE
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
# 5. V1 BLOCKING KEY
# ============================================================

def create_block(df):

    df["name_block"] = (
        df["country_norm"]
        + "_"
        + df["name_norm"].str[:6]
    )

    return df


s1 = create_block(s1)
s2 = create_block(s2)
s3 = create_block(s3)


# ============================================================
# 6. BUILD INDEX
# ============================================================

def build_index(df):

    index = defaultdict(list)

    for key, entity_id in zip(
        df["name_block"],
        df["entity_id"]
    ):

        if key:

            index[key].append(
                entity_id
            )

    return index


print("\nBuilding indexes...")

s2_index = build_index(s2)
s3_index = build_index(s3)


# ============================================================
# 7. MAKE LOOKUP TABLE
# ============================================================

print("Creating lookup tables...")

s2_lookup = s2.set_index(
    "entity_id"
)[
    [
        "business_name",
        "business_address",
        "name_norm",
        "address_norm"
    ]
].to_dict("index")


s3_lookup = s3.set_index(
    "entity_id"
)[
    [
        "business_name",
        "business_address",
        "name_norm",
        "address_norm"
    ]
].to_dict("index")


# ============================================================
# 8. RAPIDFUZZ RANKING
# ============================================================

def rank_candidates(row, candidate_ids, lookup):

    results = []

    source_name = row["name_norm"]
    source_address = row["address_norm"]

    for entity_id in candidate_ids:

        candidate = lookup.get(
            entity_id
        )

        if candidate is None:
            continue

        # ----------------------------------------------------
        # NAME SIMILARITY
        # ----------------------------------------------------

        name_score = ratio(
            source_name,
            candidate["name_norm"]
        )

        # ----------------------------------------------------
        # ADDRESS SIMILARITY
        # ----------------------------------------------------

        address_score = ratio(
            source_address,
            candidate["address_norm"]
        )

        # ----------------------------------------------------
        # COMBINED SCORE
        # ----------------------------------------------------

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
# 9. PROCESS SOURCE 1
# ============================================================

print("\n======================================")
print("V5 RAPIDFUZZ RANKING")
print("======================================")


all_results = []


for i, (_, row) in enumerate(
    s1.iterrows()
):

    block = row["name_block"]

    s2_candidates = s2_index.get(
        block,
        []
    )

    s3_candidates = s3_index.get(
        block,
        []
    )


    # --------------------------------------------------------
    # Rank Source 2
    # --------------------------------------------------------

    ranked_s2 = rank_candidates(
        row,
        s2_candidates,
        s2_lookup
    )


    # --------------------------------------------------------
    # Rank Source 3
    # --------------------------------------------------------

    ranked_s3 = rank_candidates(
        row,
        s3_candidates,
        s3_lookup
    )


    # --------------------------------------------------------
    # Keep top 10 from each source
    # --------------------------------------------------------

    top_s2 = ranked_s2[:10]
    top_s3 = ranked_s3[:10]


    for entity_id, name_score, address_score, final_score in top_s2:

        all_results.append({
            "source1_entity_id": row["entity_id"],
            "candidate_entity_id": entity_id,
            "candidate_source": "source2",
            "name_score": name_score,
            "address_score": address_score,
            "final_score": final_score
        })


    for entity_id, name_score, address_score, final_score in top_s3:

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
# 10. SAVE RESULTS
# ============================================================

results_df = pd.DataFrame(
    all_results
)

results_df.to_csv(
    "v5_candidates.csv",
    index=False
)


# ============================================================
# 11. SHOW RESULTS
# ============================================================

print("\n======================================")
print("V5 RESULTS")
print("======================================")

print(
    "Source 1 records:",
    len(s1)
)

print(
    "Ranked candidate pairs:",
    len(results_df)
)

print(
    "Average candidates kept per Source 1:",
    round(
        len(results_df) / len(s1),
        2
    )
)

print("\nTop candidate examples:")

print(
    results_df
    .sort_values(
        "final_score",
        ascending=False
    )
    .head(20)
    .to_string(index=False)
)

print("\nSaved:")
print("v5_candidates.csv")

print("\nDONE!")