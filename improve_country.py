import pandas as pd
import os
import time

MATCH_FILE = "output/matching_results_NEW.tsv"
OUT_FILE = "output/matching_results_country.tsv"

S1_FILE = "dataset/test/test_source1.tsv"
S2_FILE = "dataset/test/test_source2.tsv"
S3_FILE = "dataset/test/test_source3.tsv"

CHUNK_SIZE = 50000

print("=" * 70)
print("FAST COUNTRY FILTER")
print("=" * 70)

# ---------------------------------------------------------
# 1. Load ONLY entity_id + country
# ---------------------------------------------------------

print("\nLoading country data...")

s1 = pd.read_csv(
    S1_FILE,
    sep="\t",
    usecols=["entity_id", "country"],
    dtype=str
).fillna("")

s2 = pd.read_csv(
    S2_FILE,
    sep="\t",
    usecols=["entity_id", "country"],
    dtype=str
).fillna("")

s3 = pd.read_csv(
    S3_FILE,
    sep="\t",
    usecols=["entity_id", "country"],
    dtype=str
).fillna("")

print("S1:", len(s1))
print("S2:", len(s2))
print("S3:", len(s3))

# ---------------------------------------------------------
# 2. Normalize country
# ---------------------------------------------------------

def norm_country(x):
    return (
        x.astype(str)
         .str.lower()
         .str.strip()
         .str.replace(r"[^a-z0-9]+", "", regex=True)
    )

s1["country"] = norm_country(s1["country"])
s2["country"] = norm_country(s2["country"])
s3["country"] = norm_country(s3["country"])

# Source-1 country lookup
s1_country = dict(zip(s1["entity_id"], s1["country"]))

# Candidate country lookup
candidate_country = {}

candidate_country.update(
    zip(s2["entity_id"], s2["country"])
)

candidate_country.update(
    zip(s3["entity_id"], s3["country"])
)

print("Country lookup ready:", len(candidate_country))

# ---------------------------------------------------------
# 3. Process matching results in chunks
# ---------------------------------------------------------

print("\nProcessing matching results...")
print("This replaces the slow iterrows() version.\n")

first_chunk = True
total = 0
changed = 0
start = time.time()

for chunk in pd.read_csv(
    MATCH_FILE,
    sep="\t",
    dtype=str,
    chunksize=CHUNK_SIZE
):

    chunk["matched_entity_ids"] = chunk["matched_entity_ids"].fillna("")

    def filter_candidates(row):

        source_id = row["source1_entity_id"]
        candidates = row["matched_entity_ids"]

        if not candidates:
            return candidates

        source_country = s1_country.get(source_id, "")

        # If source country is unavailable,
        # leave candidates unchanged.
        if not source_country:
            return candidates

        result = []

        for cid in candidates.split(","):
            cid = cid.strip()

            if candidate_country.get(cid, "") == source_country:
                result.append(cid)

        # IMPORTANT:
        # Don't destroy a row if country filtering
        # removes everything.
        if result:
            return ",".join(result)

        return candidates

    chunk["matched_entity_ids"] = chunk.apply(
        filter_candidates,
        axis=1
    )

    if first_chunk:
        chunk.to_csv(
            OUT_FILE,
            sep="\t",
            index=False,
            mode="w"
        )
        first_chunk = False
    else:
        chunk.to_csv(
            OUT_FILE,
            sep="\t",
            index=False,
            mode="a",
            header=False
        )

    total += len(chunk)

    print(
        f"Processed {total:,} rows | "
        f"Elapsed {(time.time()-start)/60:.1f} min"
    )

print("\n" + "=" * 70)
print("DONE")
print("=" * 70)

print("Saved:", OUT_FILE)
print("Rows:", total)
print("Time:", round((time.time()-start)/60, 2), "minutes")