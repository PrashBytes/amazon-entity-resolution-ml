import pandas as pd
import shutil
import os

print("=" * 70)
print("CREATING FALLBACK SUBMISSION")
print("=" * 70)

MATCH_FILE = "output/matching_results.tsv"
CANDIDATE_FILE = "output/candidate_pairs.tsv"

BACKUP_FILE = "output/matching_results_BACKUP.tsv"
FALLBACK_FILE = "output/matching_results_fallback.tsv"

# ------------------------------------------------------------
# 1. CHECK FILES
# ------------------------------------------------------------

if not os.path.exists(MATCH_FILE):
    raise FileNotFoundError(f"Missing: {MATCH_FILE}")

if not os.path.exists(CANDIDATE_FILE):
    raise FileNotFoundError(f"Missing: {CANDIDATE_FILE}")

# ------------------------------------------------------------
# 2. BACKUP ORIGINAL
# ------------------------------------------------------------

print("\nBacking up original submission...")

shutil.copy2(MATCH_FILE, BACKUP_FILE)

print(f"Backup saved: {BACKUP_FILE}")

# ------------------------------------------------------------
# 3. LOAD FILES
# ------------------------------------------------------------

print("\nLoading matching results...")

matches = pd.read_csv(
    MATCH_FILE,
    sep="\t",
    dtype=str
)

print(f"Matching rows: {len(matches):,}")

print("\nLoading candidate pairs...")

candidates = pd.read_csv(
    CANDIDATE_FILE,
    sep="\t",
    dtype=str
)

print(f"Candidate rows: {len(candidates):,}")

# ------------------------------------------------------------
# 4. VALIDATE COLUMNS
# ------------------------------------------------------------

required_match_cols = [
    "source1_entity_id",
    "matched_entity_ids"
]

required_candidate_cols = [
    "source1_entity_id",
    "candidate_entity_ids"
]

for col in required_match_cols:
    if col not in matches.columns:
        raise ValueError(f"Missing column in matching file: {col}")

for col in required_candidate_cols:
    if col not in candidates.columns:
        raise ValueError(f"Missing column in candidate file: {col}")

# ------------------------------------------------------------
# 5. MERGE CANDIDATES
# ------------------------------------------------------------

print("\nMatching candidate rows to Source-1 IDs...")

candidate_map = candidates.set_index(
    "source1_entity_id"
)["candidate_entity_ids"]

# ------------------------------------------------------------
# 6. FIND EMPTY MATCHES
# ------------------------------------------------------------

empty_mask = (
    matches["matched_entity_ids"].isna()
    | (matches["matched_entity_ids"].astype(str).str.strip() == "")
)

empty_before = int(empty_mask.sum())

print(f"\nEmpty matches before fallback: {empty_before:,}")

# ------------------------------------------------------------
# 7. FILL ONLY EMPTY ROWS
# ------------------------------------------------------------

print("\nFilling empty rows using top candidate...")

filled = 0
no_candidate = 0

for idx in matches.index[empty_mask]:

    source_id = matches.at[idx, "source1_entity_id"]

    candidate_string = candidate_map.get(source_id)

    if pd.isna(candidate_string) or not str(candidate_string).strip():
        no_candidate += 1
        continue

    candidate_string = str(candidate_string)

    # Candidate list is comma-separated.
    candidate_list = [
        x.strip()
        for x in candidate_string.split(",")
        if x.strip()
    ]

    if not candidate_list:
        no_candidate += 1
        continue

    # Use ONLY the first/top candidate.
    matches.at[idx, "matched_entity_ids"] = candidate_list[0]

    filled += 1

# ------------------------------------------------------------
# 8. SAVE FALLBACK
# ------------------------------------------------------------

matches.to_csv(
    FALLBACK_FILE,
    sep="\t",
    index=False
)

# ------------------------------------------------------------
# 9. FINAL CHECK
# ------------------------------------------------------------

empty_after = int(
    matches["matched_entity_ids"].isna().sum()
)

print("\n" + "=" * 70)
print("FALLBACK COMPLETE")
print("=" * 70)

print(f"Original empty rows : {empty_before:,}")
print(f"Rows filled         : {filled:,}")
print(f"No candidate found  : {no_candidate:,}")
print(f"Empty rows remaining: {empty_after:,}")

print(f"\nOriginal preserved at:")
print(BACKUP_FILE)

print(f"\nFallback saved at:")
print(FALLBACK_FILE)

print("\nOriginal file was NOT modified.")

print("=" * 70)