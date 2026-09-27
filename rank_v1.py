import os
import glob
import re
import numpy as np
import pandas as pd

from rapidfuzz.fuzz import ratio, token_set_ratio
import lightgbm as lgb


# ============================================================
# CONFIG
# ============================================================

CANDIDATE_FILE = "v6_candidates_improved.csv"
GT_FILE = "dataset/train/train_ground_truth.tsv"

# IMPORTANT:
# First experiment only.
# Your current V7 candidate file has 1000 rows anyway.
MAX_ROWS = 1000

OUTPUT_FILE = "v7_ranked_candidates.tsv"
MODEL_FILE = "v7_lgbm_ranker.txt"


# ============================================================
# HELPERS
# ============================================================

def normalize(x):
    if pd.isna(x):
        return ""

    x = str(x).lower().strip()

    x = re.sub(r"[^a-z0-9\s]", " ", x)
    x = re.sub(r"\s+", " ", x)

    return x


def token_jaccard(a, b):
    sa = set(a.split())
    sb = set(b.split())

    if not sa or not sb:
        return 0.0

    return len(sa & sb) / len(sa | sb)


def exact(a, b):
    return int(a != "" and a == b)


def prefix_match(a, b, n):
    if not a or not b:
        return 0
    return int(a[:n] == b[:n])


def get_id_column(df, prefix):
    for col in df.columns:
        c = col.lower()

        if prefix in c and "id" in c:
            return col

    for col in df.columns:
        if "entity_id" in col.lower():
            return col

    return None


def get_column(df, names):

    lower_map = {
        c.lower(): c
        for c in df.columns
    }

    for name in names:
        if name.lower() in lower_map:
            return lower_map[name.lower()]

    # fuzzy fallback
    for col in df.columns:

        lc = col.lower()

        for name in names:
            if name.lower() in lc:
                return col

    return None


# ============================================================
# FIND SOURCE FILES
# ============================================================

print("=" * 70)
print("V7 + LIGHTGBM RANKER V1")
print("=" * 70)

print("\nSearching dataset files...")

files = glob.glob(
    "dataset/**/*.csv",
    recursive=True
)

files += glob.glob(
    "dataset/**/*.tsv",
    recursive=True
)

files = list(dict.fromkeys(files))

print("Files found:", len(files))


# ============================================================
# LOAD CANDIDATES
# ============================================================

print("\nLoading V7 candidates...")

candidates = pd.read_csv(
    CANDIDATE_FILE,
    sep="\t",
    dtype=str,
    nrows=MAX_ROWS
)

print("Candidate rows:", len(candidates))

print("Candidate columns:")
print(list(candidates.columns))


source1_col = "source1_entity_id"
candidate_col = "candidate_entity_ids"


# ============================================================
# LOAD GROUND TRUTH
# ============================================================

print("\nLoading ground truth...")

gt = pd.read_csv(
    GT_FILE,
    sep="\t",
    dtype=str
)

gt_map = {}

for _, row in gt.iterrows():

    sid = row["source1_entity_id"]

    raw = row["matched_entity_ids"]

    if pd.isna(raw):
        matches = set()
    else:
        matches = {
            x.strip()
            for x in str(raw).split(",")
            if x.strip()
        }

    gt_map[sid] = matches


print("Ground-truth records:", len(gt_map))


# ============================================================
# LOAD SOURCE DATA
# ============================================================

print("\nLocating source records...")

source1 = None
source2 = None
source3 = None


for f in files:

    name = os.path.basename(f).lower()

    try:

        # Skip obvious non-source files
        if any(x in name for x in [
            "ground_truth",
            "candidate",
            "match",
            "submission",
            "result"
        ]):
            continue

        df = pd.read_csv(
            f,
            sep="\t" if f.endswith(".tsv") else ",",
            dtype=str,
            nrows=5
        )

        cols = [c.lower() for c in df.columns]

        # identify by entity ID prefixes / filenames
        if "source1" in name or "source_1" in name:
            source1 = f

        elif "source2" in name or "source_2" in name:
            source2 = f

        elif "source3" in name or "source_3" in name:
            source3 = f

    except Exception:
        continue


print("Detected Source 1:", source1)
print("Detected Source 2:", source2)
print("Detected Source 3:", source3)


# ============================================================
# FALLBACK: SEARCH ALL FILES BY ID PREFIX
# ============================================================

if source2 is None or source3 is None:

    print("\nTrying automatic ID-prefix detection...")

    for f in files:

        if f.endswith(".tsv"):
            sep = "\t"
        else:
            sep = ","

        try:

            sample = pd.read_csv(
                f,
                sep=sep,
                dtype=str,
                nrows=20
            )

            for col in sample.columns:

                vals = sample[col].dropna().astype(str)

                if len(vals) == 0:
                    continue

                s = vals.iloc[0]

                if source2 is None and s.startswith("S2-"):
                    source2 = f

                if source3 is None and s.startswith("S3-"):
                    source3 = f

        except Exception:
            continue


print("Final Source 2:", source2)
print("Final Source 3:", source3)


if source2 is None or source3 is None:

    print("\nERROR: Could not automatically locate Source 2/3.")

    print("\nRun this command and send me the output:")
    print(
        'python -c "import glob; print(glob.glob(\'dataset/**/*\',recursive=True))"'
    )

    raise SystemExit


# ============================================================
# LOAD SOURCE 2
# ============================================================

def load_source(path):

    sep = "\t" if path.endswith(".tsv") else ","

    return pd.read_csv(
        path,
        sep=sep,
        dtype=str
    )


print("\nLoading Source 2...")

s2 = load_source(source2)

print("Source 2 rows:", len(s2))


print("\nLoading Source 3...")

s3 = load_source(source3)

print("Source 3 rows:", len(s3))


# ============================================================
# IDENTIFY COLUMNS
# ============================================================

def prepare_source(df, prefix):

    id_col = get_id_column(
        df,
        prefix
    )

    if id_col is None:

        # find ID by values
        for col in df.columns:

            vals = df[col].dropna().astype(str)

            if len(vals) and vals.iloc[0].startswith(prefix.upper() + "-"):
                id_col = col
                break

    name_col = get_column(
        df,
        [
            "name",
            "entity_name",
            "business_name",
            "company_name"
        ]
    )

    address_col = get_column(
        df,
        [
            "address",
            "entity_address",
            "street",
            "location"
        ]
    )

    country_col = get_column(
        df,
        [
            "country",
            "country_name"
        ]
    )

    print(
        f"\n{prefix} columns:"
    )

    print("ID:", id_col)
    print("Name:", name_col)
    print("Address:", address_col)
    print("Country:", country_col)

    if id_col is None:
        raise ValueError(
            f"Could not find ID column for {prefix}"
        )

    out = pd.DataFrame()

    out["entity_id"] = df[id_col].astype(str)

    if name_col:
        out["name"] = df[name_col].map(normalize)
    else:
        out["name"] = ""

    if address_col:
        out["address"] = df[address_col].map(normalize)
    else:
        out["address"] = ""

    if country_col:
        out["country"] = df[country_col].map(normalize)
    else:
        out["country"] = ""

    return out


s2 = prepare_source(s2, "S2")
s3 = prepare_source(s3, "S3")


# ============================================================
# BUILD LOOKUP
# ============================================================

print("\nBuilding candidate lookup...")

lookup = pd.concat(
    [s2, s3],
    ignore_index=True
)

lookup = lookup.drop_duplicates(
    "entity_id"
)

lookup = lookup.set_index(
    "entity_id"
)

print(
    "Candidate entities indexed:",
    len(lookup)
)


# ============================================================
# SOURCE 1
# ============================================================

# Try to locate Source 1 if needed.

if source1 is not None:

    print("\nLoading Source 1...")

    s1 = load_source(source1)

    s1 = prepare_source(
        s1,
        "S1"
    )

    s1 = s1.set_index(
        "entity_id"
    )

else:

    print(
        "\nWARNING: Source 1 file was not detected."
    )

    print(
        "Trying to find Source 1 using candidate IDs..."
    )

    # Ground truth does not contain names,
    # so Source 1 is required for feature generation.

    raise SystemExit(
        "Source 1 could not be located automatically."
    )


# ============================================================
# BUILD TRAINING DATA
# ============================================================

print("\n" + "=" * 70)
print("BUILDING RANKING FEATURES")
print("=" * 70)


X = []
y = []
groups = []

processed = 0


for _, row in candidates.iterrows():

    sid = str(
        row[source1_col]
    )

    if sid not in s1.index:
        continue

    source = s1.loc[sid]

    raw = row[candidate_col]

    if pd.isna(raw):
        continue

    candidate_ids = [
        x.strip()
        for x in str(raw).split(",")
        if x.strip()
    ]

    truth = gt_map.get(
        sid,
        set()
    )

    row_count = 0

    for cid in candidate_ids:

        if cid not in lookup.index:
            continue

        cand = lookup.loc[cid]

        name1 = source["name"]
        name2 = cand["name"]

        addr1 = source["address"]
        addr2 = cand["address"]

        country1 = source["country"]
        country2 = cand["country"]

        # ----------------------------------------------------
        # FEATURES
        # ----------------------------------------------------

        features = [

            # Name similarity
            ratio(
                name1,
                name2
            ) / 100.0,

            token_set_ratio(
                name1,
                name2
            ) / 100.0,

            token_jaccard(
                name1,
                name2
            ),

            exact(
                name1,
                name2
            ),

            prefix_match(
                name1,
                name2,
                3
            ),

            prefix_match(
                name1,
                name2,
                5
            ),

            # Address similarity
            ratio(
                addr1,
                addr2
            ) / 100.0,

            token_set_ratio(
                addr1,
                addr2
            ) / 100.0,

            token_jaccard(
                addr1,
                addr2
            ),

            exact(
                addr1,
                addr2
            ),

            prefix_match(
                addr1,
                addr2,
                5
            ),

            # Country
            exact(
                country1,
                country2
            ),

            # Length differences
            abs(
                len(name1) -
                len(name2)
            ),

            abs(
                len(addr1) -
                len(addr2)
            )
        ]

        X.append(features)

        y.append(
            1 if cid in truth else 0
        )

        row_count += 1

    if row_count > 0:
        groups.append(row_count)

    processed += 1

    if processed % 100 == 0:
        print(
            f"Processed {processed}/{len(candidates)}"
        )


X = np.asarray(
    X,
    dtype=np.float32
)

y = np.asarray(
    y,
    dtype=np.int8
)

groups = np.asarray(
    groups,
    dtype=np.int32
)


print("\nTraining rows:", len(X))
print("Positive matches:", int(y.sum()))
print("Groups:", len(groups))


if len(X) == 0:
    raise SystemExit(
        "No training rows generated."
    )


# ============================================================
# LIGHTGBM RANKER
# ============================================================

print("\n" + "=" * 70)
print("TRAINING LIGHTGBM RANKER")
print("=" * 70)


model = lgb.LGBMRanker(

    objective="lambdarank",

    metric="ndcg",

    n_estimators=250,

    learning_rate=0.05,

    num_leaves=31,

    max_depth=-1,

    min_child_samples=20,

    subsample=0.9,

    colsample_bytree=0.9,

    random_state=42,

    verbosity=-1
)


model.fit(
    X,
    y,
    group=groups
)


model.booster_.save_model(
    MODEL_FILE
)

print(
    "\nModel saved:",
    MODEL_FILE
)


# ============================================================
# RANK CANDIDATES
# ============================================================

print("\n" + "=" * 70)
print("RANKING CANDIDATES")
print("=" * 70)


results = []

row_index = 0


for _, row in candidates.iterrows():

    sid = str(
        row[source1_col]
    )

    if sid not in s1.index:
        continue

    source = s1.loc[sid]

    raw = row[candidate_col]

    if pd.isna(raw):
        continue

    candidate_ids = [
        x.strip()
        for x in str(raw).split(",")
        if x.strip()
    ]

    feature_rows = []
    valid_ids = []

    for cid in candidate_ids:

        if cid not in lookup.index:
            continue

        cand = lookup.loc[cid]

        name1 = source["name"]
        name2 = cand["name"]

        addr1 = source["address"]
        addr2 = cand["address"]

        country1 = source["country"]
        country2 = cand["country"]

        feature_rows.append([

            ratio(name1, name2) / 100.0,

            token_set_ratio(
                name1,
                name2
            ) / 100.0,

            token_jaccard(
                name1,
                name2
            ),

            exact(
                name1,
                name2
            ),

            prefix_match(
                name1,
                name2,
                3
            ),

            prefix_match(
                name1,
                name2,
                5
            ),

            ratio(
                addr1,
                addr2
            ) / 100.0,

            token_set_ratio(
                addr1,
                addr2
            ) / 100.0,

            token_jaccard(
                addr1,
                addr2
            ),

            exact(
                addr1,
                addr2
            ),

            prefix_match(
                addr1,
                addr2,
                5
            ),

            exact(
                country1,
                country2
            ),

            abs(
                len(name1) -
                len(name2)
            ),

            abs(
                len(addr1) -
                len(addr2)
            )

        ])

        valid_ids.append(cid)

    if not feature_rows:
        continue

    scores = model.predict(
        np.asarray(
            feature_rows,
            dtype=np.float32
        )
    )

    order = np.argsort(
        -scores
    )

    ranked_ids = [
        valid_ids[i]
        for i in order
    ]

    results.append({

        "source1_entity_id":
            sid,

        "candidate_entity_ids":
            ",".join(ranked_ids)

    })


# ============================================================
# SAVE
# ============================================================

out = pd.DataFrame(
    results
)

out.to_csv(
    OUTPUT_FILE,
    sep="\t",
    index=False
)


print("\n" + "=" * 70)
print("RANKING COMPLETE")
print("=" * 70)

print(
    "Rows:",
    len(out)
)

print(
    "Saved:",
    OUTPUT_FILE
)

print(
    "\nNext: evaluate the ranked candidates."
)