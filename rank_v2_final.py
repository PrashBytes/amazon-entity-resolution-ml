import os
import glob
import re
import warnings
import numpy as np
import pandas as pd

from rapidfuzz.fuzz import ratio, token_set_ratio
import lightgbm as lgb

warnings.filterwarnings("ignore")


# ============================================================
# CONFIG
# ============================================================

# IMPORTANT:
# This should be your FULL V7 candidate file when you generate it.
#
# For your current testing file, keep:
CANDIDATE_FILE = "v6_candidates_improved.csv"

GT_FILE = "dataset/train/train_ground_truth.tsv"

OUTPUT_FILE = "v7_ranked_final.tsv"
MODEL_FILE = "v7_lgbm_final.txt"

# Number of candidate rows used to TRAIN the ranker.
#
# 1000 is enough for our current experiment.
# The model will then be used to rank ALL candidate rows.
TRAIN_ROWS = 1000

# Set to None when using the FULL candidate file.
#
# IMPORTANT:
# This is different from TRAIN_ROWS.
#
# TRAIN_ROWS = how many rows are used for training.
# MAX_RANK_ROWS = how many rows are actually ranked.
#
# Current testing:
MAX_RANK_ROWS = 1000

# Final candidate list size.
TOP_K = 100


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

    return int(
        a != "" and
        a == b
    )


def prefix_match(a, b, n):

    if not a or not b:
        return 0

    return int(
        a[:n] == b[:n]
    )


def suffix_match(a, b, n):

    if not a or not b:
        return 0

    return int(
        a[-n:] == b[-n:]
    )


def first_word(a):

    if not a:
        return ""

    return a.split()[0]


def last_word(a):

    if not a:
        return ""

    return a.split()[-1]


def address_number(a):

    if not a:
        return ""

    m = re.search(r"\b\d+\b", a)

    if m:
        return m.group(0)

    return ""


def get_id_column(df, prefix):

    for col in df.columns:

        c = col.lower()

        if prefix.lower() in c and "id" in c:
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
# FEATURE BUILDER
# ============================================================

def build_features(source, cand):

    name1 = source["name"]
    name2 = cand["name"]

    addr1 = source["address"]
    addr2 = cand["address"]

    country1 = source["country"]
    country2 = cand["country"]

    # --------------------------------------------------------
    # NAME FEATURES
    # --------------------------------------------------------

    name_ratio = ratio(
        name1,
        name2
    ) / 100.0

    name_token_ratio = token_set_ratio(
        name1,
        name2
    ) / 100.0

    name_jaccard = token_jaccard(
        name1,
        name2
    )

    name_exact = exact(
        name1,
        name2
    )

    name_prefix3 = prefix_match(
        name1,
        name2,
        3
    )

    name_prefix5 = prefix_match(
        name1,
        name2,
        5
    )

    name_suffix3 = suffix_match(
        name1,
        name2,
        3
    )

    name_first_word = exact(
        first_word(name1),
        first_word(name2)
    )

    name_last_word = exact(
        last_word(name1),
        last_word(name2)
    )

    # --------------------------------------------------------
    # ADDRESS FEATURES
    # --------------------------------------------------------

    address_ratio = ratio(
        addr1,
        addr2
    ) / 100.0

    address_token_ratio = token_set_ratio(
        addr1,
        addr2
    ) / 100.0

    address_jaccard = token_jaccard(
        addr1,
        addr2
    )

    address_exact = exact(
        addr1,
        addr2
    )

    address_prefix5 = prefix_match(
        addr1,
        addr2,
        5
    )

    address_suffix5 = suffix_match(
        addr1,
        addr2,
        5
    )

    # --------------------------------------------------------
    # ADDRESS NUMBER
    # --------------------------------------------------------

    num1 = address_number(addr1)
    num2 = address_number(addr2)

    address_number_match = int(
        num1 != "" and
        num1 == num2
    )

    # --------------------------------------------------------
    # COUNTRY
    # --------------------------------------------------------

    country_exact = exact(
        country1,
        country2
    )

    # --------------------------------------------------------
    # LENGTH FEATURES
    # --------------------------------------------------------

    name_length_diff = abs(
        len(name1) -
        len(name2)
    )

    address_length_diff = abs(
        len(addr1) -
        len(addr2)
    )

    # --------------------------------------------------------
    # TOKEN COUNTS
    # --------------------------------------------------------

    name_token_count_diff = abs(
        len(name1.split()) -
        len(name2.split())
    )

    address_token_count_diff = abs(
        len(addr1.split()) -
        len(addr2.split())
    )

    return [

        # NAME
        name_ratio,
        name_token_ratio,
        name_jaccard,
        name_exact,
        name_prefix3,
        name_prefix5,
        name_suffix3,
        name_first_word,
        name_last_word,

        # ADDRESS
        address_ratio,
        address_token_ratio,
        address_jaccard,
        address_exact,
        address_prefix5,
        address_suffix5,
        address_number_match,

        # COUNTRY
        country_exact,

        # LENGTH
        name_length_diff,
        address_length_diff,

        # TOKEN COUNTS
        name_token_count_diff,
        address_token_count_diff
    ]


FEATURE_NAMES = [

    "name_ratio",
    "name_token_ratio",
    "name_jaccard",
    "name_exact",
    "name_prefix3",
    "name_prefix5",
    "name_suffix3",
    "name_first_word",
    "name_last_word",

    "address_ratio",
    "address_token_ratio",
    "address_jaccard",
    "address_exact",
    "address_prefix5",
    "address_suffix5",
    "address_number_match",

    "country_exact",

    "name_length_diff",
    "address_length_diff",

    "name_token_count_diff",
    "address_token_count_diff"
]


# ============================================================
# START
# ============================================================

print("=" * 70)
print("V7 + LIGHTGBM RANKER V2 FINAL")
print("=" * 70)

print("\nCandidate file:", CANDIDATE_FILE)

print("Training rows:", TRAIN_ROWS)

print("Ranking rows:", MAX_RANK_ROWS)

print("Top-K:", TOP_K)


# ============================================================
# LOAD CANDIDATES
# ============================================================

print("\nLoading candidate file...")

candidates = pd.read_csv(
    CANDIDATE_FILE,
    sep="\t",
    dtype=str
)

print(
    "Total candidate rows:",
    len(candidates)
)

print(
    "Candidate columns:",
    list(candidates.columns)
)


source1_col = "source1_entity_id"
candidate_col = "candidate_entity_ids"


if source1_col not in candidates.columns:

    raise SystemExit(
        f"Missing column: {source1_col}"
    )


if candidate_col not in candidates.columns:

    raise SystemExit(
        f"Missing column: {candidate_col}"
    )


# ============================================================
# LIMIT RANKING ROWS
# ============================================================

if MAX_RANK_ROWS is None:

    ranking_candidates = candidates

else:

    ranking_candidates = candidates.head(
        MAX_RANK_ROWS
    ).copy()


# ============================================================
# TRAINING DATA
# ============================================================

training_candidates = candidates.head(
    TRAIN_ROWS
).copy()


print(
    "\nRows used for training:",
    len(training_candidates)
)

print(
    "Rows used for ranking:",
    len(ranking_candidates)
)


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

    sid = str(
        row["source1_entity_id"]
    )

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


print(
    "Ground-truth records:",
    len(gt_map)
)


# ============================================================
# FIND SOURCE FILES
# ============================================================

print("\nSearching dataset files...")

files = glob.glob(
    "dataset/**/*.csv",
    recursive=True
)

files += glob.glob(
    "dataset/**/*.tsv",
    recursive=True
)

files = list(
    dict.fromkeys(files)
)


source1 = None
source2 = None
source3 = None


for f in files:

    name = os.path.basename(
        f
    ).lower()

    try:

        if any(
            x in name
            for x in [
                "ground_truth",
                "candidate",
                "match",
                "submission",
                "result"
            ]
        ):
            continue

        if "source1" in name or "source_1" in name:

            source1 = f

        elif "source2" in name or "source_2" in name:

            source2 = f

        elif "source3" in name or "source_3" in name:

            source3 = f

    except Exception:

        continue


# ============================================================
# FALLBACK SOURCE DETECTION
# ============================================================

if source2 is None or source3 is None:

    for f in files:

        sep = (
            "\t"
            if f.endswith(".tsv")
            else ","
        )

        try:

            sample = pd.read_csv(
                f,
                sep=sep,
                dtype=str,
                nrows=20
            )

            for col in sample.columns:

                vals = (
                    sample[col]
                    .dropna()
                    .astype(str)
                )

                if len(vals) == 0:
                    continue

                s = vals.iloc[0]

                if (
                    source2 is None
                    and s.startswith("S2-")
                ):

                    source2 = f

                if (
                    source3 is None
                    and s.startswith("S3-")
                ):

                    source3 = f

        except Exception:

            continue


print("\nSource 1:", source1)

print("Source 2:", source2)

print("Source 3:", source3)


if source1 is None:

    raise SystemExit(
        "Source 1 could not be located."
    )


if source2 is None:

    raise SystemExit(
        "Source 2 could not be located."
    )


if source3 is None:

    raise SystemExit(
        "Source 3 could not be located."
    )


# ============================================================
# LOAD SOURCE
# ============================================================

def load_source(path):

    sep = (
        "\t"
        if path.endswith(".tsv")
        else ","
    )

    return pd.read_csv(
        path,
        sep=sep,
        dtype=str
    )


def prepare_source(df, prefix):

    id_col = get_id_column(
        df,
        prefix
    )

    if id_col is None:

        for col in df.columns:

            vals = (
                df[col]
                .dropna()
                .astype(str)
            )

            if (
                len(vals)
                and
                vals.iloc[0].startswith(
                    prefix.upper() + "-"
                )
            ):

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

    print(
        "ID:",
        id_col
    )

    print(
        "Name:",
        name_col
    )

    print(
        "Address:",
        address_col
    )

    print(
        "Country:",
        country_col
    )


    if id_col is None:

        raise ValueError(
            f"Could not find ID column for {prefix}"
        )


    out = pd.DataFrame()

    out["entity_id"] = (
        df[id_col]
        .astype(str)
    )


    if name_col:

        out["name"] = (
            df[name_col]
            .map(normalize)
        )

    else:

        out["name"] = ""


    if address_col:

        out["address"] = (
            df[address_col]
            .map(normalize)
        )

    else:

        out["address"] = ""


    if country_col:

        out["country"] = (
            df[country_col]
            .map(normalize)
        )

    else:

        out["country"] = ""


    return out


# ============================================================
# LOAD SOURCE 2
# ============================================================

print("\nLoading Source 2...")

s2 = load_source(
    source2
)

print(
    "Source 2 rows:",
    len(s2)
)

s2 = prepare_source(
    s2,
    "S2"
)


# ============================================================
# LOAD SOURCE 3
# ============================================================

print("\nLoading Source 3...")

s3 = load_source(
    source3
)

print(
    "Source 3 rows:",
    len(s3)
)

s3 = prepare_source(
    s3,
    "S3"
)


# ============================================================
# CANDIDATE LOOKUP
# ============================================================

print("\nBuilding candidate lookup...")

lookup = pd.concat(
    [
        s2,
        s3
    ],
    ignore_index=True
)


lookup = lookup.drop_duplicates(
    "entity_id"
)


lookup = lookup.set_index(
    "entity_id"
)


print(
    "Candidate entities:",
    len(lookup)
)


# ============================================================
# LOAD SOURCE 1
# ============================================================

print("\nLoading Source 1...")

s1 = load_source(
    source1
)

print(
    "Source 1 rows:",
    len(s1)
)

s1 = prepare_source(
    s1,
    "S1"
)

s1 = s1.set_index(
    "entity_id"
)


# ============================================================
# BUILD TRAINING DATA
# ============================================================

print("\n" + "=" * 70)
print("BUILDING TRAINING DATA")
print("=" * 70)


X = []
y = []
groups = []


processed = 0


for _, row in training_candidates.iterrows():

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


        features = build_features(
            source,
            cand
        )


        X.append(
            features
        )


        y.append(
            1
            if cid in truth
            else 0
        )


        row_count += 1


    if row_count > 0:

        groups.append(
            row_count
        )


    processed += 1


    if processed % 100 == 0:

        print(
            f"Training rows processed: "
            f"{processed}/{len(training_candidates)}"
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


print(
    "\nTraining examples:",
    len(X)
)

print(
    "Positive matches:",
    int(y.sum())
)

print(
    "Training groups:",
    len(groups)
)


if len(X) == 0:

    raise SystemExit(
        "No training examples generated."
    )


if y.sum() == 0:

    raise SystemExit(
        "No positive matches found."
    )


# ============================================================
# TRAIN LIGHTGBM
# ============================================================

print("\n" + "=" * 70)
print("TRAINING LIGHTGBM")
print("=" * 70)


model = lgb.LGBMRanker(

    objective="lambdarank",

    metric="ndcg",

    n_estimators=350,

    learning_rate=0.04,

    num_leaves=31,

    max_depth=-1,

    min_child_samples=20,

    subsample=0.9,

    colsample_bytree=0.9,

    reg_alpha=0.1,

    reg_lambda=0.1,

    random_state=42,

    verbosity=-1
)


model.fit(

    X,

    y,

    group=groups,

    feature_name=FEATURE_NAMES
)


model.booster_.save_model(
    MODEL_FILE
)


print(
    "\nModel saved:",
    MODEL_FILE
)


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

print("\n" + "=" * 70)
print("FEATURE IMPORTANCE")
print("=" * 70)


importance = model.booster_.feature_importance(
    importance_type="gain"
)


feature_importance = sorted(
    zip(
        FEATURE_NAMES,
        importance
    ),
    key=lambda x: x[1],
    reverse=True
)


for name, value in feature_importance:

    print(
        f"{name:30s} {value:.2f}"
    )


# ============================================================
# RANK CANDIDATES
# ============================================================

print("\n" + "=" * 70)
print("RANKING CANDIDATES")
print("=" * 70)


results = []

processed = 0


for _, row in ranking_candidates.iterrows():

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


        feature_rows.append(
            build_features(
                source,
                cand
            )
        )


        valid_ids.append(
            cid
        )


    if not feature_rows:

        continue


    features_array = np.asarray(
        feature_rows,
        dtype=np.float32
    )


    scores = model.predict(
        features_array
    )


    order = np.argsort(
        -scores
    )


    ranked_ids = [

        valid_ids[i]

        for i in order[:TOP_K]

    ]


    results.append({

        "source1_entity_id":
            sid,

        "candidate_entity_ids":
            ",".join(ranked_ids)

    })


    processed += 1


    if processed % 100 == 0:

        print(
            f"Ranked rows: "
            f"{processed}/{len(ranking_candidates)}"
        )


# ============================================================
# SAVE
# ============================================================

print("\n" + "=" * 70)
print("SAVING RESULTS")
print("=" * 70)


out = pd.DataFrame(
    results
)


out.to_csv(
    OUTPUT_FILE,
    sep="\t",
    index=False
)


print(
    "\nRows:",
    len(out)
)


print(
    "Saved:",
    OUTPUT_FILE
)


print("\n" + "=" * 70)
print("RANKING COMPLETE")
print("=" * 70)