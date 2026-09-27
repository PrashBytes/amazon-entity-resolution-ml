import pandas as pd
import re
import unicodedata
import time

from collections import defaultdict, Counter

from rapidfuzz.fuzz import ratio, token_set_ratio

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from scipy.sparse import csr_matrix


# ============================================================
# CONFIGURATION
# ============================================================

TEST_ROWS = 1000

# Number of final candidates retained per Source-1 row.
TOP_CANDIDATES = 100

# Number of TF-IDF candidates retrieved.
TFIDF_TOP_K = 50

# Number of candidates taken from each blocking method.
MAX_FROM_BLOCK = 300

# Ignore extremely common tokens.
MAX_TOKEN_FREQUENCY = 5000

OUTPUT_FILE = "v8_candidates_tfidf.tsv"


# ============================================================
# NORMALIZATION
# ============================================================

def normalize_text(text):

    if pd.isna(text):
        return ""

    text = unicodedata.normalize(
        "NFKD",
        str(text)
    )

    text = (
        text
        .encode("ascii", "ignore")
        .decode("ascii")
    )

    text = text.lower()

    text = re.sub(
        r"[^a-z0-9\s]",
        " ",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    return text


def tokenize(text):

    if not text:
        return []

    return text.split()


def get_first_word(text):

    tokens = tokenize(text)

    if tokens:
        return tokens[0]

    return ""


def get_address_number(text):

    if not text:
        return ""

    match = re.search(
        r"\b\d{1,6}\b",
        text
    )

    if match:
        return match.group(0)

    return ""


def sorted_token_key(text):

    tokens = tokenize(text)

    if not tokens:
        return ""

    return " ".join(
        sorted(set(tokens))
    )


# ============================================================
# START
# ============================================================

print("=" * 70)
print("V8 FAST TF-IDF + V7 HIGH-RECALL CANDIDATE GENERATION")
print("=" * 70)

start_total = time.time()


# ============================================================
# LOAD DATA
# ============================================================

print("\nLoading datasets...")

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
# TEST MODE
# ============================================================

if TEST_ROWS is not None:

    s1 = s1.head(TEST_ROWS).copy()

    print()
    print("=" * 70)
    print(
        f"TEST MODE: {TEST_ROWS} Source-1 rows"
    )
    print("=" * 70)

else:

    print()
    print("=" * 70)
    print("FULL MODE")
    print("=" * 70)


# ============================================================
# NORMALIZATION
# ============================================================

print("\nNormalizing...")

norm_start = time.time()


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

    df["name_tokens"] = (
        df["name_norm"]
        .apply(tokenize)
    )

    df["address_tokens"] = (
        df["address_norm"]
        .apply(tokenize)
    )

    df["name_sorted"] = (
        df["name_norm"]
        .apply(sorted_token_key)
    )

    df["first_word"] = (
        df["name_norm"]
        .apply(get_first_word)
    )

    df["address_number"] = (
        df["address_norm"]
        .apply(get_address_number)
    )


print(
    f"Normalization time: "
    f"{time.time() - norm_start:.1f}s"
)


# ============================================================
# COMBINE SOURCES 2 + 3
# ============================================================

print("\nPreparing candidate data...")

s2["candidate_source"] = "source2"
s3["candidate_source"] = "source3"

candidates = pd.concat(
    [s2, s3],
    ignore_index=True
)

print(
    "Total candidate records:",
    len(candidates)
)


# ============================================================
# FAST INDEX BUILDING
# ============================================================

print("\nBuilding blocking indexes...")

index_start = time.time()


# ------------------------------------------------------------
# EXACT NAME
# ------------------------------------------------------------

exact_name_index = defaultdict(list)

for idx, value in enumerate(
    candidates["name_norm"].values
):

    if value:
        exact_name_index[value].append(idx)


# ------------------------------------------------------------
# SORTED NAME
# ------------------------------------------------------------

sorted_name_index = defaultdict(list)

for idx, value in enumerate(
    candidates["name_sorted"].values
):

    if value:
        sorted_name_index[value].append(idx)


# ------------------------------------------------------------
# FIRST WORD
# ------------------------------------------------------------

first_word_index = defaultdict(list)

for idx, value in enumerate(
    candidates["first_word"].values
):

    if value:
        first_word_index[value].append(idx)


# ------------------------------------------------------------
# ADDRESS NUMBER
# ------------------------------------------------------------

address_number_index = defaultdict(list)

for idx, value in enumerate(
    candidates["address_number"].values
):

    if value:
        address_number_index[value].append(idx)


# ------------------------------------------------------------
# COUNTRY + FIRST WORD
# ------------------------------------------------------------

country_first_word_index = defaultdict(list)

for idx in range(len(candidates)):

    country = candidates.iloc[idx][
        "country_norm"
    ]

    first_word = candidates.iloc[idx][
        "first_word"
    ]

    if country and first_word:

        key = (
            country
            + "|"
            + first_word
        )

        country_first_word_index[
            key
        ].append(idx)


# ------------------------------------------------------------
# COUNTRY + PREFIX
# ------------------------------------------------------------

country_prefix_index = defaultdict(list)

for idx in range(len(candidates)):

    country = candidates.iloc[idx][
        "country_norm"
    ]

    name = candidates.iloc[idx][
        "name_norm"
    ]

    if not country or not name:
        continue

    prefixes = set()

    if len(name) >= 3:
        prefixes.add(name[:3])

    if len(name) >= 5:
        prefixes.add(name[:5])

    for prefix in prefixes:

        key = (
            country
            + "|"
            + prefix
        )

        country_prefix_index[
            key
        ].append(idx)


print(
    "Exact name keys:",
    len(exact_name_index)
)

print(
    "Sorted name keys:",
    len(sorted_name_index)
)

print(
    "First word keys:",
    len(first_word_index)
)

print(
    "Address number keys:",
    len(address_number_index)
)

print(
    "Country+first word keys:",
    len(country_first_word_index)
)

print(
    "Country+prefix keys:",
    len(country_prefix_index)
)

print(
    f"Blocking index time: "
    f"{time.time() - index_start:.1f}s"
)


# ============================================================
# TOKEN FREQUENCY
# ============================================================

print("\nBuilding token indexes...")

token_start = time.time()

name_frequency = Counter()
address_frequency = Counter()


for tokens in candidates["name_tokens"]:

    for token in set(tokens):

        if len(token) >= 2:
            name_frequency[token] += 1


for tokens in candidates["address_tokens"]:

    for token in set(tokens):

        if len(token) >= 2:
            address_frequency[token] += 1


# ------------------------------------------------------------
# NAME TOKEN INDEX
# ------------------------------------------------------------

name_token_index = defaultdict(list)

for idx, tokens in enumerate(
    candidates["name_tokens"]
):

    for token in set(tokens):

        if len(token) < 2:
            continue

        if (
            name_frequency[token]
            <= MAX_TOKEN_FREQUENCY
        ):

            name_token_index[
                token
            ].append(idx)


# ------------------------------------------------------------
# ADDRESS TOKEN INDEX
# ------------------------------------------------------------

address_token_index = defaultdict(list)

for idx, tokens in enumerate(
    candidates["address_tokens"]
):

    for token in set(tokens):

        if len(token) < 2:
            continue

        if (
            address_frequency[token]
            <= MAX_TOKEN_FREQUENCY
        ):

            address_token_index[
                token
            ].append(idx)


print(
    "Name token keys:",
    len(name_token_index)
)

print(
    "Address token keys:",
    len(address_token_index)
)

print(
    f"Token index time: "
    f"{time.time() - token_start:.1f}s"
)


# ============================================================
# TF-IDF INDEX
# ============================================================

print()
print("=" * 70)
print("BUILDING CHARACTER TF-IDF INDEX")
print("=" * 70)

tfidf_start = time.time()


# Character n-grams are useful for:
#
# McDonald's
# McDonalds
#
# Restaurant
# Restarant
#
# ABC Pvt Ltd
# ABC Private Limited
#
# and other small spelling / formatting changes.

vectorizer = TfidfVectorizer(
    analyzer="char_wb",
    ngram_range=(2, 5),
    min_df=2,
    max_df=0.98,
    sublinear_tf=True,
    dtype="float32"
)


candidate_names = (
    candidates["name_norm"]
    .fillna("")
    .values
)


candidate_matrix = vectorizer.fit_transform(
    candidate_names
)


# Normalize rows so dot product becomes cosine similarity.

candidate_matrix = normalize(
    candidate_matrix,
    norm="l2",
    axis=1,
    copy=False
)


print(
    "TF-IDF matrix shape:",
    candidate_matrix.shape
)

print(
    "TF-IDF features:",
    len(vectorizer.vocabulary_)
)

print(
    f"TF-IDF build time: "
    f"{time.time() - tfidf_start:.1f}s"
)


# ============================================================
# BLOCK HELPER
# ============================================================

def add_block(
    target_set,
    values,
    max_items=MAX_FROM_BLOCK
):

    if not values:
        return

    if len(values) <= max_items:

        target_set.update(values)

    else:

        target_set.update(
            values[:max_items]
        )


# ============================================================
# TF-IDF RETRIEVAL
# ============================================================

def tfidf_candidates(
    source_name
):

    if not source_name:
        return set()

    query_vector = vectorizer.transform(
        [source_name]
    )

    query_vector = normalize(
        query_vector,
        norm="l2",
        axis=1,
        copy=False
    )


    # Sparse matrix multiplication.
    #
    # This is much faster than calculating
    # RapidFuzz against every candidate.

    scores = (
        candidate_matrix
        @
        query_vector.T
    )


    scores = scores.toarray().ravel()


    # Get only the highest-scoring candidates.

    if len(scores) <= TFIDF_TOP_K:

        top_indices = scores.argsort()[
            ::-1
        ]

    else:

        # Partial sort first.
        # Much faster than sorting all 10M records.

        partition_indices = (
            scores.argpartition(
                -TFIDF_TOP_K
            )[-TFIDF_TOP_K:]
        )

        top_indices = (
            partition_indices[
                scores[
                    partition_indices
                ].argsort()[::-1]
            ]
        )


    return set(
        int(x)
        for x in top_indices
    )


# ============================================================
# V7-STYLE BLOCKING
# ============================================================

def collect_block_candidates(row):

    candidate_indices = set()

    source_name = row["name_norm"]
    source_country = row["country_norm"]
    source_tokens = row["name_tokens"]
    source_address_tokens = row["address_tokens"]
    source_first_word = row["first_word"]
    source_address_number = row["address_number"]
    source_sorted_name = row["name_sorted"]


    # --------------------------------------------------------
    # EXACT NAME
    # --------------------------------------------------------

    if source_name:

        add_block(
            candidate_indices,
            exact_name_index.get(
                source_name,
                []
            )
        )


    # --------------------------------------------------------
    # SORTED TOKEN NAME
    # --------------------------------------------------------

    if source_sorted_name:

        add_block(
            candidate_indices,
            sorted_name_index.get(
                source_sorted_name,
                []
            )
        )


    # --------------------------------------------------------
    # COUNTRY + FIRST WORD
    # --------------------------------------------------------

    if (
        source_country
        and source_first_word
    ):

        key = (
            source_country
            + "|"
            + source_first_word
        )

        add_block(
            candidate_indices,
            country_first_word_index.get(
                key,
                []
            )
        )


    # --------------------------------------------------------
    # COUNTRY + PREFIX
    # --------------------------------------------------------

    if (
        source_country
        and source_name
    ):

        prefixes = set()

        if len(source_name) >= 3:
            prefixes.add(
                source_name[:3]
            )

        if len(source_name) >= 5:
            prefixes.add(
                source_name[:5]
            )

        for prefix in prefixes:

            key = (
                source_country
                + "|"
                + prefix
            )

            add_block(
                candidate_indices,
                country_prefix_index.get(
                    key,
                    []
                )
            )


    # --------------------------------------------------------
    # RARE NAME TOKENS
    # --------------------------------------------------------

    useful_tokens = []

    for token in set(
        source_tokens
    ):

        if len(token) < 2:
            continue

        frequency = name_frequency.get(
            token,
            0
        )

        if (
            frequency > 0
            and frequency <= MAX_TOKEN_FREQUENCY
        ):

            useful_tokens.append(
                (
                    frequency,
                    token
                )
            )


    # Rarest first.

    useful_tokens.sort(
        key=lambda x: x[0]
    )


    for _, token in useful_tokens[:5]:

        add_block(
            candidate_indices,
            name_token_index.get(
                token,
                []
            )
        )


    # --------------------------------------------------------
    # ADDRESS NUMBER
    # --------------------------------------------------------

    if source_address_number:

        add_block(
            candidate_indices,
            address_number_index.get(
                source_address_number,
                []
            )
        )


    # --------------------------------------------------------
    # ADDRESS TOKENS
    # --------------------------------------------------------

    useful_address_tokens = []

    for token in set(
        source_address_tokens
    ):

        if len(token) < 2:
            continue

        frequency = address_frequency.get(
            token,
            0
        )

        if (
            frequency > 0
            and frequency <= MAX_TOKEN_FREQUENCY
        ):

            useful_address_tokens.append(
                (
                    frequency,
                    token
                )
            )


    useful_address_tokens.sort(
        key=lambda x: x[0]
    )


    for _, token in useful_address_tokens[:5]:

        add_block(
            candidate_indices,
            address_token_index.get(
                token,
                []
            )
        )


    return candidate_indices


# ============================================================
# RANKING
# ============================================================

def rank_candidates(
    row,
    candidate_indices
):

    results = []

    source_name = row["name_norm"]
    source_address = row["address_norm"]
    source_country = row["country_norm"]

    source_name_tokens = set(
        row["name_tokens"]
    )

    source_address_tokens = set(
        row["address_tokens"]
    )


    for idx in candidate_indices:

        candidate = candidates.iloc[
            idx
        ]

        candidate_name = candidate[
            "name_norm"
        ]

        candidate_address = candidate[
            "address_norm"
        ]

        candidate_country = candidate[
            "country_norm"
        ]


        # ----------------------------------------------------
        # NAME
        # ----------------------------------------------------

        name_ratio = ratio(
            source_name,
            candidate_name
        )

        name_token_score = token_set_ratio(
            source_name,
            candidate_name
        )


        # ----------------------------------------------------
        # ADDRESS
        # ----------------------------------------------------

        address_ratio = ratio(
            source_address,
            candidate_address
        )

        address_token_score = token_set_ratio(
            source_address,
            candidate_address
        )


        # ----------------------------------------------------
        # NAME TOKEN OVERLAP
        # ----------------------------------------------------

        candidate_name_tokens = set(
            candidate["name_tokens"]
        )

        if source_name_tokens:

            name_overlap = (
                len(
                    source_name_tokens
                    &
                    candidate_name_tokens
                )
                /
                max(
                    1,
                    len(source_name_tokens)
                )
            )

        else:

            name_overlap = 0


        # ----------------------------------------------------
        # ADDRESS TOKEN OVERLAP
        # ----------------------------------------------------

        candidate_address_tokens = set(
            candidate["address_tokens"]
        )

        if source_address_tokens:

            address_overlap = (
                len(
                    source_address_tokens
                    &
                    candidate_address_tokens
                )
                /
                max(
                    1,
                    len(source_address_tokens)
                )
            )

        else:

            address_overlap = 0


        # ----------------------------------------------------
        # COUNTRY BONUS
        # ----------------------------------------------------

        country_bonus = 0

        if (
            source_country
            and candidate_country
            and source_country
            ==
            candidate_country
        ):

            country_bonus = 5


        # ----------------------------------------------------
        # FINAL SCORE
        # ----------------------------------------------------

        final_score = (

            0.32 * name_ratio

            + 0.25 * name_token_score

            + 0.16 * address_ratio

            + 0.10 * address_token_score

            + 5.0 * name_overlap

            + 3.0 * address_overlap

            + country_bonus

        )


        results.append(
            (
                idx,
                final_score
            )
        )


    results.sort(
        key=lambda x: x[1],
        reverse=True
    )

    return results


# ============================================================
# PROCESS SOURCE 1
# ============================================================

print()
print("=" * 70)
print("GENERATING V8 CANDIDATES")
print("=" * 70)

process_start = time.time()

all_rows = []


for i, (_, row) in enumerate(
    s1.iterrows()
):

    # --------------------------------------------------------
    # V7 BLOCKING
    # --------------------------------------------------------

    block_candidates = (
        collect_block_candidates(
            row
        )
    )


    # --------------------------------------------------------
    # TF-IDF RETRIEVAL
    # --------------------------------------------------------

    fuzzy_candidates = (
        tfidf_candidates(
            row["name_norm"]
        )
    )


    # --------------------------------------------------------
    # UNION
    # --------------------------------------------------------

    candidate_indices = (
        block_candidates
        |
        fuzzy_candidates
    )


    # --------------------------------------------------------
    # RANK
    # --------------------------------------------------------

    ranked = rank_candidates(
        row,
        candidate_indices
    )


    # --------------------------------------------------------
    # TOP 100
    # --------------------------------------------------------

    ranked = ranked[
        :TOP_CANDIDATES
    ]


    # --------------------------------------------------------
    # ENTITY IDS
    # --------------------------------------------------------

    entity_ids = []

    for idx, score in ranked:

        entity_id = candidates.iloc[
            idx
        ]["entity_id"]

        entity_ids.append(
            str(entity_id)
        )


    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    all_rows.append({

        "source1_entity_id":
            str(row["entity_id"]),

        "candidate_entity_ids":
            ",".join(entity_ids)

    })


    # --------------------------------------------------------
    # PROGRESS
    # --------------------------------------------------------

    if (
        (i + 1) % 100
        == 0
    ):

        elapsed = (
            time.time()
            -
            process_start
        )

        rate = (
            (i + 1)
            /
            max(
                elapsed,
                0.001
            )
        )

        remaining = (
            len(s1)
            -
            (i + 1)
        )

        eta = (
            remaining
            /
            max(
                rate,
                0.001
            )
        )

        print(
            f"Processed "
            f"{i + 1}/"
            f"{len(s1)} "
            f"| "
            f"{rate:.1f} rows/sec "
            f"| ETA "
            f"{eta:.1f}s"
        )


# ============================================================
# SAVE
# ============================================================

results = pd.DataFrame(
    all_rows
)


results.to_csv(
    OUTPUT_FILE,
    sep="\t",
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

candidate_counts = (
    results[
        "candidate_entity_ids"
    ]
    .apply(
        lambda x:
        len(
            x.split(",")
        )
        if x
        else 0
    )
)


print()
print("=" * 70)
print("V8 RESULTS")
print("=" * 70)

print(
    "Source 1 rows:",
    len(s1)
)

print(
    "Output rows:",
    len(results)
)

print(
    "Average candidates:",
    f"{candidate_counts.mean():.2f}"
)

print(
    "Maximum candidates:",
    candidate_counts.max()
)

print(
    "Total runtime:",
    f"{time.time() - start_total:.1f}s"
)

print()
print(
    "Saved:",
    OUTPUT_FILE
)

print()
print("=" * 70)
print("DONE!")
print("=" * 70)