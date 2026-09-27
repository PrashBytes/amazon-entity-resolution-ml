import pandas as pd
import sqlite3
import re
import unicodedata
import time
import os
from rapidfuzz.fuzz import ratio


# ============================================================
# V8 FTS5 FAST CANDIDATE GENERATION
#
# Strategy:
#   Source 2 + Source 3
#        ↓
#   SQLite FTS5 trigram indexes
#        ↓
#   fast candidate retrieval
#        ↓
#   RapidFuzz reranking
#        ↓
#   top 100 candidates
#
# IMPORTANT:
#   - NO TF-IDF
#   - NO LightGBM
#   - NO millions of Python dictionaries
#   - INDEX IS PERSISTENT
# ============================================================


print("=" * 75)
print("V8 FTS5 FAST CANDIDATE GENERATION")
print("=" * 75)


# ============================================================
# SETTINGS
# ============================================================

TEST_ROWS = 1000

# Persistent database.
# It will be created only once.
DB_FILE = "candidate_fts_v8.db"

OUTPUT_FILE = "v8_fts_candidates.tsv"

TOP_CANDIDATES = 100

# Number retrieved from each search strategy
NAME_RETRIEVE = 150
ADDRESS_RETRIEVE = 100

# SQLite insertion batch
BATCH_SIZE = 10000


# ============================================================
# FILES
# ============================================================

SOURCE1_FILE = "dataset/train/train_source1.tsv"
SOURCE2_FILE = "dataset/train/train_source2.tsv"
SOURCE3_FILE = "dataset/train/train_source3.tsv"


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


def make_fts_text(text):

    """
    FTS5 trigram tokenizer works best when the text
    contains normal contiguous characters.

    We keep spaces because business names and addresses
    contain useful word boundaries.
    """

    return normalize_text(text)


# ============================================================
# LOAD SOURCE 1
# ============================================================

print("\nLoading Source 1...")

start = time.time()

s1 = pd.read_csv(
    SOURCE1_FILE,
    sep="\t",
    dtype=str
)

print(
    "Source 1:",
    len(s1)
)


if TEST_ROWS is not None:

    print()
    print("=" * 75)
    print(
        f"TEST MODE: {TEST_ROWS} Source-1 rows"
    )
    print("=" * 75)

    s1 = s1.head(TEST_ROWS).copy()


# ============================================================
# SQLITE DATABASE
# ============================================================

print()
print("=" * 75)
print("PREPARING PERSISTENT FTS5 INDEX")
print("=" * 75)

db_exists = os.path.exists(DB_FILE)

conn = sqlite3.connect(
    DB_FILE
)

cursor = conn.cursor()


# ============================================================
# CHECK FTS5 SUPPORT
# ============================================================

try:

    cursor.execute(
        """
        CREATE VIRTUAL TABLE IF NOT EXISTS
        fts_test
        USING fts5(
            text,
            tokenize='trigram'
        )
        """
    )

    cursor.execute(
        "DROP TABLE fts_test"
    )

    conn.commit()

except Exception as e:

    print()
    print("ERROR: SQLite FTS5 trigram is not available.")
    print("Details:", e)
    conn.close()
    raise SystemExit


print("SQLite FTS5 trigram support: OK")


# ============================================================
# CREATE DATABASE TABLES
# ============================================================

cursor.execute(
    """
    CREATE TABLE IF NOT EXISTS candidates (
        entity_id TEXT PRIMARY KEY,
        source TEXT,
        name_norm TEXT,
        address_norm TEXT,
        country_norm TEXT
    )
    """
)


cursor.execute(
    """
    CREATE VIRTUAL TABLE IF NOT EXISTS
    name_fts
    USING fts5(
        entity_id UNINDEXED,
        name_norm,
        tokenize='trigram'
    )
    """
)


cursor.execute(
    """
    CREATE VIRTUAL TABLE IF NOT EXISTS
    address_fts
    USING fts5(
        entity_id UNINDEXED,
        address_norm,
        tokenize='trigram'
    )
    """
)


conn.commit()


# ============================================================
# CHECK WHETHER INDEX ALREADY EXISTS
# ============================================================

cursor.execute(
    "SELECT COUNT(*) FROM candidates"
)

existing_count = cursor.fetchone()[0]


# Expected number of candidates
#
# We don't actually need to load Source 2/3 just to know
# the count if the DB already contains data.
# ============================================================

if existing_count > 0:

    print()
    print(
        "Existing persistent index found!"
    )

    print(
        "Indexed candidate records:",
        existing_count
    )

    print(
        "Reusing existing index."
    )

else:

    print()
    print(
        "No existing candidate index found."
    )

    print(
        "Building index for Source 2 + Source 3..."
    )

    # --------------------------------------------------------
    # LOAD SOURCE 2
    # --------------------------------------------------------

    print("\nLoading Source 2...")

    s2 = pd.read_csv(
        SOURCE2_FILE,
        sep="\t",
        dtype=str
    )

    print(
        "Source 2:",
        len(s2)
    )


    # --------------------------------------------------------
    # LOAD SOURCE 3
    # --------------------------------------------------------

    print("\nLoading Source 3...")

    s3 = pd.read_csv(
        SOURCE3_FILE,
        sep="\t",
        dtype=str
    )

    print(
        "Source 3:",
        len(s3)
    )


    # --------------------------------------------------------
    # INSERT CANDIDATES
    # --------------------------------------------------------

    print()
    print(
        "Normalizing and inserting candidates..."
    )

    build_start = time.time()

    batch_candidates = []
    batch_names = []
    batch_addresses = []

    inserted = 0


    def flush_batch():

        global batch_candidates
        global batch_names
        global batch_addresses

        if not batch_candidates:
            return

        cursor.executemany(
            """
            INSERT OR IGNORE INTO candidates
            (
                entity_id,
                source,
                name_norm,
                address_norm,
                country_norm
            )
            VALUES (?, ?, ?, ?, ?)
            """,
            batch_candidates
        )

        cursor.executemany(
            """
            INSERT INTO name_fts
            (
                entity_id,
                name_norm
            )
            VALUES (?, ?)
            """,
            batch_names
        )

        cursor.executemany(
            """
            INSERT INTO address_fts
            (
                entity_id,
                address_norm
            )
            VALUES (?, ?)
            """,
            batch_addresses
        )

        conn.commit()

        batch_candidates = []
        batch_names = []
        batch_addresses = []


    # --------------------------------------------------------
    # PROCESS SOURCE 2
    # --------------------------------------------------------

    for i, (_, row) in enumerate(
        s2.iterrows(),
        start=1
    ):

        entity_id = str(
            row["entity_id"]
        )

        name = make_fts_text(
            row.get(
                "business_name",
                ""
            )
        )

        address = make_fts_text(
            row.get(
                "business_address",
                ""
            )
        )

        country = str(
            row.get(
                "country",
                ""
            )
        ).lower().strip()


        batch_candidates.append(
            (
                entity_id,
                "source2",
                name,
                address,
                country
            )
        )

        batch_names.append(
            (
                entity_id,
                name
            )
        )

        batch_addresses.append(
            (
                entity_id,
                address
            )
        )

        inserted += 1


        if len(batch_candidates) >= BATCH_SIZE:

            flush_batch()

            if inserted % 100000 == 0:

                elapsed = (
                    time.time()
                    -
                    build_start
                )

                print(
                    f"Indexed "
                    f"{inserted:,} Source-2 "
                    f"records | "
                    f"{elapsed:.1f}s"
                )


    flush_batch()


    # --------------------------------------------------------
    # PROCESS SOURCE 3
    # --------------------------------------------------------

    source3_inserted = 0

    for i, (_, row) in enumerate(
        s3.iterrows(),
        start=1
    ):

        entity_id = str(
            row["entity_id"]
        )

        name = make_fts_text(
            row.get(
                "business_name",
                ""
            )
        )

        address = make_fts_text(
            row.get(
                "business_address",
                ""
            )
        )

        country = str(
            row.get(
                "country",
                ""
            )
        ).lower().strip()


        batch_candidates.append(
            (
                entity_id,
                "source3",
                name,
                address,
                country
            )
        )

        batch_names.append(
            (
                entity_id,
                name
            )
        )

        batch_addresses.append(
            (
                entity_id,
                address
            )
        )

        source3_inserted += 1
        inserted += 1


        if len(batch_candidates) >= BATCH_SIZE:

            flush_batch()

            if source3_inserted % 100000 == 0:

                elapsed = (
                    time.time()
                    -
                    build_start
                )

                print(
                    f"Indexed "
                    f"{source3_inserted:,} "
                    f"Source-3 records | "
                    f"Total: "
                    f"{inserted:,} | "
                    f"{elapsed:.1f}s"
                )


    flush_batch()


    print()
    print(
        "Index build completed."
    )

    print(
        "Total indexed:",
        inserted
    )

    print(
        "Index build time:",
        f"{time.time() - build_start:.1f}s"
    )


# ============================================================
# LOAD CANDIDATE LOOKUP
# ============================================================

print()
print(
    "Loading candidate metadata..."
)

cursor.execute(
    """
    SELECT
        entity_id,
        source,
        name_norm,
        address_norm,
        country_norm
    FROM candidates
    """
)

candidate_lookup = {}

for row in cursor.fetchall():

    entity_id = row[0]

    candidate_lookup[
        entity_id
    ] = {
        "source": row[1],
        "name_norm": row[2],
        "address_norm": row[3],
        "country_norm": row[4]
    }


print(
    "Candidate lookup loaded:",
    len(candidate_lookup)
)


# ============================================================
# FTS SEARCH
# ============================================================

def fts_search(
    table,
    column,
    query,
    limit
):

    if not query:
        return []

    # Need at least 3 characters for trigram.
    if len(query) < 3:
        return []

    # FTS5 MATCH syntax.
    #
    # We search the normalized string directly.
    # Trigram tokenizer handles partial/fuzzy character
    # overlap.

    try:

        sql = f"""
        SELECT entity_id
        FROM {table}
        WHERE {table}
        MATCH ?
        LIMIT ?
        """

        cursor.execute(
            sql,
            (
                query,
                limit
            )
        )

        return [
            row[0]
            for row in cursor.fetchall()
        ]

    except Exception:

        return []


# ============================================================
# CANDIDATE RETRIEVAL
# ============================================================

def retrieve_candidates(
    source_name,
    source_address
):

    candidate_ids = set()


    # --------------------------------------------------------
    # NAME SEARCH
    # --------------------------------------------------------

    name_results = fts_search(
        "name_fts",
        "name_norm",
        source_name,
        NAME_RETRIEVE
    )

    candidate_ids.update(
        name_results
    )


    # --------------------------------------------------------
    # ADDRESS SEARCH
    # --------------------------------------------------------

    address_results = fts_search(
        "address_fts",
        "address_norm",
        source_address,
        ADDRESS_RETRIEVE
    )

    candidate_ids.update(
        address_results
    )


    # --------------------------------------------------------
    # Additional shorter name searches
    #
    # These help when the complete name contains extra text.
    # --------------------------------------------------------

    if len(source_name) >= 8:

        prefix = source_name[:8]

        results = fts_search(
            "name_fts",
            "name_norm",
            prefix,
            100
        )

        candidate_ids.update(
            results
        )


    if len(source_name) >= 6:

        prefix = source_name[:6]

        results = fts_search(
            "name_fts",
            "name_norm",
            prefix,
            100
        )

        candidate_ids.update(
            results
        )


    return candidate_ids


# ============================================================
# RANKING
# ============================================================

def rank_candidates(
    source_name,
    source_address,
    candidate_ids
):

    ranked = []


    for entity_id in candidate_ids:

        candidate = candidate_lookup.get(
            entity_id
        )

        if candidate is None:
            continue


        candidate_name = candidate[
            "name_norm"
        ]

        candidate_address = candidate[
            "address_norm"
        ]


        # ----------------------------------------------------
        # NAME SIMILARITY
        # ----------------------------------------------------

        name_ratio = ratio(
            source_name,
            candidate_name
        )


        # ----------------------------------------------------
        # ADDRESS SIMILARITY
        # ----------------------------------------------------

        address_ratio = ratio(
            source_address,
            candidate_address
        )


        # ----------------------------------------------------
        # WRatio-like structural checks using RapidFuzz
        # ----------------------------------------------------

        name_prefix_score = 0

        if source_name and candidate_name:

            common_prefix = 0

            max_prefix = min(
                len(source_name),
                len(candidate_name),
                10
            )

            for i in range(
                max_prefix
            ):

                if (
                    source_name[i]
                    ==
                    candidate_name[i]
                ):
                    common_prefix += 1
                else:
                    break

            name_prefix_score = (
                common_prefix
                /
                max(
                    1,
                    max_prefix
                )
            )


        # ----------------------------------------------------
        # FINAL SCORE
        # ----------------------------------------------------

        final_score = (

            0.62 * name_ratio

            +

            0.33 * address_ratio

            +

            8.0 * name_prefix_score
        )


        ranked.append(
            (
                entity_id,
                final_score
            )
        )


    ranked.sort(
        key=lambda x: x[1],
        reverse=True
    )


    return [
        entity_id
        for entity_id, score
        in ranked[
            :TOP_CANDIDATES
        ]
    ]


# ============================================================
# PROCESS SOURCE 1
# ============================================================

print()
print("=" * 75)
print("GENERATING V8 FTS5 CANDIDATES")
print("=" * 75)

process_start = time.time()

results = []

total = len(s1)


for i, (_, row) in enumerate(
    s1.iterrows(),
    start=1
):

    source_name = make_fts_text(
        row.get(
            "business_name",
            ""
        )
    )

    source_address = make_fts_text(
        row.get(
            "business_address",
            ""
        )
    )


    # --------------------------------------------------------
    # RETRIEVE
    # --------------------------------------------------------

    candidate_ids = retrieve_candidates(
        source_name,
        source_address
    )


    # --------------------------------------------------------
    # RERANK
    # --------------------------------------------------------

    ranked_ids = rank_candidates(
        source_name,
        source_address,
        candidate_ids
    )


    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    results.append({

        "source1_entity_id":
            str(row["entity_id"]),

        "candidate_entity_ids":
            ",".join(ranked_ids)
    })


    # --------------------------------------------------------
    # PROGRESS
    # --------------------------------------------------------

    if i % 50 == 0:

        elapsed = (
            time.time()
            -
            process_start
        )

        rate = (
            i
            /
            max(
                elapsed,
                0.001
            )
        )

        remaining = (
            total - i
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
            f"{i:,}/{total:,} "
            f"| "
            f"{rate:.1f} rows/sec "
            f"| "
            f"ETA {eta:.1f}s"
        )


# ============================================================
# SAVE
# ============================================================

output = pd.DataFrame(
    results
)

output.to_csv(
    OUTPUT_FILE,
    sep="\t",
    index=False
)


# ============================================================
# VALIDATION
# ============================================================

candidate_counts = (
    output[
        "candidate_entity_ids"
    ]
    .fillna("")
    .apply(
        lambda x:
        len(
            [
                y
                for y in x.split(",")
                if y
            ]
        )
    )
)


print()
print("=" * 75)
print("V8 FTS5 RESULTS")
print("=" * 75)

print(
    "Source 1 rows:",
    len(output)
)

print(
    "Average candidates:",
    round(
        candidate_counts.mean(),
        2
    )
)

print(
    "Maximum candidates:",
    candidate_counts.max()
)

print(
    "Rows with candidates:",
    (
        candidate_counts > 0
    ).sum()
)

print(
    "Empty candidate rows:",
    (
        candidate_counts == 0
    ).sum()
)

print(
    "Processing time:",
    f"{time.time() - process_start:.1f}s"
)

print(
    "Total runtime:",
    f"{time.time() - start:.1f}s"
)

print()
print(
    "Saved:",
    OUTPUT_FILE
)

print()
print("=" * 75)
print("DONE!")
print("=" * 75)

conn.close()