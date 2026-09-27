# Amazon Entity Resolution ML

An entity-resolution and record-linkage pipeline for matching business records across multiple data sources.

The project focuses on identifying whether records from different sources refer to the same real-world business entity.

## Project Overview

The dataset contains millions of business records distributed across three sources:

* **Source 1:** 2,206,821 records
* **Source 2:** 5,034,616 records
* **Source 3:** 5,285,603 records
* **Total candidate records:** 10,320,219

The goal is to match each Source-1 entity against corresponding entities in Source 2 and Source 3.

## Pipeline

The matching system is developed as a multi-stage pipeline:

```text
Raw Data
   ↓
Text Normalization
   ↓
Candidate Generation
   ↓
Blocking
   ↓
Candidate Ranking
   ↓
ML Matching Model
   ↓
Final Entity Matches
```

## Candidate Generation

Multiple candidate-generation approaches were experimented with, including:

* Exact-name blocking
* Name prefix blocking
* First/last word blocking
* Name signatures
* Sorted-name matching
* Address-number blocking
* Address prefix blocking
* Country-aware blocking
* Rare-token blocking
* Dynamic candidate generation
* SQLite FTS5 experimentation
* TF-IDF experimentation

The main objective of candidate generation is **high recall while keeping the number of candidates manageable**.

## Candidate Generation Experiments

Several versions were developed during experimentation:

| Version | Approach                               | Status          |
| ------- | -------------------------------------- | --------------- |
| V6      | Multi-blocking + RapidFuzz             | Baseline        |
| V7      | High-recall multi-blocking             | Strong baseline |
| V7.1    | Faster multi-blocking                  | Experimental    |
| V7.2    | Targeted blocking                      | Experimental    |
| V8      | FTS5-based retrieval                   | Experimental    |
| Dynamic | Dynamic threshold candidate generation | Experimental    |

The current experiments are evaluated using Recall@K.

## Evaluation Metrics

Candidate generation is evaluated using:

* Recall@1
* Recall@5
* Recall@10
* Recall@20
* Recall@50
* Recall@100

Recall@K measures whether the correct entity appears within the top K generated candidates.

## Repository Structure

```text
ML Amazon Challenge/
│
├── dataset/
│   ├── train/
│   └── output/
│
├── candidate_generation.py
├── candidate_generation_v2.py
├── candidate_generation_v3.py
├── candidate_generation_v4.py
├── candidate_generation_v6.py
├── candidate_generation_v7.py
├── candidate_generation_v7_1.py
├── candidate_generation_v7_2.py
├── candidate_generation_v8_fast.py
├── candidate_generation_v8_fts.py
├── candidate_generation_v8_tfidf.py
│
├── evaluate_v2.py
├── evaluate_v5.py
├── evaluate_v6.py
├── evaluate_v7_2.py
├── evaluate_v8_fts.py
├── evaluate_dynamic_candidates.py
├── evaluate_dynamic_train.py
│
├── train_match_model.py
├── train_match_model_v2.py
├── v5_ranking.py
│
├── submit_ml_v2.py
├── submit_ml_v4.py
├── submit_ml_v6.py
├── submit_ml_v7.py
├── submit_ml_fast.py
├── submit_ml_fast_v2.py
│
└── README.md
```

## Technologies

* Python
* Pandas
* NumPy
* RapidFuzz
* SQLite / FTS5
* Machine Learning
* Record Linkage
* Entity Resolution

## Current Focus

The primary optimization target is improving candidate recall while keeping candidate generation computationally efficient.

The project is being developed iteratively by comparing different blocking, retrieval, and ranking strategies against the provided ground-truth matches.

## Important Note

The original datasets and generated large candidate/output files are intentionally excluded from version control because of their size.

This repository contains the source code and experimentation history required to reproduce and extend the matching pipeline.
