# Project 10: Open-Domain Question Answering, Method 3

**M Lekhya Reddy (23CS10039) · Group 28 · Dataset: [`rag-datasets/rag-mini-wikipedia`](https://huggingface.co/datasets/rag-datasets/rag-mini-wikipedia)** (3,200 passages, 918 questions)

All three tasks are in one Colab notebook, [`NLP_RAG.ipynb`](NLP_RAG.ipynb), which runs from top to bottom:

| Notebook section | What it does |
|---|---|
| Setup, 1. Data, 2. Evaluation metrics | installs packages, loads both dataset configs, defines SQuAD-style EM and F1 |
| **Task 1** (sections 3–6) | BM25 retrieval, recall@k, best-sentence selection, rule-based span extraction |
| **Task 2** (sections 7–10) | dense retrieval (MiniLM + FAISS), LLM answer generation (RAG) |
| **Task 3** (sections 11–14) | Reciprocal Rank Fusion of BM25 + dense, lexical-overlap groundedness check |
| Final results, Conclusions | summary tables and discussion |

## Method

| Task | Retriever | Reader |
|---|---|---|
| 1 · Baseline | BM25 (`rank_bm25`, k1 = 1.5, b = 0.75). Tokens: lowercase alphanumeric, scikit-learn English stopwords removed | top-1 passage → sentence with the most IDF-weighted question words → first chunk of non-question words that fits the expected answer type (number, date, name), max 5 words. Yes/no questions are answered "yes" |
| 2 · RAG | `sentence-transformers/all-MiniLM-L6-v2`, L2-normalised, FAISS `IndexFlatIP` (cosine), top-5 | `Qwen/Qwen2.5-1.5B-Instruct` in float16, 200 tokens per passage, ≤ 64 new tokens, greedy decoding, first line of the reply kept |
| 3 · Hybrid | RRF of the BM25 and dense top-100 lists, score = Σ 1 / (60 + rank), top-5 | same LLM, prompt and decoding as Task 2, so any change comes from retrieval only |

**Evaluation.** EM and token F1 after SQuAD normalisation, on all 918 questions. There are no gold-passage labels, so Recall@k counts a hit when the normalised gold answer appears as whole words in one of the top-k passages. It is computed on the 508 questions whose answer is not yes/no.

**Groundedness check (Task 3).** overlap = share of the answer's content words found in its top-5 passages. For yes/no answers, it is the share of the question's content words found in the best single passage. An answer is flagged when overlap < τ, and it counts as correct when F1 ≥ 0.5.

## Results

From the notebook's output (all 918 questions; "span" = the 508 questions whose answer is not yes/no):

| System | EM | F1 | span EM | span F1 | R@1 | R@3 | R@5 | R@10 |
|---|---|---|---|---|---|---|---|---|
| Task 1: BM25 + extractive | 0.413 | 0.459 | 0.098 | 0.182 | **0.478** | 0.571 | 0.602 | 0.630 |
| Task 2: Dense + LLM | 0.550 | 0.641 | 0.295 | 0.460 | 0.398 | 0.528 | 0.589 | 0.632 |
| Task 3: Hybrid RRF + LLM | **0.564** | **0.657** | **0.321** | **0.488** | 0.474 | **0.587** | **0.622** | **0.661** |

Yes/no EM is 0.802 for Task 1 (it always answers "yes") and 0.866 for Tasks 2 and 3.

Groundedness check at τ = 0.5:

| Answers | check accuracy | correct answers with high overlap | incorrect answers with low overlap | flagged | baseline (flag nothing) |
|---|---|---|---|---|---|
| Task 2 | 0.678 | 0.961 | 0.163 | 76 | 0.645 |
| Task 3 | 0.680 | 0.975 | 0.106 | 48 | 0.660 |

**Main findings**
- BM25 beats dense retrieval at R@1 (0.478 vs 0.398), because the questions reuse the passages' names and rare words. RRF is the best retriever from R@3 onwards.
- The LLM reader gives the biggest gain: span EM goes from 0.098 to 0.295. Hybrid retrieval adds a smaller gain (EM 0.550 → 0.564).
- The overlap check keeps almost all correct answers but catches only 11–16% of wrong ones. Most wrong answers copy the wrong entity from the passages, so they still have high overlap.

The full analysis, including all τ values (0.25, 0.5, 0.75, 1.0), is in [`outputs/results_summary.md`](outputs/results_summary.md).

## How to run

**Google Colab (used for the reported results)**
1. Open [Google Colab](https://colab.research.google.com) → `File → Upload notebook` → `NLP_RAG.ipynb`.
2. `Runtime → Change runtime type → T4 GPU`. Tasks 2 and 3 need a GPU, because the LLM is loaded with `.to("cuda")`.
3. `Runtime → Run all`. The first cell installs `rank_bm25`, `sentence-transformers`, `faiss-cpu` and `datasets`; everything else is already in Colab. A Hugging Face token is not needed, and the warning about unauthenticated requests can be ignored.

The dataset (about 1 MB), the encoder (about 90 MB) and the LLM (about 3 GB) are downloaded automatically on the first run.

**Local machine**
```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook NLP_RAG.ipynb
```
You need an NVIDIA GPU with about 4 GB of memory for the 1.5B model in float16. `requirements.txt` lists package names only. Skip the notebook's first `!pip install` cell if you installed the requirements already.

## Files

```
method3_lekhya_23CS10039/
├── NLP_RAG.ipynb                 all three tasks, with the outputs of the reported run
├── README.md
├── requirements.txt              packages for a local run (names only)
└── outputs/results_summary.md    full results tables and analysis
```

The notebook keeps its results in memory and prints them; it does not write result files or plots.
