# Project 10: Open-Domain Question Answering – From Lexical Retrieval to RAG

**Mehul Mathur (23CS30033) · Group 28 · Project 10 · Dataset: rag-mini-wikipedia**

All code is in one Colab notebook, `notebooks/Project10_Group28_23CS30033.ipynb`, split into clear sections:

| Section | What it does |
|---|---|
| 0. Setup | Installs packages, sets config and seeds, loads the dataset, defines EM / F1 / recall@k |
| Task 1 | BM25 retrieval + rule-based extractive answering |
| Task 2 | Dense retrieval (MiniLM + FAISS) + LLM generation (RAG) |
| Task 3 | Reciprocal Rank Fusion of BM25 + dense, and lexical-overlap groundedness check |
| Results & Conclusions | Summary table, plots, discussion |

## 1. Environment setup

**Recommended: Google Colab with a T4 GPU**
1. Open the notebook in Colab.
2. Runtime → Change runtime type → **T4 GPU**.
3. Run the first Setup cell. It installs the packages that Colab does not have:
   ```
   pip install rank_bm25==0.2.2 faiss-cpu==1.11.0
   python -m spacy download en_core_web_sm
   ```
   spaCy may print "Restart to reload dependencies". This is not needed: the model loads fine without a restart.

**Local machine (optional)**
```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
jupyter notebook notebooks/Project10_Group28_23CS30033.ipynb
```
You need a CUDA GPU with at least 8 GB memory for the 3B LLM in float16. It will run on CPU, but generation will be very slow.

The exact versions used for the reported results are in `requirements.txt` (Python 3.13, Colab, Nov 2026). The notebook also writes the versions of the current run to `results/requirements.txt`.

## 2. Datasets

Only **rag-mini-wikipedia** was assigned to our group and used for all three tasks.

| Dataset | How to get it | Used? |
|---|---|---|
| rag-mini-wikipedia | Downloads automatically in the notebook: `load_dataset("rag-datasets/rag-mini-wikipedia", "text-corpus")` (3,200 passages) and `load_dataset("rag-datasets/rag-mini-wikipedia", "question-answer")` (918 QA pairs, `test` split). No manual preparation. | **Yes** |
| Natural Questions (open) | `load_dataset("google-research-datasets/nq_open")`; a Wikipedia-paragraph index must be built as described in the course Datasets document, Section 10 | No |
| TriviaQA (rc.nocontext) | `load_dataset("mandarjoshi/trivia_qa", "rc.nocontext")`; a Wikipedia-paragraph index must be built as described in the course Datasets document, Section 10 | No |

The passage index is built once in Setup and reused by Tasks 1, 2 and 3.

## 3. Pretrained checkpoints

| Use | Checkpoint | Source |
|---|---|---|
| Dense retriever (Task 2, 3) | `sentence-transformers/all-MiniLM-L6-v2` | https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2 |
| Generator LLM (Task 2, 3) | `Qwen/Qwen2.5-3B-Instruct` | https://huggingface.co/Qwen/Qwen2.5-3B-Instruct |
| Sentence splitting, noun phrases, number entities (Task 1) | `en_core_web_sm` 3.8.0 (spaCy) | https://github.com/explosion/spacy-models |

No model is trained or fine-tuned. All checkpoints download automatically on first use.

## 4. How to run

**End to end (in sequence):** Runtime → Run all. Takes about 20 minutes on a T4, most of it LLM generation (4 runs × 918 questions).

**Each task on its own:** always run the Setup section first (all cells under "0. Setup"), then:

| To run | Run these sections | Notes |
|---|---|---|
| Task 1 | Setup → Task 1 | No GPU needed |
| Task 2 | Setup → Task 2 | Retrieval and generation do not need Task 1. Only the last comparison plot in 2.3 uses BM25 recall from Task 1; run Task 1 first (takes ~30 s) to get it |
| Task 3 | Setup → Task 1 → Task 2 → Task 3 | Task 3 fuses the BM25 and dense rankings, so both must exist |

LLM answers are cached in `results/pred_*.json`. If you run a section again with the same questions, it reads the cache instead of generating again. Delete those files to force new generation.

**Quick test run:** set `N_QUESTIONS = 50` in the config cell (Setup) to use a random sample of 50 questions (sampled with `SEED = 42`).

## 5. Configuration and seeds

All settings are in the config cell at the top of the notebook:

| Setting | Value |
|---|---|
| Random seed (Python, NumPy, PyTorch, question sampling) | `SEED = 42` |
| BM25 | `rank_bm25.BM25Okapi`, k1 = 1.5, b = 0.75; lowercase, `\w+` tokens, scikit-learn English stop-words removed |
| Passages given to the reader | top-k = 5 |
| recall@k | k ∈ {1, 3, 5, 10} |
| Dense index | FAISS `IndexFlatIP` on L2-normalised embeddings (= cosine similarity) |
| Generation | greedy (temperature = 0), max new tokens = 64, each passage cut to 200 tokens, batch size 16, float16 |
| RRF constant | 60, fused over the top-100 of each retriever |
| Groundedness threshold | τ = 0.5 (fixed before the run) |
| "Correct" for groundedness analysis | token-F1 ≥ 0.5 |

recall@k counts a question as a hit if the normalised gold answer appears in any of the top-k passages (the dataset has no gold passage IDs). It is computed on the 508 span-answer questions only; the 410 yes/no questions are excluded.

## 6. Outputs

Everything is written to `results/` (in Colab: `/content/results`):

| File | Contents |
|---|---|
| `metrics_summary.csv` | EM, F1 (all / span / yes-no) and recall@k for every system |
| `groundedness.csv` | Groundedness-check results |
| `predictions.csv` | Per-question gold answer and predictions of all main systems, with overlap scores |
| `pred_*.json` | Cached LLM answers (dense, bm25, hybrid, closed-book) |
| `dataset_overview.png`, `task1.png`, `task2.png`, `task3_recall.png`, `task3_grid.png`, `task3_groundedness.png`, `summary.png` | Figures |
| `requirements.txt` | Package versions of the current run |

## 7. Main results

| System | EM | F1 | R@5 | R@10 |
|---|---|---|---|---|
| Task 1: BM25 + Extractive (spec) | 1.96 | 5.04 | 60.24 | 62.80 |
| Task 2: Dense + LLM (RAG) | 54.47 | 64.30 | 58.86 | 63.19 |
| Task 3: Hybrid (RRF) + LLM | **55.99** | **66.02** | **61.42** | **66.14** |
| Closed-book LLM (reference) | 25.49 | 30.77 | – | – |

Groundedness check on Task 2 answers (τ = 0.5): 99.6% of correct answers have high overlap; 10.4% of incorrect answers have low overlap; 22 of 23 flagged answers are wrong. See the Discussion section of the notebook for the full analysis.

## References

- Lewis et al., *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks*, NeurIPS 2020 (arXiv:2005.11401). Reference retriever: https://github.com/facebookresearch/DPR
- Chen, Fisch, Weston, Bordes, *Reading Wikipedia to Answer Open-Domain Questions* (DrQA), ACL 2017
- Joshi, Choi, Weld, Zettlemoyer, *TriviaQA*, ACL 2017
