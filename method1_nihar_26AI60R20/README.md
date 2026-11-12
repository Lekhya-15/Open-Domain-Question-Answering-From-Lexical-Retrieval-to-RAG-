# Project 10 – Open-Domain Question Answering: From Lexical Retrieval to RAG

**Group 28** · **Dataset:** [`rag-datasets/rag-mini-wikipedia`](https://huggingface.co/datasets/rag-datasets/rag-mini-wikipedia) (3,200 Wikipedia passages, 918 questions)

| Name | Roll No. | Method |
|---|---|---|
| M Lekhya Reddy | 23CS10039 | Method 3 |
| Mehul Mathur | 23CS30033 | Method 2 |
| Nihar Ranjan Das | 26AI60R20 | Method 1 |

We built three implementations of the full pipeline (Methods 1–3). They follow the same brief but differ in text cleaning, tokenisation, the extraction rule, the LLM size, the prompt and the overlap formula. The group report compares them side by side: [`report/Project10_Group28_Report.pdf`](../report/Project10_Group28_Report.pdf) (in the repository root).

This repository contains the code and results of **Method 1**. Methods 2 and 3 are single end-to-end Colab notebooks by their authors (`Project10_Group28_23CS30033.ipynb` and `NLP_RAG.ipynb`).

| Task | Retriever | Reader | Extra (Method 1) |
|---|---|---|---|
| **1 · Baseline** | BM25 (`rank_bm25`, k1 = 1.5, b = 0.75), top-5 | extractive: best sentence of the top passage → longest noun phrase / number span | error analysis, variant that skips question words |
| **2 · RAG reproduction** | `all-MiniLM-L6-v2` + FAISS flat index (cosine), top-5 | Qwen2.5-Instruct (1.5B in Methods 1 and 3, 3B in Method 2), 200 tokens per passage, ≤ 64 new tokens, greedy decoding | retrieval vs. generation errors, yes/no analysis |
| **3 · Combine & improve** | Reciprocal Rank Fusion of BM25 + dense (constant 60), top-5 | same LLM, prompt and decoding as Task 2 | lexical-overlap groundedness check, abstention, McNemar tests |

All systems are scored with Exact Match, token F1 and Recall@k (k = 1, 3, 5, 10) on the same passage index.

## Results

### All three methods (EM / F1 in %, all 918 questions)

| System | Method 1 | Method 2 | Method 3 |
|---|---|---|---|
| Task 1: BM25 + extractive | 1.9 / 5.5 | 2.0 / 5.0 | 41.3 / 45.9 * |
| Task 2: Dense + LLM | 56.4 / 67.1 | 54.5 / 64.3 | 55.0 / 64.1 |
| Task 3: Hybrid (RRF) + LLM | **57.2 / 68.4** | 56.0 / 66.0 | 56.4 / 65.7 |

\* Method 3's Task 1 answers "yes" to every yes/no question, which alone gives 80.2% EM on those questions.

### Method 1 in detail (%, all 918 questions; in brackets: the 498 factual, non-yes/no questions)

| System | EM | F1 | R@1 | R@3 | R@5 | R@10 |
|---|---|---|---|---|---|---|
| Task 1: BM25 + extractive | 1.9 (3.4) | 5.5 (9.9) | 27.3 (49.0) | 33.3 (58.6) | 36.3 (62.7) | 39.5 (65.7) |
| Task 2: Dense + LLM | 56.4 (31.3) | 67.1 (50.5) | 22.7 (40.0) | 30.6 (53.8) | 34.4 (60.0) | 38.0 (64.5) |
| Task 3: Hybrid (RRF) + LLM | 57.2 (34.1) | 68.4 (54.1) | 27.2 (48.4) | 34.2 (60.0) | 36.8 (63.7) | 40.4 (67.9) |

### Main findings

- **Method 1 gives the best system** (EM 57.2, F1 68.4). It stays ahead when re-scored on the same 508 factual and 410 yes/no questions that Methods 2 and 3 use (factual EM 33.5 vs 31.9 and 32.1; yes/no EM 88.3 vs 84.2 and 86.6). Dense recall is a tie, because all three methods use the same encoder.
- **The reader matters most:** replacing the extractive reader with an LLM raises EM from about 2 to 54–56 in all three methods. The 3B model (Method 2) was not better than the 1.5B model.
- **Hybrid retrieval** gives the best Recall@k for k ≥ 3 and +0.8 to +1.5 EM over Task 2 in every method. In Method 1 the EM gain is not significant (38 answers fixed, 31 broken, McNemar p = 0.47).
- **Groundedness check** (Method 1, 481 non-yes/no Task 2 answers, threshold 0.5, correct = F1 ≥ 0.5): 98.9% of correct answers pass, but only 14.7% of incorrect answers are caught (check accuracy 0.607 vs 0.547 for never flagging). Most errors are copied from the passages. On closed-book answers (Method 2) the check catches 59.7% of the wrong ones, and the answers it keeps are right 27.2% of the time against 14.2% overall.
- **Paper comparison:** on factual questions our EM (29.5–31.3) is about 13–15 points below RAG on Natural Questions (44.5), mainly because nothing is fine-tuned, many gold answers are long, the retriever is small and the reader sees only 5 passages.

## Repository layout

This folder holds Method 1. The group report and presentation are in [`../report/`](../report/), and Methods 2 and 3 are in [`../method2_mehul_23CS30033/`](../method2_mehul_23CS30033/) and [`../method3_lekhya_23CS10039/`](../method3_lekhya_23CS10039/).

```
method1_nihar_26AI60R20/
├── Task_Final/                                 Method 1 as three notebooks (Task 1 → 2 → 3), write to Task_Final/outputs/
│   ├── Task1_BM25_Extractive.ipynb
│   ├── Task2_RAG_Dense_LLM.ipynb
│   └── Task3_Hybrid_Grounding.ipynb
├── Project10_OpenDomainQA_End_to_End.ipynb     Method 1, all three tasks in one notebook (writes to Output_Final/)
├── Output_Final/                               Method 1 results, plots and generation caches of the final run
└── requirements.txt                            packages for a local run (names only)
```

Both ways of running Method 1 (one notebook, or three) use the same data, settings and evaluation code and give the same numbers.

## 1. Environment setup

### Option A – Google Colab (recommended; used for all runs)

1. Open [Google Colab](https://colab.research.google.com) → `File → Upload notebook` and select the notebook.
2. `Runtime → Change runtime type → T4 GPU` (Task 1 alone also runs on a CPU runtime).
3. `Runtime → Run all`.

The first code cell of every notebook installs what Colab does not have (`%pip install …`), including the spaCy model `en_core_web_sm` where it is needed. Nothing else has to be installed. A Hugging Face token is not needed; the warning about unauthenticated requests can be ignored.

### Option B – Local machine

Python 3.11 or newer. An NVIDIA GPU is recommended for Tasks 2 and 3: the 1.5B LLM needs about 3 GB of GPU memory in float16 (Method 2's 3B model needs a GPU with at least 8 GB). On a CPU everything runs, but answer generation is much slower.

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
jupyter notebook
```

`requirements.txt` lists only the package names, so pip installs current releases. The first cell of each Method 1 notebook installs the versions its results depend on (`rank_bm25` 0.2.2, spaCy 3.8 and `en_core_web_sm` 3.8.0). The exact versions used for the reported runs of all three methods are listed in Appendix A of the report. For a smaller CPU-only PyTorch, run `pip install torch --index-url https://download.pytorch.org/whl/cpu` before installing the requirements.

## 2. Dataset

All three tasks use one dataset, `rag-datasets/rag-mini-wikipedia`, which is the dataset assigned to our group. The other two datasets of the project (Natural Questions and TriviaQA) are used by other groups and are not part of this project.

| Configuration | Split | Size | Used as |
|---|---|---|---|
| `text-corpus` | `passages` | 3,200 passages | the retrieval index for all three tasks |
| `question-answer` | `test` | 918 questions with gold answers | evaluation for all three tasks |

**Download.** Nothing has to be downloaded by hand. Each notebook loads both configurations with `datasets.load_dataset` (Method 1 at the pinned revision `1f9f3b53fbc5995b85aab8e993504ad42c5f16f6`), and Hugging Face caches them in `~/.cache/huggingface/`. To download everything in advance, for example for an offline run, use:

```bash
python -c "
from datasets import load_dataset
from huggingface_hub import snapshot_download
rev = '1f9f3b53fbc5995b85aab8e993504ad42c5f16f6'
for config in ['text-corpus', 'question-answer']:
    load_dataset('rag-datasets/rag-mini-wikipedia', config, revision=rev)
snapshot_download('sentence-transformers/all-MiniLM-L6-v2')   # dense retriever (about 90 MB)
snapshot_download('Qwen/Qwen2.5-1.5B-Instruct')              # generator of Methods 1 and 3 (about 3 GB)
"
```

**Preparation** in Method 1 (identical in all its notebooks):
- **Encoding repair:** 483 passages contain UTF-8 text that was decoded as Latin-1 (e.g. `Lincolnâ\x80\x99s`); `fix_mojibake` reverses this. Only characters change; the number, order and boundaries of the passages stay the same. Methods 2 and 3 use the passages as provided.
- **One index for all tasks:** the passages are indexed exactly as provided, with no re-chunking. BM25 and the dense retriever index the same list in the same order, and all rankings are stored as positions in it.
- **Answer normalisation:** SQuAD-style (lowercase, remove punctuation including Unicode punctuation, remove *a/an/the*, collapse whitespace), used for EM, F1 and answer-string Recall@k.
- **Question subsets:** Method 1 counts an answer as yes/no if its first word is yes or no, giving 420 yes/no and 498 factual questions. Methods 2 and 3 count only answers that are exactly "yes" or "no" (410 and 508). Recall@k is reported on the factual questions, where answer-string matching is meaningful; the report's cross-method table re-scores Method 1 on the 508 / 410 sets.

## 3. Running the tasks

Approximate run times on a Colab T4 (Method 1): Task 1 about 2–3 minutes (CPU is enough), Task 2 about 5 minutes, Task 3 about 5 minutes, and the combined notebook about 10–15 minutes. Most of the time goes into LLM generation.

### 3.1 Method 1: all three tasks end-to-end in one notebook

Run `Project10_OpenDomainQA_End_to_End.ipynb` from top to bottom (`Run all`). Task 2 and Task 3 use the results of the earlier tasks directly from memory, so no files have to be uploaded. All outputs go to `Output_Final/`.

From the command line (local environment from Option B):

```bash
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 Project10_OpenDomainQA_End_to_End.ipynb
```

Running it locally overwrites the submitted results in `Output_Final/`, so copy that folder first if you want to keep them.

### 3.2 Method 1: the three tasks in sequence (separate notebooks)

Run the notebooks in `Task_Final/` in this order: **Task 1 → Task 2 → Task 3**. Each notebook saves a JSON file that the next one reads:

| Notebook | Reads | Writes |
|---|---|---|
| `Task1_BM25_Extractive.ipynb` | – | `task1_results.json` |
| `Task2_RAG_Dense_LLM.ipynb` | `task1_results.json` (optional, for the comparison with Task 1) | `task2_results.json` |
| `Task3_Hybrid_Grounding.ipynb` | `task1_results.json` and `task2_results.json` (required) | `task3_results.json` |

How the files get from one notebook to the next:
- **Locally:** automatically, because all three notebooks use the same `outputs/` folder next to them.
  ```bash
  cd Task_Final
  jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 Task1_BM25_Extractive.ipynb
  jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 Task2_RAG_Dense_LLM.ipynb
  jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 Task3_Hybrid_Grounding.ipynb
  ```
- **Colab with Google Drive:** set `USE_DRIVE = True` in the first code cell of every notebook. All three then use `MyDrive/NLP_Project_2/outputs/`.
- **Colab without Drive (default):** each Colab runtime has its own temporary disk. After Task 1 (and Task 2), download `outputs/task1_results.json` (and `outputs/task2_results.json`) from the *Files* panel. When Task 2 or Task 3 says a file is not found, upload it in the file picker that appears. Upload **one file per prompt**: Task 3 asks for the two files one after the other.

### 3.3 Method 1: running each task independently

- **Task 1** needs no input files.
- **Task 2** runs on its own. Without `task1_results.json` it skips only the comparison with Task 1 (Section 8), the complementarity check and the BM25 curves in its plot. In Colab, cancelling the upload prompt does the same.
- **Task 3** needs the results of Tasks 1 and 2. To run it without re-running them, use the files from the final run:
  ```bash
  mkdir -p Task_Final/outputs
  cp Output_Final/task1_results.json Output_Final/task2_results.json Task_Final/outputs/
  ```
  In Colab, upload these two files when asked. Task 3 checks that both files come from the same dataset revision and passage order and cover all 918 questions.

Optionally, copy `Output_Final/task2_generation_cache.json` or `Output_Final/task3_hybrid_generation_cache.json` into the output folder as well. A notebook reuses a cache only if its fingerprint matches the current settings and rankings; otherwise it generates the answers again.

### 3.4 Method 1: settings at the top of the notebooks

| Setting | Default | Effect |
|---|---|---|
| `USE_DRIVE` | `False` | `True`: mount Google Drive (Colab only) and save the results there |
| `QUICK_TEST` | `False` | `True`: run on the first 40 questions only, to check that the code works (Task 2, Task 3 and the combined notebook) |
| `RESET_CACHE` | `False` | `True`: ignore saved LLM answers and generate them again (Task 2, Task 3 and the combined notebook) |
| `BATCH_SIZE` | `8` | prompts generated together. Task 3 takes it from `task2_results.json`. Lower it only for "CUDA out of memory", since a different batch size can change a few greedy answers |

A `QUICK_TEST` run does not save the result JSON files, but it does save its plots (and a `*_quick` cache) into the same output folder. Run the full notebook again afterwards so that the plots match the results.

### 3.5 Methods 2 and 3

Each is one notebook that runs all three tasks end to end: upload it to Colab, select a T4 GPU and use `Run all`.
- **Method 2** (`Project10_Group28_23CS30033.ipynb`): Qwen2.5-3B-Instruct; four LLM runs (dense, closed-book, BM25, hybrid), about 20 minutes. Set `N_QUESTIONS` in its config cell to a small number for a quick test.
- **Method 3** (`NLP_RAG.ipynb`): Qwen2.5-1.5B-Instruct; two LLM runs (dense, hybrid).

## 4. Where results and checkpoints are written (Method 1)

| How you run | Output folder |
|---|---|
| Combined notebook | `Output_Final/` next to the notebook (in Colab: `/content/Output_Final/`; with `USE_DRIVE = True`: `MyDrive/NLP_Project_2/Output_Final/`). In Colab without Drive, the folder is also downloaded as `Output_Final.zip` at the end. |
| Separate notebooks | `outputs/` next to the notebooks, i.e. `Task_Final/outputs/` (in Colab: `/content/outputs/`; with `USE_DRIVE = True`: `MyDrive/NLP_Project_2/outputs/`) |

| File | Content |
|---|---|
| `task1_results.json` | BM25 top-100 ranking per question, baseline and variant predictions, metric tables, passage ids, configuration |
| `task1_recall.png` | BM25 Recall@k |
| `task2_results.json` | dense top-100 ranking per question, generated answers, metric table, passage ids, generation settings |
| `task2_recall.png` | dense vs. BM25 Recall@k |
| `task3_results.json` | fused top-100 ranking, hybrid answers, comparison and groundedness tables, overlap scores, threshold sweep, SHA-256 hashes of the two input files |
| `task3_recall_compare.png` | Recall@k of BM25, dense and hybrid retrieval |
| `task3_overlap_hist.png` | overlap of correct vs. incorrect answers with their passages |
| `task3_threshold_sweep.png` | coverage, F1 and check accuracy for overlap thresholds 0–1 |
| `task2_generation_cache.json`, `task3_hybrid_generation_cache.json` | generation checkpoints (see below) |

**Checkpoints.** No model is trained or fine-tuned, so there are no model checkpoints. The only checkpoints are the two generation caches. They are saved every 10 batches, so an interrupted run resumes where it stopped when the cell is run again. In Colab this survives a disconnect only with `USE_DRIVE = True`, because the runtime's disk is wiped. Each cache stores a fingerprint of the settings (model, system prompt, top-k, token limits, batch size, dataset revision, questions and rankings), and a cache made with other settings is not reused.

Downloaded models and the dataset are cached by Hugging Face in `~/.cache/huggingface/` (change it with the `HF_HOME` environment variable).

## 5. Rebuilding the report

The report source (`report.html`, figures, `plots.py`, `build_pdf.sh`) is not included in this repository; only the finished PDF is, in [`../report/`](../report/).

## 6. Reproducibility

- The dataset is loaded at a pinned revision in Method 1, and all its notebooks use the same passage order, text repair and evaluation code.
- BM25 rankings use a stable sort, so ties always keep the corpus order. Random seeds are fixed to 42 in Methods 1 and 2; Method 3 sets none, but its decoding is greedy and therefore deterministic.
- Generation is greedy in all methods (temperature 0, no sampling). Method 1 sets the repetition penalty to 1.0 and uses a fixed batch size of 8; Method 2 batches 16 prompts sorted by length; Method 3 batches 16 prompts with Qwen's default repetition penalty of 1.1. Method 1's Task 3 takes the LLM name, system prompt, token limits and batch size from `task2_results.json`, so only the retriever changes between Task 2 and Task 3.
- `requirements.txt` lists package names only. The versions used for the reported runs are listed in Appendix A of the report.

## 7. Troubleshooting

| Problem | Fix |
|---|---|
| `CUDA out of memory` | Lower `BATCH_SIZE` to 4 or 2 (in Task 2 or the combined notebook); this can change a few greedy answers |
| Task 3 stops with `task1_results.json not found` | Run Task 1 and Task 2 first, or copy the two files from `Output_Final/` (Section 3.3). In Colab, upload one file per prompt |
| An uploaded file was wrong | Delete it from the `outputs/` folder in the *Files* panel and run the cell again |
| `Can't find model 'en_core_web_sm'` | Run `python -m spacy download en_core_web_sm` |
| Warning about unauthenticated requests to the HF Hub | Harmless; set `HF_TOKEN` only if downloads are rate-limited |
| `IProgress not found` warning | Progress bars fall back to text; install `ipywidgets` for graphical ones |

## References

- P. Lewis, E. Perez, A. Piktus et al. (2020). *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks.* NeurIPS. arXiv:2005.11401
- V. Karpukhin et al. (2020). *Dense Passage Retrieval for Open-Domain Question Answering.* EMNLP.
- S. Robertson and H. Zaragoza (2009). *The Probabilistic Relevance Framework: BM25 and Beyond.* Foundations and Trends in Information Retrieval.
- G. V. Cormack, C. L. A. Clarke and S. Büttcher (2009). *Reciprocal Rank Fusion Outperforms Condorcet and Individual Rank Learning Methods.* SIGIR.
- Qwen Team (2024). *Qwen2.5 Technical Report.* arXiv:2412.15115
