# Open-Domain Question Answering: From Lexical Retrieval to RAG

**Project 10 · Group 28 · Dataset: [`rag-datasets/rag-mini-wikipedia`](https://huggingface.co/datasets/rag-datasets/rag-mini-wikipedia)** (3,200 Wikipedia passages, 918 questions)

The project builds three question-answering systems and compares them with Exact Match, token F1 and Recall@k:

| Task | Retriever | Reader |
|---|---|---|
| 1 · Baseline | BM25 | rule-based extractive answer |
| 2 · RAG | MiniLM embeddings + FAISS | LLM generation (Qwen2.5-Instruct) |
| 3 · Hybrid | Reciprocal Rank Fusion of BM25 + dense | LLM generation + groundedness check |

Each team member built their own full implementation of all three tasks (a "method"):

| Name | Roll No. | Method | Folder |
|---|---|---|---|
| Nihar Ranjan Das | 26AI60R20 | Method 1 | [`method1_nihar_26AI60R20/`](method1_nihar_26AI60R20/) |
| Mehul Mathur | 23CS30033 | Method 2 | [`method2_mehul_23CS30033/`](method2_mehul_23CS30033/) |
| M Lekhya Reddy | 23CS10039 | Method 3 | [`method3_lekhya_23CS10039/`](method3_lekhya_23CS10039/) |

## Results (EM / F1 in %, all 918 questions)

| System | Method 1 | Method 2 | Method 3 |
|---|---|---|---|
| Task 1: BM25 + extractive | 1.9 / 5.5 | 2.0 / 5.0 | 41.3 / 45.9 * |
| Task 2: Dense + LLM | 56.4 / 67.1 | 54.5 / 64.3 | 55.0 / 64.1 |
| Task 3: Hybrid (RRF) + LLM | **57.2 / 68.4** | 56.0 / 66.0 | 56.4 / 65.7 |

\* Method 3's Task 1 answers "yes" to every yes/no question, which alone gives 80.2% EM on those questions.

The group report compares the three methods side by side: [`report/Project10_Group28_Report.pdf`](report/Project10_Group28_Report.pdf).

## Repository layout

```
.
├── report/
│   ├── Project10_Group28_Report.pdf          group report (all three methods)
│   └── Project10_Group28_Presentation.pdf
├── method1_nihar_26AI60R20/
│   ├── README.md                             detailed setup, dataset and run instructions
│   ├── requirements.txt
│   ├── Project10_OpenDomainQA_End_to_End.ipynb   all three tasks in one notebook
│   ├── Task_Final/                           Task 1–3 as separate notebooks
│   └── Output_Final/                         result JSONs, generation caches, plots
├── method2_mehul_23CS30033/
│   ├── README.md
│   ├── Project_Report.pdf
│   ├── requirements.txt
│   └── notebooks/Project10_Group28_23CS30033.ipynb
└── method3_lekhya_23CS10039/
    └── NLP_RAG.ipynb                         all three tasks in one Colab notebook
```

## Running

All notebooks were run on Google Colab with a T4 GPU. Open a notebook in Colab, set **Runtime → Change runtime type → T4 GPU**, and use **Runtime → Run all**. The first cells install anything Colab is missing. See the Method 1 README for local setup and more detail.
