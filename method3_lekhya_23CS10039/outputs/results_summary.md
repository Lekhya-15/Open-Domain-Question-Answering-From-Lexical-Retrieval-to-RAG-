# Project 10: Open-Domain QA on rag-mini-wikipedia, results and analysis

These numbers are the outputs of [`../NLP_RAG.ipynb`](../NLP_RAG.ipynb) (all 918 questions, run on a Colab T4).

**Setup.**
- Corpus: 3,200 passages (`text-corpus`).
- Evaluation: all 918 questions (`question-answer`). 410 have yes/no gold answers and 508 have span answers.
- Generator: `Qwen/Qwen2.5-1.5B-Instruct` in fp16 on a Colab T4. Passages are truncated to 200 tokens, generation is capped at 64 new tokens, and decoding is greedy (temperature 0).
- BM25 tokenisation: lowercase alphanumeric tokens with scikit-learn's English stopwords removed.
- EM and F1 use SQuAD normalisation.
- recall@k uses answer-string containment over the 508 span-answer questions, because the dataset has no gold-passage labels.

## 1. Main results

| System | EM | F1 | span EM | span F1 | R@1 | R@3 | R@5 | R@10 |
|---|---|---|---|---|---|---|---|---|
| Task 1: BM25 (k1=1.5, b=0.75) + extractive | 0.413 | 0.459 | 0.098 | 0.182 | **0.478** | 0.571 | 0.602 | 0.630 |
| Task 2: Dense (MiniLM + FAISS cosine) + LLM | 0.550 | 0.641 | 0.295 | 0.460 | 0.398 | 0.528 | 0.589 | 0.632 |
| Task 3: Hybrid RRF (k=60) + LLM | **0.564** | **0.657** | **0.321** | **0.488** | 0.474 | **0.587** | **0.622** | **0.661** |

Yes/no EM: Task 1 scores 0.802 because it always answers "yes", which is the majority class. Tasks 2 and 3 both score 0.866.

## 2. Retrieval

* **BM25 beats dense at R@1 by 8 points (0.478 vs 0.398), and the two converge by R@10.** The questions were written while reading these passages, so they reuse exact names and rare words ("Gettysburg", "Grace Bedell"). That favours lexical matching. MiniLM is small, and it embeds only the first ~256 word pieces of each passage.
* **Dense retrieval helps when the question paraphrases the passage.** Example: "When did the Gettysburg address argue that America was born?" Dense rank 1 is the passage stating "…born not in 1789 but in 1776".
* **RRF is the best retriever from k=3 up**, gaining +1.6 at R@3, +2.0 at R@5 and +2.9 at R@10 over the better single retriever. BM25 and dense make different errors, and rank fusion rewards passages that both rank reasonably high.
* **RRF is slightly below BM25 at R@1** (0.474 vs 0.478). With equal weights, a confidently wrong dense top-1 can push BM25's correct passage down a rank.
* **Recall is a lower bound.** Containment misses long or paraphrased gold answers, and some questions depend on context ("Do they prefer forested areas?").

## 3. Answer quality

* **Generation is the main gain:** Task 1 → Task 2 adds +13.7 EM overall and +19.7 span EM.
  * The extractive reader sees only the top-1 passage. For example, the gold "18 months" was not in BM25's top-1 at all, but the LLM found it among the top 5.
  * The extractive reader also cannot tell which entity of the right type is meant. Task 1 answered "November 19 1863" where the gold is "1776".
* **Fusion adds a smaller, controlled gain:** Task 2 → Task 3 gives +1.4 EM and +2.6 span EM. The generator, prompt and settings are identical, so the gain is due only to retrieval (R@5 0.589 → 0.622).
* **EM understates the LLM.** Correct answers in a different surface form score EM 0: "Grace Bedell" vs "11-year-old Grace Bedell" (F1 0.8), and "United States Note" vs a full-sentence gold answer (F1 0.46).
* **Task 1 error sources:**
  * retrieval misses
  * word-form mismatches (no stemming: "establish" vs "established")
  * a crude answer-type rule that picks the first date or name in the sentence

## 4. Groundedness check

**Check:** overlap is the share of the answer's content words found in its 5 retrieved passages. For yes/no answers it is the share of the question's content words found in the best single passage. An answer is flagged when overlap < τ. An answer counts as correct when F1 ≥ 0.5.

| Answers | τ | check acc. | correct w/ high overlap | incorrect w/ low overlap | flagged | baseline: flag nothing |
|---|---|---|---|---|---|---|
| Task 2 | 0.25 | 0.674 | 0.990 | 0.101 | 39 | 0.645 |
| Task 2 | **0.5** | **0.678** | 0.961 | 0.163 | 76 | 0.645 |
| Task 2 | 0.75 | 0.658 | 0.856 | 0.298 | 182 | 0.645 |
| Task 2 | 1.0 | 0.602 | 0.731 | 0.368 | 279 | 0.645 |
| Task 3 | 0.25 | 0.675 | 0.988 | 0.067 | 28 | 0.660 |
| Task 3 | **0.5** | **0.680** | 0.975 | 0.106 | 48 | 0.660 |
| Task 3 | 0.75 | 0.664 | 0.870 | 0.266 | 162 | 0.660 |
| Task 3 | 1.0 | 0.612 | 0.744 | 0.356 | 266 | 0.660 |

* **The high-overlap side holds and the low-overlap side doesn't.** At τ = 0.5, 96–98% of correct answers have high overlap. But only 11–16% of incorrect answers have low overlap.
* **It barely beats the trivial baseline:** +3.3 points for Task 2 and +2.0 for Task 3 over "trust everything".
* **Why it fails: errors are grounded but wrong.** The LLM rarely invents text. It copies the wrong entity from the retrieved passages (e.g. "Kentucky" for "Hardin County"). Such answers have overlap 1.0, and lexical overlap cannot distinguish the right span from a wrong one inside the same evidence.
* **Why it flags correct answers:** surface-form mismatches ("7" vs "seven", "Christianity" vs "Christian") and paraphrased answers.
* **Raising τ trades precision for recall of errors without improving accuracy.** At τ = 1.0 the check catches 36–37% of errors but flags about 25% of correct answers, and accuracy falls below the baseline. τ = 0.5 is the best setting.

## 5. Conclusions

1. On this small, lexically aligned corpus, **BM25 is a stronger first-stage retriever than off-the-shelf MiniLM**. **RRF fusion is the best overall**, and gives the best end-to-end QA (EM 0.564, F1 0.657).
2. **Most of the improvement over the classical pipeline comes from generation**, and fusion adds a smaller, retrieval-attributable gain. Retrieval still caps performance: about 38% of span questions have no answer-bearing passage in the top 5.
3. **Lexical overlap is a weak groundedness signal.** It is a safe filter, because it rarely rejects good answers, but it misses most errors, because RAG errors are mostly grounded but wrong.

**Possible improvements:**
* stemming and number normalisation in both the BM25 tokenizer and the overlap check
* an NLI or entailment-based groundedness check
* a cross-encoder reranker or weighted RRF to fix R@1
* a larger generator
