# Rewrite progress

**17 / 116 sections complete.**

Tracking the expansion of every section to the standard in [AUTHORING.md](AUTHORING.md).

Baseline readable word counts are from before the rewrite began. A section counts as done
only when it meets the standard *and* `node test/smoke.js` reports `errors : 0`.

## Part 0 — Start here

| | § | Section | File | Baseline words |
|---|---|---|---|---|
| x | 0.1 | `what-is-ml` | 10-onramp.js | 3246 |
| x | 0.2 | `linear-algebra-basics` | 10-onramp.js | 615 |
| x | 0.3 | `calculus-basics` | 10-onramp.js | 486 |
| x | 0.4 | `probability-basics` | 10-onramp.js | 427 |
| x | 0.5 | `python-toolkit` | 10-onramp.js | 653 |
| x | 0.6 | `notation` | 11-onramp-b.js | 999 |
| x | 0.7 | `matrix-calculus` | 11-onramp-b.js | 1185 |
| x | 0.8 | `first-model` | 11-onramp-b.js | 911 |

## Part 1 — Mathematical & statistical foundations

| | § | Section | File | Baseline words |
|---|---|---|---|---|
| x | 1.1 | `bayes` | 20-foundations.js | 458 |
| x | 1.2 | `distributions` | 20-foundations.js | 398 |
| x | 1.3 | `expectation` | 20-foundations.js | 264 |
| x | 1.4 | `concentration` | 20-foundations.js | 379 |
|   | 1.5 | `mle-map` | 20-foundations.js | 331 |
| x | 1.6 | `intervals` | 21-foundations-b.js | 974 |
| x | 1.7 | `causal` | 21-foundations-b.js | 616 |
| x | 1.8 | `linear-algebra` | 21-foundations-b.js | 406 |
| x | 1.9 | `calculus-ml` | 21-foundations-b.js | 457 |
| x | 1.10 | `information` | 21-foundations-b.js | 500 |
|   | 1.12 | `optimization` | 22-foundations-c.js | 1287 |
|   | 1.13 | `sampling` | 22-foundations-c.js | 858 |
|   | 1.14 | `bayesian-inference` | 22-foundations-c.js | 1011 |
|   | 1.15 | `numerics` | 22-foundations-c.js | 1075 |
|   | 1.16 | `part1-recall` | 21-foundations-b.js | 239 |

## Part 2 — Core & classical machine learning

| | § | Section | File | Baseline words |
|---|---|---|---|---|
|   | 2.1 | `supervised-setup` | 30-classical-a.js | 445 |
|   | 2.2 | `bias-variance` | 30-classical-a.js | 413 |
|   | 2.3 | `regularization` | 30-classical-a.js | 521 |
|   | 2.4 | `linear-logistic` | 30-classical-a.js | 468 |
|   | 2.5 | `knn-nb` | 30-classical-a.js | 479 |
|   | 2.6 | `svm` | 30-classical-a.js | 419 |
|   | 2.7 | `trees` | 31-classical-b.js | 424 |
|   | 2.8 | `boosting` | 31-classical-b.js | 844 |
|   | 2.9 | `unsupervised` | 31-classical-b.js | 774 |
|   | 2.10 | `pca` | 31-classical-b.js | 411 |
|   | 2.11 | `features` | 31-classical-b.js | 643 |
|   | 2.12 | `calibration` | 32-classical-c.js | 737 |
|   | 2.13 | `metrics` | 32-classical-c.js | 780 |
|   | 2.14 | `validation` | 32-classical-c.js | 349 |
|   | 2.15 | `hyperparameters` | 32-classical-c.js | 412 |
|   | 2.16 | `ensembles` | 33-classical-d.js | 848 |
|   | 2.17 | `interpretability` | 32-classical-c.js | 601 |
|   | 2.18 | `production` | 32-classical-c.js | 1022 |
|   | 2.19 | `fairness` | 32-classical-c.js | 861 |
|   | 2.21 | `gp-bayesopt` | 33-classical-d.js | 820 |
|   | 2.22 | `self-supervised` | 33-classical-d.js | 937 |
|   | 2.23 | `active-transfer` | 33-classical-d.js | 847 |
|   | 2.24 | `ranking` | 33-classical-d.js | 892 |
|   | 2.25 | `experimentation` | 33-classical-d.js | 977 |
|   | 2.26 | `part2-recall` | 32-classical-c.js | 177 |

## Part 3 — Neural networks & deep learning

| | § | Section | File | Baseline words |
|---|---|---|---|---|
|   | 3.1 | `nn-fundamentals` | 40-deep.js | 500 |
|   | 3.2 | `backprop` | 40-deep.js | 405 |
|   | 3.3 | `activations` | 40-deep.js | 468 |
|   | 3.4 | `initialisation` | 40-deep.js | 383 |
|   | 3.5 | `optimisers` | 40-deep.js | 456 |
|   | 3.6 | `normalisation` | 40-deep.js | 427 |
|   | 3.7 | `cnn` | 41-deep-b.js | 527 |
|   | 3.8 | `rnn` | 41-deep-b.js | 478 |
|   | 3.9 | `embeddings` | 41-deep-b.js | 474 |
|   | 3.11 | `autodiff` | 42-deep-c.js | 1016 |
|   | 3.12 | `training-dynamics` | 42-deep-c.js | 969 |
|   | 3.13 | `compression` | 42-deep-c.js | 1029 |
|   | 3.14 | `robustness` | 42-deep-c.js | 926 |
|   | 3.15 | `part3-recall` | 41-deep-b.js | 158 |

## Part 4 — LLMs & transformers

| | § | Section | File | Baseline words |
|---|---|---|---|---|
|   | 4.1 | `why-attention` | 50-transformers-a.js | 297 |
|   | 4.2 | `tokenization` | 50-transformers-a.js | 554 |
|   | 4.3 | `attention` | 50-transformers-a.js | 371 |
|   | 4.4 | `rope` | 50-transformers-a.js | 680 |
|   | 4.5 | `block` | 50-transformers-a.js | 425 |
|   | 4.6 | `architectures` | 50-transformers-a.js | 380 |
|   | 4.7 | `kv-cache` | 50-transformers-a.js | 590 |
|   | 4.8 | `moe` | 50-transformers-a.js | 343 |
|   | 4.9 | `pretraining` | 51-transformers-b.js | 498 |
|   | 4.10 | `scaling-laws` | 51-transformers-b.js | 369 |
|   | 4.11 | `distributed` | 51-transformers-b.js | 649 |
|   | 4.12 | `post-training` | 51-transformers-b.js | 882 |
|   | 4.13 | `lora` | 51-transformers-b.js | 666 |
|   | 4.14 | `serving` | 51-transformers-b.js | 790 |
|   | 4.15 | `decoding` | 51-transformers-b.js | 447 |
|   | 4.16 | `llm-eval` | 51-transformers-b.js | 574 |
|   | 4.17 | `safety` | 51-transformers-b.js | 695 |
|   | 4.19 | `multimodal` | 52-transformers-c.js | 945 |
|   | 4.20 | `reasoning` | 52-transformers-c.js | 889 |
|   | 4.21 | `structured-output` | 52-transformers-c.js | 1004 |
|   | 4.22 | `speculative` | 52-transformers-c.js | 936 |
|   | 4.23 | `part4-recall` | 51-transformers-b.js | 187 |

## Part 5 — RAG, agents, MCP, production

| | § | Section | File | Baseline words |
|---|---|---|---|---|
|   | 5.1 | `rag` | 60-applied.js | 1499 |
|   | 5.2 | `rag-vs-ft` | 60-applied.js | 239 |
|   | 5.3 | `agents` | 60-applied.js | 876 |
|   | 5.4 | `multi-agent` | 60-applied.js | 442 |
|   | 5.5 | `mcp` | 60-applied.js | 692 |
|   | 5.6 | `frameworks` | 60-applied.js | 557 |
|   | 5.7 | `production-ai` | 60-applied.js | 1174 |
|   | 5.9 | `chunking` | 61-applied-b.js | 928 |
|   | 5.10 | `vector-search` | 61-applied-b.js | 953 |
|   | 5.11 | `evals` | 61-applied-b.js | 965 |
|   | 5.12 | `mlops` | 61-applied-b.js | 815 |
|   | 5.13 | `decision-ladder` | 60-applied.js | 325 |

## Part 6 — Beyond the notebook

| | § | Section | File | Baseline words |
|---|---|---|---|---|
|   | 6.1 | `rl` | 70-frontier.js | 397 |
|   | 6.2 | `diffusion` | 70-frontier.js | 354 |
|   | 6.3 | `vae-gan` | 70-frontier.js | 419 |
|   | 6.4 | `gnn` | 70-frontier.js | 478 |
|   | 6.6 | `bandits` | 71-frontier-b.js | 1034 |
|   | 6.7 | `vision-tasks` | 71-frontier-b.js | 911 |
|   | 6.8 | `privacy` | 71-frontier-b.js | 1115 |
|   | 6.9 | `exclusions` | 70-frontier.js | 637 |

## Part 7 — The ML interview

| | § | Section | File | Baseline words |
|---|---|---|---|---|
|   | 7.1 | `interview-map` | 80-interview.js | 1291 |
|   | 7.2 | `interview-breadth` | 80-interview.js | 3314 |
|   | 7.3 | `interview-depth` | 80-interview.js | 1027 |
|   | 7.4 | `ml-system-design` | 80-interview.js | 1912 |
|   | 7.5 | `coding-round` | 81-interview-b.js | 1132 |
|   | 7.6 | `case-round` | 81-interview-b.js | 1461 |
|   | 7.7 | `behavioural` | 81-interview-b.js | 1452 |

## Reference & drill room

| | § | Section | File | Baseline words |
|---|---|---|---|---|
|   | R.1 | `numbers` | 90-reference.js | 514 |
|   | R.2 | `formulas` | 90-reference.js | 295 |
|   | R.3 | `glossary` | 90-reference.js | 77 |
|   | R.4 | `sources` | 90-reference.js | 781 |
|   | R.5 | `drill` | 90-reference.js | 62 |

