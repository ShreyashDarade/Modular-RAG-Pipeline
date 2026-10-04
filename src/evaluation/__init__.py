"""Offline evaluation of retrieval and answer quality.

A labelled *dataset* (JSONL) is run through the real pipeline; per-query *metrics* (hit/recall/
precision@k, MRR, nDCG, plus optional LLM-judged faithfulness and correctness) are aggregated with
bootstrap confidence intervals, saved as a *report*, and compared between configurations with a
paired test - so "the cross-encoder helps" is a measured, falsifiable statement. Nothing here is
imported by the serving path.
"""
