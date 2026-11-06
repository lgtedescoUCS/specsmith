"""Offline evaluation of the extraction pipeline.

Pure code: the dataset schema and the field-level scoring. Nothing here
calls an LLM or touches the database; the runner in eval/run_eval.py does
the calling and hands the resulting specs to this package.
"""
