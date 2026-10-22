"""
eml_summarizer
===============
Module for parsing `.eml` files from the local storage folder and
summarising their content via the Ollama LLM (llama3.2-vision).

Public API:
    - ``parse_eml_file(filename)`` – parse a `.eml` into a structured dict.
    - ``list_eml_files()``         – list all `.eml` files in the storage dir.
    - ``summarise_eml(filename)``  – parse + summarise in one call.
"""
