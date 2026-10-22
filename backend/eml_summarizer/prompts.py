"""
eml_summarizer/prompts.py
============================
Prompt templates for EML email summarisation via Ollama.
Prompts are strict and grounding-focused to prevent hallucination.
"""

EML_SUMMARY_SYSTEM_PROMPT = (
    "You are a factual email summariser. You MUST follow these rules strictly:\n"
    "1. ONLY use information that is explicitly written in the email below.\n"
    "2. Do NOT add any information that is not in the email.\n"
    "3. Do NOT guess, assume, or infer anything beyond what is written.\n"
    "4. If the email body is short or lacks detail, summarise only what is there.\n"
    "5. NEVER say 'there is no summary' — always summarise whatever content exists, even if it is just one line.\n"
    "6. Use simple, clear language."
)


def build_eml_summary_prompt(parsed_email: dict) -> str:
    """
    Build the prompt that asks the model to summarise a parsed email.

    Args:
        parsed_email: dict with keys filename, subject, from, to, cc,
                      date, body — as returned by ``parser.parse_eml_file``.
    """
    return (
        f"Summarise ONLY the following email. Do NOT add any outside information.\n\n"
        f"From: {parsed_email['from']}\n"
        f"To: {parsed_email['to']}\n"
        f"CC: {parsed_email.get('cc', '')}\n"
        f"Date: {parsed_email['date']}\n"
        f"Subject: {parsed_email['subject']}\n\n"
        f"--- Email Body Start ---\n"
        f"{parsed_email['body']}\n"
        f"--- Email Body End ---\n\n"
        f"Write a summary covering ONLY what is in the email above:\n"
        f"1. Who sent it and to whom\n"
        f"2. Main topic\n"
        f"3. Key points mentioned in the body\n"
        f"4. Any action items or requests mentioned\n"
        f"5. Any dates or deadlines mentioned\n\n"
        f"If a point has no information in the email, skip it. Do NOT make anything up."
    )


# Prompt used to extract the .eml filename from a user's natural-language
# question (e.g. "summarise report_q3.eml" -> "report_q3.eml").
EML_FILENAME_EXTRACTION_SYSTEM_PROMPT = (
    "You extract the .eml filename from a user message. "
    "Respond with ONLY the filename (including the .eml extension), nothing else. "
    "Do not add quotes, explanation, or any other text. "
    "If the user did not mention a specific .eml filename, respond with exactly: none"
)


def build_filename_extraction_prompt(question: str, available_files: list[str]) -> str:
    """
    Build the prompt that asks the model to identify which `.eml` file
    the user is referring to.
    """
    files_list = "\n".join(f"- {f}" for f in available_files)
    return (
        f"Available .eml files:\n{files_list}\n\n"
        f"User message: \"{question}\"\n\n"
        f"Which .eml file is the user referring to? "
        f"Respond with ONLY the exact filename from the list above. Nothing else."
    )
