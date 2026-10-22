"""
chatbot/prompts.py
=====================
Prompt templates and constants used by the RAG-based chatbot.

Contains the LLM system prompt for grounded email Q&A, the greeting
response template, and the greeting detection word set.
"""

# ── LLM system prompt for RAG answers ────────────────────────
SYSTEM_PROMPT = (
    "You are an email assistant for RBL Bank's leadership team. "
    "Answer ONLY from the provided email context. Be concise and direct. "
    "If the answer is not in the context, say so. "
    "Mention which email (by subject) the information comes from. "
    "When the response contains multiple pieces of information, "
    "format them as bullet points using '•' for clarity."
)

# ── Greeting response (no LLM call needed) ───────────────────
GREETING_RESPONSE = (
    "Hello! I'm your Email Assistant. I have access to your emails and can help you with:\n\n"
    "• **Finding specific information** from your emails\n"
    "• **Summarizing** email content\n"
    "• **Identifying** senders, recipients, dates, and attachments\n"
    "• **Answering questions** about the topics discussed\n\n"
    "Go ahead, ask me anything about your emails!"
)

# ── Greeting detection word set ──────────────────────────────
GREETINGS = {
    "hi", "hello", "hey", "good morning", "good afternoon", "good evening",
    "howdy", "greetings", "what's up", "whats up", "sup", "yo",
    "hi there", "hello there", "hey there",
    "hii", "hiii", "hiiii", "helloo", "hellooo", "heyyy", "heyy",
    "namaste", "namaskar", "good day",
    "how are you", "how r u", "wassup", "wazzup",
}
