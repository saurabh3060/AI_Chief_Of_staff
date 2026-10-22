"""
chatbot package
=================
RAG-based Email Chatbot Agent.

Responsibility: answer natural-language questions about email content
by retrieving relevant chunks from a Weaviate vector index built over
email messages and composing grounded answers via Ollama LLM.

The chatbot uses a three-step RAG pipeline:
1. Index email messages into Weaviate (done at startup or on demand).
2. For each query, embed it and retrieve the top-k most relevant chunks.
3. Format retrieved context and ask the LLM to compose a grounded answer.

Greeting detection is handled via regex-based pattern matching (no LLM
call needed for simple greetings).
"""
