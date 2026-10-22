"""Central settings. Change model names here only; model files and CPU tuning live in models.ini."""
import os
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent

# Ollama server — shared with the email bot, so there is only one LLM server to run.
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")

# Names are the section names in models.ini.
# Vision model reads charts once at ingestion. granite-vision is trained for chart-to-CSV.
VISION_MODEL = os.getenv("VISION_MODEL", "granite-vision-4.1-4b")
VISION_MODELS = ("granite-vision-4.1-4b",)
# Chooses lookups (tool calls) and writes answers from the evidence; never sees page images.
# Ollama tag, as listed by `ollama list` ("granite4.2:3b"). The old llama-server preset name
# "granite-4.2-3b" returns 404 from Ollama, which made every LLM lookup fail.
CHAT_MODEL = os.getenv("CHAT_MODEL", "granite4.2:3b")
CHAT_MODELS = (CHAT_MODEL,)
# Chat models that can reason before answering (optional in the UI; much slower on CPU).
REASONING_MODELS: tuple[str, ...] = (CHAT_MODEL,)
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")

DATA_DIR = Path(os.getenv("DATA_DIR", PROJECT_DIR / "data"))
PAGES_JSON = DATA_DIR / "pages.json"
PAGE_IMG_DIR = DATA_DIR / "pages"
CHROMA_DIR = DATA_DIR / "chroma"
COLLECTION = "pdf_pages"

# Document library row data lives in PostgreSQL (see core/db.py +
# documents/db_schema.py); block/series embedding vectors live in
# Weaviate (see documents/weaviate_store.py). No local SQLite file.
# Page images copied per document, since ingest.py reuses PAGE_IMG_DIR for each new PDF.
DOC_IMG_DIR = DATA_DIR / "docs"
METRICS_YAML = PROJECT_DIR / "metrics.yaml"
# Optional business glossary workbook. When set, matching terms and calculation templates expand
# retrieval queries; they are never treated as source evidence or report values.
_GLOSSARY_FILE = os.getenv("GLOSSARY_FILE", "").strip()
GLOSSARY_FILE = Path(_GLOSSARY_FILE).expanduser() if _GLOSSARY_FILE else None
GLOSSARY_SHEET = os.getenv("GLOSSARY_SHEET", "Glossary")
EMBED_DIM = 768  # nomic-embed-text
# Optional cross-encoder for passage search (a models.ini section name; "" turns it off).
# jina-reranker-v2 runs on CPU only, leaving the 8 GB of dev VRAM to gemma. Benchmarked on 71
# questions: hit@2 48 -> 50 over plain fusion at ~2.3 s/question CPU with 10 candidates cut to
# 1200 chars; bge-reranker scored 51 but took ~7.7 s.
# Disabled by default — Ollama has no rerank endpoint. Set to a model name
# only if running a local reranker server (e.g. jina-reranker-v2 via llama-server).
RERANK_MODEL = os.getenv("RERANK_MODEL", "")
RERANK_CANDIDATES = 10
RERANK_CHARS = 1200
# Passages and metric series put into one answer's evidence. Every item costs prompt tokens,
# and gemma reads ~26 tokens/s on CPU.
PASSAGES_K = 2
SERIES_K = 6

RENDER_DPI = 170
# Pages sent to the chat model per question. Fewer pages = shorter prompt = faster on CPU.
TOP_K = 2

# Tables scoring below this (0..1, see tables.validate) are re-read with the vision model.
TABLE_MIN_SCORE = float(os.getenv("TABLE_MIN_SCORE", "0.75"))
# Cap on one vision reply. At temperature 0 the model can fall into a repetition loop and
# generate for half an hour; a dense 25x10 table needs about 2,000 tokens.
VISION_MAX_TOKENS = int(os.getenv("VISION_MAX_TOKENS", "3000"))

# A dense dashboard can expose several chart-like grids as weak tables. Bound the expensive
# vision fallback per page; native/grid OCR remains available for every other region.
VISION_TABLE_MAX_REGIONS = int(os.getenv("VISION_TABLE_MAX_REGIONS", "2"))
