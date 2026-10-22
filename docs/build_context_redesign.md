# build_context redesign

Redesign build_context to be robust and general instead of adding point fixes. No hard-coded words, example phrases, company names or question-specific rules anywhere. All thresholds come from config.py / .env. Show a plan and the diff first, and wait.

## Goal
Send the model enough context to answer any question about the emails, while keeping the prompt small enough for a CPU-only server (about 1,000-2,000 tokens). A change for one question must never reduce the quality for another, so results are checked by asking questions in the application. No golden test set, no eval script, no baseline run.

## Known bugs the design must remove
- Case A: short value lines are dropped by the 25-character minimum rule while their label line is kept.
- Case B: in the table path, a value line with more than 4 letters is treated as a label, then the row budget cuts the value and keeps the label.
Both disappear if a block is kept or dropped as one unit and is never split. Remove the table-path letter limit. Do not add a new point rule.

## Step 1 - Lossless cleaning (no relevance scoring here)
Per thread, remove only things that carry no information: quoted repeats of earlier messages (exact or near-duplicate blocks inside the thread and across the retrieved threads), mail headers (from/to/cc/sent/date/subject lines), signatures, address blocks, disclaimers, separator lines, and broken-encoding artifacts (decode as UTF-8; fix mojibake). Keep everything else byte-for-byte. Never split or rewrite text here.

## Step 2 - Block-based units
Split each cleaned thread into BLOCKS: paragraphs, bullet/numbered groups, and tables (consecutive short or number-heavy lines, or lines with | or tab). A block is atomic: it is kept whole or dropped whole. A label line stays with the value lines under it because they are in the same block. Never cut inside a block, except as the last resort in Step 3f.

## Step 3 - Selection
a. If the cleaned text of the top-N threads fits CHATBOT_CONTEXT_MAX_CHARS, send all of it unchanged, in original order. Nothing is dropped.
b. Otherwise rank blocks with a hybrid score: the existing BM25 score plus embedding similarity between the question and the block (reuse the existing embedding client; cache block vectors per request only). Add a small bonus for blocks with figures. Score a block by its best match against the whole question and against each sub-question when the question is multi-part (split on commas, "and", ";", numbering). This is generic, not rule-based.
c. Share the budget between threads in proportion to their relevance, with a minimum share per thread so none is dropped completely.
d. Within a thread, pick the best blocks and then add their neighbours (CHATBOT_CONTEXT_NEIGHBOR_BLOCKS, default 1) so surrounding context and labels come along. Keep the final blocks in original order, grouped under "[Email N: subject]".
e. Fill order when the budget runs short: tables and blocks that are the best match for any sub-question first, neighbours second, the rest last. Only the last tier is cut.
f. If a block is larger than the remaining budget, keep its first part and its lines containing figures, and log a WARNING with thread subject and what was cut. Never cut silently.

## Step 4 - Settings (config.py + .env, with defaults)
CHATBOT_CONTEXT_MAX_CHARS, CHATBOT_CONTEXT_TOP_THREADS, CHATBOT_CONTEXT_NEIGHBOR_BLOCKS, CHATBOT_CONTEXT_MIN_THREAD_SHARE, CHATBOT_CONTEXT_FIGURE_BONUS, CHATBOT_CONTEXT_USE_EMBEDDINGS (on/off, so it can fall back to BM25 only if embedding is slow). Keep existing values unchanged.

## Step 5 - Logging and debug dump
Per request log: threads kept, blocks kept/total, characters before cleaning, after cleaning and sent, whether "everything fit" (3a) or selection was used (3b), and any cut WARNING. When CHATBOT_DEBUG_PROMPT=1 the debug dump lists the same information and the chosen blocks.

## Step 6 - Tests
Only small synthetic unit tests. One must be shaped like the failing thread: a heading, a label (about 30 characters), a value line like "<number>+ <3 words> <1 word>" (more than 4 letters), a quoted line, another label, then a value line starting with "<number>%". Assert that every label is kept together with its value, and that no block is ever split.

## Rules
Do not change retrieval queries, the Redis cache, the system prompt, or the hallucination regex. Do not commit .env or real email data. Do not create a golden set, eval script or shared helper module. Show the plan and diff first and wait for approval.