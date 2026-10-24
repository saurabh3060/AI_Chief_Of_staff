"""
agent_email/schema.py
========================
Pydantic models for the Email & Summary agent.
 
Stores the shapes used to represent:
- `EmailMessage`: a normalized view of a single Graph API mail message.
- `ParsedAttachment`: an attachment plus its OCR-extracted text (if it
  was a supported document/image type).
- `ConversationThread`: the set of messages considered part of one
  "conversation" (in this simple pass, all messages sharing the same
  `conversation_key`, most recent N from the inbox).
- `EmailSummary`: the LLM-generated summary for a conversation, plus the
  action-item-relevant text extracted from it - this is what the Action
  & Follow-up Tracker agent reads via `agent_email.service`.
 
Internal to this agent's routes/service - other packages must go
through `agent_email.service` functions, not import these models to
reach into Email agent internals directly (except for the shared
`EmailSummary`/`ActionCandidate` shape that Action Tracker legitimately
consumes as this agent's public output).
"""
 
from datetime import datetime
 
from pydantic import BaseModel, Field, field_validator, model_validator
 
 
class EmailMessage(BaseModel):
    """A normalized view of a single Microsoft Graph API mail message."""
 
    id: str
    # Graph's stable per-thread id: every message in the same reply/
    # forward chain shares this value. Defaults to "" for callers/
    # fixtures that don't populate it.
    conversation_id: str = ""
    subject: str
    from_name: str | None = None
    from_address: str | None = None
    to_recipients: list[str] = Field(default_factory=list)  # List of 'to' email addresses
    cc_recipients: list[str] = Field(default_factory=list)  # List of 'cc' email addresses
    received_at: datetime
    has_attachments: bool = False
    body_text: str = ""
 
 
class ParsedAttachment(BaseModel):
    """An email attachment plus its OCR-extracted text, if applicable."""
 
    attachment_id: str
    name: str
    content_type: str | None = None
    extracted_text: str = ""
    ocr_applied: bool = False
    source_message_id: str | None = None
 
 
class ConversationThread(BaseModel):
    """A set of related messages (and their parsed attachments) considered one conversation."""
 
    thread_id: str
    subject: str
    messages: list[EmailMessage] = Field(default_factory=list)
    attachments: list[ParsedAttachment] = Field(default_factory=list)
 
 
class ActionCandidate(BaseModel):
    """
    A short, extracted action-item-shaped line from a summary (e.g.
    "Finance team to send the reconciled branch-wise breakup by
    Friday"). The Action & Follow-up Tracker agent classifies these
    further (overdue/in-progress) - this agent only extracts candidates,
    it does not classify them.
    """
 
    text: str
    source_message_id: str | None = None
 
 
class EmailItemSummary(BaseModel):
    """
    The per-email breakdown of a conversation thread's summary: one
    entry per message in the thread, each with its OWN summary, action
    items, and top questions (rather than one summary blended across
    every message in the thread).
    """
 
    message_id: str
    subject: str
    from_name: str | None = None
    from_address: str | None = None
    received_at: datetime
    summary_text: str
    action_items: list[str] = Field(default_factory=list)
    top_questions: list[str] = Field(default_factory=list)
 
 
class EmailSummary(BaseModel):
    """
    The Email & Summary agent's public output for one conversation
    thread: a generated summary plus extracted action-item candidates.
    This is the object persisted via storage and read by
    `agent_action_tracker` through `agent_email.service.get_latest_summaries()`.
 
    `email_summaries` holds the per-individual-email breakdown (the
    primary output going forward). `summary_text`/`action_items`/
    `top_questions` remain as a thread-level rollup (a concatenation of
    the per-email results) for backward compatibility with existing
    callers/UI that expect a single flat view of the thread.
    """
 
    thread_id: str
    subject: str
    generated_at: datetime
    email_summaries: list[EmailItemSummary] = Field(default_factory=list)
    summary_text: str
    action_candidates: list[ActionCandidate] = Field(default_factory=list)
    action_items: list[str] = Field(default_factory=list)  # Crisp action items from LLM
    top_questions: list[str] = Field(default_factory=list)  # Questions from LLM
   
    def model_dump_with_json(self, **kwargs):
        """Return a JSON-serializable dict with datetime converted to ISO format."""
        data = self.model_dump(**kwargs)
        if isinstance(data.get('generated_at'), str):
            pass  # Already serialized
        else:
            data['generated_at'] = self.generated_at.isoformat()
        return data
 
 
class EmailCandidate(BaseModel):
    """One relevance-filtered mailbox message entering the cluster pipeline."""
 
    message: EmailMessage
    relevance_score: float = 0.0
    token_score: float = 0.0
    semantic_score: float = 0.0
    attachment_text: str = ""
    topic_key: str = ""
    # Structured semantics extracted from subject + FULL body (not subject
    # alone). `topic_key` is derived from `business_objective` for the
    # inspection CSV / display; the lists capture the distinct sub-points a
    # single multi-topic email carries.
    business_objective: str = ""
    topics: list[str] = Field(default_factory=list)
    entities: list[str] = Field(default_factory=list)
    actions: list[str] = Field(default_factory=list)
    subtopics: list[str] = Field(default_factory=list)
    decisions: list[str] = Field(default_factory=list)
    context: str = ""
    # Meeting ids the relevance matcher accepted. Anchoring files the email
    # under these ids. Empty when the matcher named nothing.
    matched_meeting_ids: list[str] = Field(default_factory=list)
    # JSON list of the one email vector built at the gate. Clustering reuses
    # it instead of embedding the email a second time.
    embedding_json: str = ""
    # Meeting-anchored clustering: id/start/end of the PAST meeting occurrence
    # this email was mapped to (empty when it anchors to no meeting and is
    # clustered purely by semantics). `has_mom` marks a minutes/MoM-bearing
    # mail, used as a strong occurrence anchor.
    meeting_id: str = ""
    meeting_start: str = ""
    meeting_end: str = ""
    has_mom: bool = False
 
 
class EmailTopicRecord(BaseModel):
    """One row of the `email_topics` table: per-mail structured semantics.
 
    Written once at the relevance stage (the single extractor) and read back
    by clustering, so the former second topic-tag LLM call is gone. The
    `topic_key` is still derived from `business_objective` for display/CSV;
    the list fields carry the full extracted business context that the
    clustering embedding consumes.
    """
 
    message_id: str = ""
    subject: str = ""
    from_address: str = ""
    received_at: str = ""
    topic_key: str = ""
    run_id: str = ""
    business_objective: str = ""
    topics: list[str] = Field(default_factory=list)
    entities: list[str] = Field(default_factory=list)
    actions: list[str] = Field(default_factory=list)
    subtopics: list[str] = Field(default_factory=list)
    decisions: list[str] = Field(default_factory=list)
    context: str = ""
    department_score: float = 0.0
    department_margin: float = 0.0
 
    @field_validator("department_score", "department_margin", mode="before")
    @classmethod
    def _coerce_score(cls, value):
        """
        Coerce a blank/malformed numeric cell to 0.0 rather than failing
        the whole row. Some legacy/corrupted CSV rows have shifted columns
        (e.g. an empty string or the literal "embedding" landing in a score
        field); treating those as 0.0 keeps the row usable and stops the
        per-read log spam from the cluster record reader.
        """
        if value is None:
            return 0.0
        if isinstance(value, str):
            stripped = value.strip()
            if not stripped:
                return 0.0
            try:
                return float(stripped)
            except ValueError:
                return 0.0
        return value
 
 
class ClusterEmailRef(BaseModel):
    """Lightweight reference to a source email inside a cluster.
 
    ``message_id`` is the stable Microsoft Graph message id of the source
    email. It is the only field that can resolve a cluster membership back
    to an actual Graph message (subject/from_address/date are display-grade
    only), so it is the foundation for any future deleted-email
    reconciliation.
 
    It defaults to ``""`` and is OPTIONAL on purpose:
      * Backward compatibility - cluster rows written before this field
        existed serialize their refs without a ``message_id`` key, and
        those legacy refs must still deserialize cleanly (Pydantic fills
        the default).
      * Never fabricated - a blank ``message_id`` means "unknown / not
        captured", and callers must treat it as "cannot probe", NOT as a
        synthesized or guessed id.
    """
 
    message_id: str = ""
    subject: str = ""
    from_address: str = ""
    date: str = ""
 
 
class EmailClusterRecord(BaseModel):
    """
    One synthesized topic cluster (one `email_clusters` row) with
    precomputed summary, questions, action items, and embedding for
    real-time meeting matching.
    """
 
    cluster_id: str
    department: str
    topic: str = ""
    summary: str = ""           # "|"-joined bullet points
    top_questions: str = ""     # "|"-joined
    action_items: str = ""      # "|"-joined "<owner>: <task> (by <deadline>)"
    email_refs: str = ""        # JSON list of ClusterEmailRef dicts
    email_count: int = 0
    cluster_embedding: str = ""  # JSON float array
    # Meeting-anchored clustering: the PAST meeting occurrence this cluster
    # represents. For recurring series this holds the per-occurrence tag
    # (e.g. "banking+review+wholesale|2026-07") so each occurrence is a
    # distinct row and the one-active-per-meeting invariant does not force
    # sibling periods to collapse. Empty for non-meeting (semantic-only)
    # clusters. Folded into `cluster_id` so each occurrence of a recurring
    # series (April MBR, May MBR, ...) is a distinct row / UI line item.
    meeting_id: str = ""
    # The RAW Microsoft Graph meeting occurrence id this cluster anchored
    # to. Unlike `meeting_id` (which may be a per-occurrence series tag),
    # this always carries the real calendar id, so a live meeting arriving
    # with its Graph id can resolve back to this cluster during pre-meeting
    # brief / meeting-match. Empty for semantic-only clusters.
    anchored_meeting_id: str = ""
    meeting_start: str = ""
    # Active-vs-historical flag. 1 = this is the current/live cluster for
    # its meeting; 0 = superseded by a newer cluster for the same meeting
    # (kept in PostgreSQL for history, dropped from Weaviate/Redis). New
    # records default to active.
    is_active: int = 1
    updated_at: str = ""
    # Pipeline/publish run that last wrote this record's content. Not part of
    # the content hash, so it never forces a rewrite on its own.
    run_id: str = ""


class EmailSummarizerTestRequest(BaseModel):
    """Select either one Graph message or the latest N inbox messages."""
 
    message_id: str | None = None
    latest_n: int | None = Field(default=None, ge=1, le=50)
 
    @model_validator(mode="after")
    def validate_selector(self):
        if (self.message_id is None) == (self.latest_n is None):
            raise ValueError("Provide exactly one of 'message_id' or 'latest_n'.")
        if self.message_id is not None and not self.message_id.strip():
            raise ValueError("'message_id' cannot be blank.")
        return self
 
 
class EmailSummarizerParsedResult(BaseModel):
    summary_text: str = ""
    action_items: list[str] = Field(default_factory=list)
    top_questions: list[str] = Field(default_factory=list)
 
 
class EmailSummarizerQualityReport(BaseModel):
    passed: bool
    checks: dict[str, bool] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
    grounding_requires_human_review: bool = True
 
 
class EmailSummarizerTestResult(BaseModel):
    timestamp: datetime
    message_id: str
    email_subject: str
    from_name: str | None = None
    from_address: str | None = None
    received_at: datetime
    input_preview: str
    attachment_types: list[str] = Field(default_factory=list)
    raw_llm_response: str | None = None
    parsed_result: EmailSummarizerParsedResult | None = None
    quality: EmailSummarizerQualityReport
    parse_success: bool
    attempts: int
    error: str | None = None
 
 
class EmailSummarizerTestResponse(BaseModel):
    requested: int
    processed: int
    succeeded: int
    failed: int
    log_file: str
    results: list[EmailSummarizerTestResult] = Field(default_factory=list)
 
class MeetingNoteCreate(BaseModel):
    """Input model for creating a new meeting note."""
    meeting_id: str = Field(..., description="The meeting ID to attach this note to")
    note_text: str = Field(..., min_length=1, description="The actual CEO notes")
    action_items: list[dict] = Field(default_factory=list, description="Extracted action items (can be empty, LLM will extract)")
    tags: list[str] = Field(default_factory=list, description="Team/department tags (@mentions) the note was directed at, e.g. ['Risk', 'BB&RL']")
    source: str = Field(default="ceo_note", description="Source identifier for the note")
 
 
class MeetingNote(BaseModel):
    """Model representing a meeting note record."""
    id: int
    meeting_id: str
    note_text: str
    action_items: list[dict]
    tags: list[str] = Field(default_factory=list)
    source: str
    created_at: str
    updated_at: str | None = None
 
 
class MeetingNoteResponse(BaseModel):
    """Response model for meeting notes endpoints."""
    notes: list[MeetingNote]
    count: int
 
