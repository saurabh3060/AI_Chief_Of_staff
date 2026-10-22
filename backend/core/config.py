
"""
core/config.py
===============
Centralised configuration for the entire backend.

Every setting that can vary between environments (secrets, hosts, model
names, feature toggles, concurrency limits, etc.) is declared here as a
typed field on the `Settings` class and loaded from a single `.env` file
in the `backend/` directory via `pydantic-settings`.

No other module in this codebase should call `os.getenv(...)` directly -
instead, import `get_settings()` from this module and read the typed
attribute. This keeps configuration in one auditable place and gives us
validation errors at startup instead of silent `None`s scattered around
the code.
"""

import json
from datetime import timedelta, timezone
from functools import lru_cache
from pathlib import Path
from urllib.parse import quote

from pydantic import BaseModel, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Absolute path to the `backend/` directory (parent of this `core/` folder).
# Used so the app finds `.env` regardless of the current working directory
# the process was started from.
BACKEND_ROOT = Path(__file__).resolve().parent.parent

# India Standard Time (UTC+5:30). Shared across the calendar agent and the
# local calendar stand-in so the wall-clock offset is defined in one place.
IST = timezone(timedelta(hours=5, minutes=30))


class Department(BaseModel):
    """
    One of the fixed BU departments the system clusters emails and
    calendar meetings into. `name` is the canonical label the
    clustering LLM must emit and the calendar agent maps to; `summary`
    is a short description of the department's scope, used both to guide
    the LLM and as extra text-match signal for calendar->cluster mapping;
    `view` is the frontend sidebar/route key this department maps to
    (one of "CRO"/"Retail"/"Wholesale"/"RetailAssets"), so clicking a
    meeting that resolved to this department routes to the right tab.
    """

    name: str
    summary: str = ""
    view: str = ""


class Settings(BaseSettings):
    """
    Typed application settings loaded from `backend/.env`.

    Fields are grouped by concern (app metadata, Graph API, storage,
    Ollama, concurrency, retry, observability, vector store, logging).
    Graph secrets are required for `delegated` / `app_only` modes and
    optional when `GRAPH_AUTH_MODE=local` (offline .eml mailbox).
    """

    model_config = SettingsConfigDict(
        env_file=str(BACKEND_ROOT / ".env"),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ---- App metadata ----
    app_name: str = "RBL Bank Digital Office Backend"
    app_env: str = "development"
    app_host: str = "0.0.0.0"
    app_port: int = 8516

    # ---- Microsoft Graph API / Azure AD App Registration ----
    # Required for GRAPH_AUTH_MODE=delegated|app_only. Optional in
    # GRAPH_AUTH_MODE=local, which reads Outlook .eml files from disk
    # and never talks to Microsoft.
    tenant_id: str = Field(default="", description="Azure AD tenant ID")
    client_id: str = Field(default="", description="Azure AD app registration client ID")
    client_secret: str = Field(default="", description="Azure AD app registration client secret")
    app_base_url: str = "http://localhost:8516"
    mailbox_email: str = ""
    mailbox_type: str = "shared"
    graph_auth_mode: str = "delegated"  # "delegated" | "app_only" | "local"
    graph_request_timeout_seconds: int = 30
    # Trusted base URL for Microsoft Graph. This is the SINGLE source of the
    # scheme/host/port (origin) every outbound Graph request can reach - it
    # is read exclusively from this configuration (default below or the
    # GRAPH_BASE_URL env var) and NEVER derived from request input. It is
    # structurally validated at startup by `_validate_graph_base_url`
    # (HTTPS only, exactly graph.microsoft.com, default port, no embedded
    # credentials/query/fragment) so a misconfigured value fails loudly
    # here rather than letting a request reach an unexpected origin. The
    # '/v1.0' path pins the API version; override only to target a
    # different Graph API version on the same host.
    graph_base_url: str = Field(
        default="https://graph.microsoft.com/v1.0",
        alias="GRAPH_BASE_URL",
        validation_alias="GRAPH_BASE_URL",
        description="Trusted Microsoft Graph base URL (origin + API version).",
    )
    # ---- Email cluster synthesis pipeline ----
    # Number of most-recent shared-mailbox messages fetched from Graph
    # before the relevance filter narrows them to the candidate pool.
    email_pipeline_fetch_top: int = Field(
        default=100,
        alias="EMAIL_PIPELINE_FETCH_TOP",
        validation_alias="EMAIL_PIPELINE_FETCH_TOP",
    )
    # After scoring all fetched messages (token overlap + semantic vs
    # BU_DEPARTMENTS), keep only the top K for embedding/clustering.
    email_pipeline_candidate_pool_size: int = Field(
        default=20,
        alias="EMAIL_PIPELINE_CANDIDATE_POOL_SIZE",
        validation_alias="EMAIL_PIPELINE_CANDIDATE_POOL_SIZE",
    )
    # Optional HARD safety FLOOR (0-100) for the relevance junk gate. The
    # actual cutoff is derived per-run from the batch's score distribution
    # (see email_relevance_adaptive_cutoff_enabled); this floor only stops
    # the adaptive break from dropping BELOW a minimum you trust. The
    # adaptive break may rise above it. 0 = no floor (fully distribution
    # driven). Not a fixed pass/fail score on its own unless adaptive is off.
    email_relevance_min_score: float = Field(
        default=0.0,
        alias="EMAIL_RELEVANCE_MIN_SCORE",
        validation_alias="EMAIL_RELEVANCE_MIN_SCORE",
    )
    # Derive the junk cutoff from THIS run's score distribution (Otsu
    # natural break between the junk group and the relevant group) instead
    # of a fixed number, so it adapts to whatever mix of mail arrived. When
    # false, the gate uses email_relevance_min_score as a plain fixed cutoff.
    email_relevance_adaptive_cutoff_enabled: bool = Field(
        default=True,
        alias="EMAIL_RELEVANCE_ADAPTIVE_CUTOFF_ENABLED",
        validation_alias="EMAIL_RELEVANCE_ADAPTIVE_CUTOFF_ENABLED",
    )
    # Strictness (0-1) for the adaptive cutoff: 0 = pure Otsu break;
    # higher shifts the cutoff that fraction of the way from the break up
    # toward the batch's top score, keeping fewer/only-stronger mails. This
    # is a distribution-relative dial, NOT a hardcoded score.
    email_relevance_adaptive_strictness: float = Field(
        default=0.0,
        alias="EMAIL_RELEVANCE_ADAPTIVE_STRICTNESS",
        validation_alias="EMAIL_RELEVANCE_ADAPTIVE_STRICTNESS",
    )
    # Weight (0-1) for token-overlap vs semantic in the relevance filter;
    # remainder goes to semantic. Default 0.5/0.5 per spec.
    email_relevance_token_weight: float = Field(
        default=0.5,
        alias="EMAIL_RELEVANCE_TOKEN_WEIGHT",
        validation_alias="EMAIL_RELEVANCE_TOKEN_WEIGHT",
    )
    email_relevance_semantic_weight: float = Field(
        default=0.5,
        alias="EMAIL_RELEVANCE_SEMANTIC_WEIGHT",
        validation_alias="EMAIL_RELEVANCE_SEMANTIC_WEIGHT",
    )
    # Weight (0-1) for the MEETING-relevance term in the relevance filter:
    # how similar an email is to the current/upcoming calendar meeting(s).
    # This lets a mail that matches an actual scheduled meeting score high
    # even when it shares NO vocabulary with any BU department dictionary
    # (e.g. "Digital Steering Committee" has no branch/wholesale product
    # words but does match a "Digital Steering Committee" calendar entry).
    # Blended with token + semantic weights and renormalized so the three
    # always sum to 1. 0 = disabled (pure token+semantic, legacy behaviour);
    # also silently contributes nothing when no meeting history is loaded.
    email_relevance_meeting_weight: float = Field(
        default=0.0,
        alias="EMAIL_RELEVANCE_MEETING_WEIGHT",
        validation_alias="EMAIL_RELEVANCE_MEETING_WEIGHT",
    )
    # Only calendar occurrences whose meeting time falls within +/- this
    # many days of the pipeline run count as "current" for meeting-aware
    # relevance. Keeps the score anchored to what is actually on the
    # calendar now, not ancient history. 0 = no time filter (any occurrence
    # in the loaded history window contributes).
    email_relevance_meeting_window_days: float = Field(
        default=7.0,
        alias="EMAIL_RELEVANCE_MEETING_WINDOW_DAYS",
        validation_alias="EMAIL_RELEVANCE_MEETING_WINDOW_DAYS",
    )
    # Clustering algorithm: "components" (cosine graph + connected
    # components), "knn" (1-NN union-find), or "agglomerative"
    # (complete-linkage, hand-rolled). Switch via .env without code changes.
    email_cluster_algo: str = Field(
        default="components",
        alias="EMAIL_CLUSTER_ALGO",
        validation_alias="EMAIL_CLUSTER_ALGO",
    )
    # Cosine similarity threshold used by BOTH clustering algorithms:
    # graph-edge threshold for "components", 1-NN join floor for "knn".
    email_cluster_similarity_threshold: float = Field(
        default=0.75,
        alias="EMAIL_CLUSTER_SIMILARITY_THRESHOLD",
        validation_alias="EMAIL_CLUSTER_SIMILARITY_THRESHOLD",
    )
    # Second-pass recurring-series merge: if two emails sit just below the
    # cosine threshold but their *subject stems* still match (calendar
    # tokens / minutes / MoM stripped, abbreviations expanded), join them.
    # This is how a short follow-up like "Actions from June's WB Review"
    # joins the monthly-review cluster without pulling in a different
    # Wholesale thread (liabilities mix, CA/SA, TAT). Runs after every
    # clustering algorithm, including agglomerative.
    email_cluster_series_near_miss: float = Field(
        default=0.68,
        alias="EMAIL_CLUSTER_SERIES_NEAR_MISS",
        validation_alias="EMAIL_CLUSTER_SERIES_NEAR_MISS",
    )
    email_cluster_series_jaccard: float = Field(
        default=0.5,
        alias="EMAIL_CLUSTER_SERIES_JACCARD",
        validation_alias="EMAIL_CLUSTER_SERIES_JACCARD",
    )
    # Third-pass residual merge for emails that are NOT a recurring
    # series: connected-components at this lower cosine floor. 0 = skip.
    # Pulls related working-paper threads (NIM / CASA / balance-sheet)
    # together after the hard clustering pass, without unlocking the
    # monthly-review series.
    email_cluster_residual_threshold: float = Field(
        default=0.72,
        alias="EMAIL_CLUSTER_RESIDUAL_THRESHOLD",
        validation_alias="EMAIL_CLUSTER_RESIDUAL_THRESHOLD",
    )
    # Minimum cosine similarity required between two emails for
    # merge_quoted_subject_links to join them, even when one email's body
    # contains the other's subject as a quoted "Subject:" header. Without
    # this gate, a single forwarded compilation email can act as a hub
    # connecting many unrelated topics through quoted headers alone.
    # 0.0 = disabled (pure string-match behaviour, old default).
    email_cluster_cite_min_cosine: float = Field(
        default=0.50,
        alias="EMAIL_CLUSTER_CITE_MIN_COSINE",
        validation_alias="EMAIL_CLUSTER_CITE_MIN_COSINE",
    )
    # Max emails appended into one LLM synthesis call per cluster batch.
    email_cluster_max_size: int = Field(
        default=10,
        alias="EMAIL_CLUSTER_MAX_SIZE",
        validation_alias="EMAIL_CLUSTER_MAX_SIZE",
    )
    # Department assignment: if the top two prototype scores (0-100)
    # differ by this much or less, break the tie with token overlap
    # against each department's own dictionary text. 0 = never tie-break.
    # Not a per-subject rule — any close pair uses the same margin.
    email_dept_assignment_margin: float = Field(
        default=3.0,
        alias="EMAIL_DEPT_ASSIGNMENT_MARGIN",
        validation_alias="EMAIL_DEPT_ASSIGNMENT_MARGIN",
    )
    # Minimum nearest-prototype score (0-100) to assign a department.
    # Below this the cluster is left unlabeled rather than forced to the
    # nearest book.
    email_dept_assignment_min_score: float = Field(
        default=40.0,
        alias="EMAIL_DEPT_ASSIGNMENT_MIN_SCORE",
        validation_alias="EMAIL_DEPT_ASSIGNMENT_MIN_SCORE",
    )
    # ------------------------------------------------------------------
    # Meeting department identification (upcoming/history occurrences).
    #
    # A meeting is tagged with a department from its OWN embedding using the
    # same nearest-prototype scoring the clusters use, but with SEPARATE,
    # independently tunable thresholds. Mislabeling a meeting is worse than
    # not labeling it (a wrong label would prune away the right clusters),
    # so these default stricter than the cluster thresholds and the filter
    # only engages behind the confidence gate below.
    #
    # Tie-break margin (0-100): if the top two prototype scores differ by
    # this much or less, break the tie with department token overlap.
    meeting_dept_assignment_margin: float = Field(
        default=3.0,
        alias="MEETING_DEPT_ASSIGNMENT_MARGIN",
        validation_alias="MEETING_DEPT_ASSIGNMENT_MARGIN",
    )
    # Confidence gate: minimum nearest-prototype score (0-100) for a meeting
    # to be assigned a department CONFIDENTLY. Below this the meeting is left
    # unlabeled and NO department filtering is applied to its candidate pool
    # (we fall back to the full-pool behaviour rather than filter on a guess).
    # Stricter than the cluster floor on purpose.
    meeting_dept_assignment_min_score: float = Field(
        default=50.0,
        alias="MEETING_DEPT_ASSIGNMENT_MIN_SCORE",
        validation_alias="MEETING_DEPT_ASSIGNMENT_MIN_SCORE",
    )
    # Embedding input caps for clustering (subject repeated below for weight).
    # 0 = no cap. These do NOT apply to cluster synthesis, which uses the
    # synth caps below (default unlimited so the LLM sees full mail+MoM).
    email_embed_body_max_chars: int = Field(
        default=0,
        alias="EMAIL_EMBED_BODY_MAX_CHARS",
        validation_alias="EMAIL_EMBED_BODY_MAX_CHARS",
    )
    email_embed_attachment_max_chars: int = Field(
        default=8000,
        alias="EMAIL_EMBED_ATTACHMENT_MAX_CHARS",
        validation_alias="EMAIL_EMBED_ATTACHMENT_MAX_CHARS",
    )
    email_embed_subject_copies: int = Field(
        default=2,
        alias="EMAIL_EMBED_SUBJECT_COPIES",
        validation_alias="EMAIL_EMBED_SUBJECT_COPIES",
    )
    # Build the clustering embedding from the LLM-extracted business
    # semantics (business objective + topic + subtopics + entities +
    # actions/decisions) instead of raw subject/body alone. This makes the
    # embedding capture the complete business CONTEXT of the email, so two
    # emails that share only a broad topic but drive different objectives do
    # NOT cluster together. Requires topic tagging (which populates those
    # fields) to have run first; falls back to the raw subject/body/
    # attachment text for any email whose semantics are empty (tagging
    # disabled or the extractor returned nothing). Set false to restore the
    # previous raw-text embedding.
    email_embed_semantic_enabled: bool = Field(
        default=True,
        alias="EMAIL_EMBED_SEMANTIC_ENABLED",
        validation_alias="EMAIL_EMBED_SEMANTIC_ENABLED",
    )
    # How many times the business objective is repeated in the semantic
    # embedding text. The objective is the DISCRIMINATOR that keeps
    # same-topic/different-objective emails apart, so it is weighted more
    # heavily than the topic/subtopic/entity/action lines. 1 = no extra
    # weight.
    email_embed_objective_weight: int = Field(
        default=2,
        alias="EMAIL_EMBED_OBJECTIVE_WEIGHT",
        validation_alias="EMAIL_EMBED_OBJECTIVE_WEIGHT",
    )
    # Chars of raw body text appended as grounding CONTEXT after the
    # structured semantic lines when semantic embedding is on. Kept small so
    # the structured objective/topic/entity signal dominates the centroid
    # while still anchoring the embedding to concrete source wording. 0 =
    # omit the raw body entirely (pure-semantic embedding).
    email_embed_semantic_body_context_chars: int = Field(
        default=600,
        alias="EMAIL_EMBED_SEMANTIC_BODY_CONTEXT_CHARS",
        validation_alias="EMAIL_EMBED_SEMANTIC_BODY_CONTEXT_CHARS",
    )
    # NOTE: the former EMAIL_TOPIC_TAGGING_ENABLED / EMAIL_TOPIC_TAG_BODY_MAX_CHARS
    # settings were removed with the second topic-tag LLM stage. Structured
    # semantics are now extracted ONCE at the relevance stage and reused by
    # clustering (see agent_email/relevance.py + clustering.tag_candidates_with_topics),
    # so there is no separate topic-tag call to toggle or cap.

    # Caps for the cluster-synthesis LLM prompt. 0 = unlimited (full
    # stripped body + full extracted MoM text). Set a positive value only
    # if a host cannot take the full cluster in one call.
    email_synth_body_max_chars: int = Field(
        default=0,
        alias="EMAIL_SYNTH_BODY_MAX_CHARS",
        validation_alias="EMAIL_SYNTH_BODY_MAX_CHARS",
    )
    email_synth_attachment_max_chars: int = Field(
        default=0,
        alias="EMAIL_SYNTH_ATTACHMENT_MAX_CHARS",
        validation_alias="EMAIL_SYNTH_ATTACHMENT_MAX_CHARS",
    )
    # Real-time meeting -> precomputed cluster match (0-100 cosine scale).
    meeting_cluster_match_threshold: float = Field(
        default=55.0,
        alias="MEETING_CLUSTER_MATCH_THRESHOLD",
        validation_alias="MEETING_CLUSTER_MATCH_THRESHOLD",
    )
    # After occurrence-split, a monthly series is many cluster rows.
    # The meeting cosine gate only *selects the series* (top hit must
    # clear `meeting_cluster_match_threshold`). Sibling occurrence rows
    # are then included by recurring-series stem, even when an
    # action-only month sits below that meeting-cosine floor.
    meeting_cluster_series_retrieve_enabled: bool = Field(
        default=True,
        alias="MEETING_CLUSTER_SERIES_RETRIEVE_ENABLED",
        validation_alias="MEETING_CLUSTER_SERIES_RETRIEVE_ENABLED",
    )
    # ---- Business-context recurrence & related-cluster retrieval ----
    # Recurrence (recurring vs one-off) and one-off related-cluster
    # retrieval are decided from repeated BUSINESS CONTEXT across history,
    # not from subject-stem word overlap. Subject stems are still used for
    # email clustering, but never as the primary live-meeting recurrence
    # decision.
    #
    # Stage-1 semantic candidate retrieval: cosine (0-100) between the seed
    # cluster embedding and each other cluster embedding must clear this
    # floor to become a candidate. The actual working cut is derived from
    # the batch distribution (Otsu) and clamped to this floor, mirroring
    # the email relevance gate.
    meeting_related_candidate_min_score: float = Field(
        default=45.0,
        alias="MEETING_RELATED_CANDIDATE_MIN_SCORE",
        validation_alias="MEETING_RELATED_CANDIDATE_MIN_SCORE",
    )
    # Tuning dial applied ON TOP OF the adaptive (Otsu) cutoff used to select
    # which candidate clusters are relevant enough to a meeting. The adaptive
    # method still determines the base cutoff from the actual candidate-score
    # distribution; this value only nudges that break up or down (in 0-100
    # cosine points) so operators can tighten or loosen relevance WITHOUT
    # hard-coding a fixed threshold.
    #   > 0  -> higher effective cutoff -> fewer, more closely matched clusters
    #   < 0  -> lower  effective cutoff -> more clusters allowed through
    #   = 0  -> pure adaptive break (recommended starting point; tune from
    #           real meeting results rather than guessing a new threshold).
    # The effective cutoff is still clamped to meeting_related_candidate_min_score.
    meeting_related_adaptive_cutoff_adjustment: float = Field(
        default=0.0,
        alias="MEETING_RELATED_ADAPTIVE_CUTOFF_ADJUSTMENT",
        validation_alias="MEETING_RELATED_ADAPTIVE_CUTOFF_ADJUSTMENT",
    )
    meeting_related_candidate_max: int = Field(
        default=12,
        alias="MEETING_RELATED_CANDIDATE_MAX",
        validation_alias="MEETING_RELATED_CANDIDATE_MAX",
    )
    # Stage-2 business-context relevance gate: run the strict LLM
    # relevance classifier over the stage-1 candidates and keep only the
    # clusters it judges materially relevant to the meeting's business
    # purpose. When disabled, stage-1 semantic candidates are kept as-is.
    meeting_related_relevance_gate_enabled: bool = Field(
        default=True,
        alias="MEETING_RELATED_RELEVANCE_GATE_ENABLED",
        validation_alias="MEETING_RELATED_RELEVANCE_GATE_ENABLED",
    )
    # Recurrence evidence: a meeting is RECURRING only when it has repeated
    # business-context evidence across MULTIPLE historical clusters that are
    # separated across DIFFERENT periods. A single similar cluster, or
    # several clusters all in the same period, is NOT sufficient — prefer
    # ONE-OFF when the evidence is weak or ambiguous.
    #
    # Minimum number of distinct, materially-relevant historical clusters
    # (excluding the seed) required to call a meeting recurring.
    meeting_recurrence_min_occurrences: int = Field(
        default=2,
        alias="MEETING_RECURRENCE_MIN_OCCURRENCES",
        validation_alias="MEETING_RECURRENCE_MIN_OCCURRENCES",
    )
    # Minimum number of DISTINCT calendar periods those relevant clusters
    # must span (so repeated context over time, not many clusters in one
    # month, is what evidences an ongoing activity).
    meeting_recurrence_min_distinct_periods: int = Field(
        default=2,
        alias="MEETING_RECURRENCE_MIN_DISTINCT_PERIODS",
        validation_alias="MEETING_RECURRENCE_MIN_DISTINCT_PERIODS",
    )
    # ---- Meeting-anchored email clustering ----
    # When enabled, past calendar occurrences are embedded and each
    # candidate email is mapped to the most recent prior matching
    # meeting whose end time is before the email arrived.
    meeting_anchored_clustering_enabled: bool = Field(
        default=True,
        alias="MEETING_ANCHORED_CLUSTERING_ENABLED",
        validation_alias="MEETING_ANCHORED_CLUSTERING_ENABLED",
    )
    meeting_history_lookback_days: int = Field(
        default=60,
        alias="MEETING_HISTORY_LOOKBACK_DAYS",
        validation_alias="MEETING_HISTORY_LOOKBACK_DAYS",
    )
    meeting_history_fetch_top: int = Field(
        default=250,
        alias="MEETING_HISTORY_FETCH_TOP",
        validation_alias="MEETING_HISTORY_FETCH_TOP",
    )
    # Re-pull calendar history at most once per N hours (0 = once per process).
    meeting_history_refresh_hours: float = Field(
        default=24.0,
        alias="MEETING_HISTORY_REFRESH_HOURS",
        validation_alias="MEETING_HISTORY_REFRESH_HOURS",
    )
    meeting_anchor_semantic_threshold: float = Field(
        # Raised from 0.62 → 0.72: a cosine of 0.62 was too permissive and
        # caused Digital Steerco / ISSC emails to anchor to the nearest
        # "Wholesale Banking MBR" occurrence when no dedicated calendar entry
        # existed for those meetings. At 0.72 only genuinely high-similarity
        # pairs anchor; everything else stays unanchored and clusters by
        # series-stem logic instead, which is correct behaviour.
        default=0.72,
        alias="MEETING_ANCHOR_SEMANTIC_THRESHOLD",
        validation_alias="MEETING_ANCHOR_SEMANTIC_THRESHOLD",
    )
    meeting_anchor_mom_semantic_threshold: float = Field(
        # Raised from 0.50 → 0.68: MoM emails were anchoring at cosines as
        # low as 0.52 (ISSC minutes → WB MBR, MD review → WB MBR). 0.68
        # is still relaxed enough to catch legitimate MoMs (WB July minutes
        # cosine=0.673 still clears) while blocking cross-domain matches.
        default=0.68,
        alias="MEETING_ANCHOR_MOM_SEMANTIC_THRESHOLD",
        validation_alias="MEETING_ANCHOR_MOM_SEMANTIC_THRESHOLD",
    )
    meeting_anchor_mom_is_strong: bool = Field(
        default=True,
        alias="MEETING_ANCHOR_MOM_IS_STRONG",
        validation_alias="MEETING_ANCHOR_MOM_IS_STRONG",
    )
    # Optional cap on how long after a meeting ends an email may still
    # anchor to it. 0 = no cap (only bounded by current pipeline run).
    meeting_anchor_max_post_meeting_days: float = Field(
        default=0.0,
        alias="MEETING_ANCHOR_MAX_POST_MEETING_DAYS",
        validation_alias="MEETING_ANCHOR_MAX_POST_MEETING_DAYS",
    )
    # Optional PRE-meeting window: allow an email that arrived up to this
    # many days BEFORE a meeting starts to still anchor to it. The default
    # 0.0 preserves the original post-meeting-only behaviour (an email must
    # arrive after the meeting ends). Set > 0 when emails that precede a
    # meeting should inform it (pre-meeting briefs), e.g. test data where
    # the relevant calendar entries are in the future relative to the mail.
    meeting_anchor_max_pre_meeting_days: float = Field(
        default=0.0,
        alias="MEETING_ANCHOR_MAX_PRE_MEETING_DAYS",
        validation_alias="MEETING_ANCHOR_MAX_PRE_MEETING_DAYS",
    )
    meeting_anchor_mom_body_detection_enabled: bool = Field(
        default=True,
        alias="MEETING_ANCHOR_MOM_BODY_DETECTION_ENABLED",
        validation_alias="MEETING_ANCHOR_MOM_BODY_DETECTION_ENABLED",
    )
    meeting_anchor_mom_body_keywords_raw: str = Field(
        default="minutes of meeting,mom,minutes,meeting notes,meeting minutes",
        alias="MEETING_ANCHOR_MOM_BODY_KEYWORDS",
        validation_alias="MEETING_ANCHOR_MOM_BODY_KEYWORDS",
    )
    # Server-side Graph filter: only fetch mail received within this many
    # days before the pipeline run. 0 falls back to meeting_history_lookback_days.
    email_pipeline_fetch_received_after_days: int = Field(
        default=0,
        alias="EMAIL_PIPELINE_FETCH_RECEIVED_AFTER_DAYS",
        validation_alias="EMAIL_PIPELINE_FETCH_RECEIVED_AFTER_DAYS",
    )
    # Relative-to-BACKEND_ROOT folder for the email agent's local data
    # files (JSONL test logs, local-mode calendar JSON, tracker JSON
    # fallbacks). Not used for cluster/meeting storage, which is in the DB.
    bu_email_data_dir: str = "data"
    # ---- BU email pipeline master switch ----
    # Global on/off switch for the email clustering pipeline. The pipeline
    # no longer runs inside this backend at all - it is invoked by the
    # external one-shot runner `backend/run_pipeline.py` (run on demand or
    # by an external scheduler). When False, that runner refuses to run and
    # every LLM/embedding guard in the pipeline (summarization, topic
    # tagging, OCR, embeddings, cluster synthesis, action-item extraction)
    # short-circuits, so no Graph fetch or Ollama call ever happens. The
    # backend's read-only routes keep serving whatever cluster/topic records
    # already exist in PostgreSQL regardless of this flag. Default True.
    bu_email_pipeline_enabled: bool = Field(
        default=True,
        alias="BU_EMAIL_PIPELINE_ENABLED",
        validation_alias="BU_EMAIL_PIPELINE_ENABLED",
    )
    # Explicit summarizer-test calls append diagnostics to a separate
    # JSONL file; the production CSV pipeline never writes to this file.
    email_summarizer_test_log_enabled: bool = True
    email_summarizer_test_log_filename: str = "email_summarizer_test_logs.jsonl"

    # ---- Attachment MoM (Minutes of Meeting) fuzzy filter ----
    # When enabled, only attachments whose filename fuzzy-matches one of the
    # MoM keywords below are parsed/extracted at all (PDF/Excel/PPTX) -
    # everything else (decks, random forwards, logos, unrelated PDFs) is
    # skipped before any extraction happens, so attachment text can never
    # blow up the LLM prompt with irrelevant content. Toggle off to restore
    # the previous "parse every attachment" behaviour.
    mom_attachment_filter_enabled: bool = True
    # Comma-separated list of keywords/phrases an attachment's filename must
    # fuzzily match at least one of (case-insensitive) to be parsed. Edit
    # directly (or via MOM_ATTACHMENT_KEYWORDS in .env) to add more MoM
    # naming conventions - no code change required.
    mom_attachment_keywords_raw: str = Field(
        default="Minutes of Meeting,MoM,Minutes,Meeting Notes,Meeting Minutes",
        alias="MOM_ATTACHMENT_KEYWORDS",
        validation_alias="MOM_ATTACHMENT_KEYWORDS",
    )
    # Minimum fuzzy-match score (0-100, via rapidfuzz) an attachment's
    # filename must reach against any one keyword to be treated as a MoM
    # document. Lower = more permissive (more false positives let through),
    # higher = stricter (more real MoMs with unusual filenames skipped).
    mom_attachment_fuzzy_threshold: int = 80

    # ---- Frontend / session (delegated SSO flow) ----
    # Public URL of the SPA the browser is sent back to after a successful
    # `/auth/callback`. The frontend runs on a *different* origin/port than
    # this backend, so this must be set explicitly (e.g. the Vite dev server
    # or the deployed SPA host).
    frontend_url: str = "http://localhost:8518"
    # Name of the HTTP-only cookie that carries the opaque session id. The
    # actual Graph tokens never leave the server - only this id does.
    session_cookie_name: str = "rbl_session"
    # Cookie flags. Default here assumes the common case: FRONTEND_URL and
    # APP_BASE_URL are the same public origin (nginx fronts both the SPA
    # and this backend), so `samesite=lax` + `secure` matching the
    # browser-facing scheme is correct and sufficient - no need for
    # `samesite=none`, which only matters when the SPA truly runs on a
    # different origin than this backend. Local http development keeps
    # `secure=False` + `samesite=lax`.
    session_cookie_secure: bool = False
    session_cookie_samesite: str = "lax"  # "lax" | "strict" | "none"
    # How long a delegated session is considered valid, in seconds. Should
    # be <= the Graph access-token lifetime (default ~3600s).
    session_ttl_seconds: int = 3600

    # ---- AWS Secrets Manager ----
    # The PostgreSQL credentials (POSTGRES_USER / POSTGRES_PASSWORD /
    # POSTGRES_DB) and the delegated Microsoft Graph OAuth refresh token are
    # sourced from a single JSON secret in AWS Secrets Manager — NOT from
    # .env or the application database. boto3 uses the standard credential
    # chain (the EC2 instance IAM role in production); no AWS keys are set
    # here. See core/secrets_manager.py and auth/token_store.py.
    aws_region: str = Field(
        default="ap-south-1",
        alias="AWS_REGION",
        validation_alias="AWS_REGION",
        description="AWS region for Secrets Manager",
    )
    aws_secret_name: str = Field(
        default="rbl-digitalofficeofceo-prod-parameterstore",
        alias="AWS_SECRET_NAME",
        validation_alias="AWS_SECRET_NAME",
        description="Name of the JSON secret holding PG creds + refresh_token",
    )

    # ---- Storage backend toggle ----
    # "memory" | "postgres" — used by the tracker stores (agent_open_tracker)
    # which fall back to JSON files when not "postgres".
    storage_backend: str = "memory"
    # PostgreSQL host:port + base DSN scheme/options. The USER / PASSWORD /
    # DB NAME are NO LONGER taken from here (or .env): they are fetched from
    # AWS Secrets Manager at startup and spliced in to build `postgres_dsn`
    # (see `_build_postgres_dsn_from_secret`). Only the non-secret
    # connection target (host, port, and any query params like sslmode)
    # should be configured via POSTGRES_HOST / POSTGRES_PORT / POSTGRES_OPTIONS.
    postgres_host: str = Field(
        default="",
        alias="POSTGRES_HOST",
        validation_alias="POSTGRES_HOST",
    )
    postgres_port: int = Field(
        default=5432,
        alias="POSTGRES_PORT",
        validation_alias="POSTGRES_PORT",
    )
    postgres_options: str = Field(
        default="",
        alias="POSTGRES_OPTIONS",
        validation_alias="POSTGRES_OPTIONS",
        description="Optional DSN query string, e.g. 'sslmode=require'",
    )
    # Resolved asyncpg DSN. Populated at startup from Secrets Manager
    # (user/password/db) + POSTGRES_HOST/PORT above. Left empty ("Postgres
    # disabled, JSON-file fallback") only when no POSTGRES_HOST is set.
    # Not read from .env anymore — the `_resolve_postgres_dsn` validator
    # overwrites any env-provided value.
    postgres_dsn: str = ""

    # ---- Email agent storage ----
    # The email cluster synthesis pipeline persists ROW data + embeddings to
    # PostgreSQL (the system of record), mirrors the embedding VECTORS into
    # Weaviate (a rebuildable searchable copy), and caches reads in Redis.
    # There is no CSV / flat-file backend. The three-layer stack is active
    # whenever `POSTGRES_DSN` is set (see `email_backend_is_postgres`).
    #
    # Weaviate connection settings for the email agent's embedding vector
    # store (cluster + meeting vectors). Entirely separate from the chatbot,
    # which keeps using its own Weaviate collection untouched by this config.
    #
    # Uses weaviate.connect_to_custom() with independent REST (http_*) and
    # gRPC (grpc_*) endpoints, matching the bank's internal Weaviate
    # deployment (REST via a TLS-terminating hostname, gRPC via the
    # internal NLB on a separate port, no gRPC TLS). Defaults are
    # intentionally generic/empty — the real endpoints must be set in
    # .env; nothing deployment-specific is hardcoded in source.
    weaviate_http_host: str = Field(
        default="localhost",
        alias="WEAVIATE_HTTP_HOST",
        validation_alias="WEAVIATE_HTTP_HOST",
    )
    weaviate_http_port: int = Field(
        default=443,
        alias="WEAVIATE_HTTP_PORT",
        validation_alias="WEAVIATE_HTTP_PORT",
    )
    weaviate_http_secure: bool = Field(
        default=True,
        alias="WEAVIATE_HTTP_SECURE",
        validation_alias="WEAVIATE_HTTP_SECURE",
    )
    weaviate_grpc_host: str = Field(
        default="localhost",
        alias="WEAVIATE_GRPC_HOST",
        validation_alias="WEAVIATE_GRPC_HOST",
    )
    weaviate_grpc_port: int = Field(
        default=50051,
        alias="WEAVIATE_GRPC_PORT",
        validation_alias="WEAVIATE_GRPC_PORT",
    )
    weaviate_grpc_secure: bool = Field(
        default=False,
        alias="WEAVIATE_GRPC_SECURE",
        validation_alias="WEAVIATE_GRPC_SECURE",
    )
    # Optional API key, if the deployment requires auth (empty = anonymous,
    # matching the connectivity test script).
    weaviate_api_key: str = Field(
        default="",
        alias="WEAVIATE_API_KEY",
        validation_alias="WEAVIATE_API_KEY",
    )
    # Collection name prefix so clusters/meetings namespaces never collide
    # inside the shared Weaviate instance.
    weaviate_email_collection_prefix: str = Field(
        default="EmailAgent",
        alias="WEAVIATE_EMAIL_COLLECTION_PREFIX",
        validation_alias="WEAVIATE_EMAIL_COLLECTION_PREFIX",
    )

    # Collection name prefix for the documents bot's block + series
    # vectors. Produces `<prefix>Blocks` and `<prefix>Series` on the
    # shared Weaviate instance. Kept separate from the email prefix so
    # the two features never share a collection.
    weaviate_doc_collection_prefix: str = Field(
        default="DocLibrary",
        alias="WEAVIATE_DOC_COLLECTION_PREFIX",
        validation_alias="WEAVIATE_DOC_COLLECTION_PREFIX",
    )

    # ---- Vector backend toggle (chatbot + documents bot) ----
    # "weaviate" (default): vectors stored in the shared Weaviate deployment.
    # "faiss":  vectors stored in local FAISS index files under FAISS_INDEX_DIR.
    # Switching backend does NOT migrate existing vectors — rebuild/re-ingest.
    chatbot_vector_db: str = Field(
        default="weaviate",
        alias="CHATBOT_VECTOR_DB",
        validation_alias="CHATBOT_VECTOR_DB",
    )
    # Directory for FAISS index files (one .faiss + .json per namespace).
    # Only used when CHATBOT_VECTOR_DB=faiss.
    faiss_index_dir: str = Field(
        default="data/faiss_indexes",
        alias="FAISS_INDEX_DIR",
        validation_alias="FAISS_INDEX_DIR",
    )

    @property
    def use_faiss(self) -> bool:
        """True when the vector backend is FAISS (not Weaviate)."""
        return self.chatbot_vector_db.strip().lower() == "faiss"

    # ---- Chatbot (RAG over raw email conversations) ----
    # The chatbot answers questions by retrieving from the raw per-message
    # email table in PostgreSQL (same database as `postgres_dsn` above —
    # the chatbot does not open a second connection pool) and from a
    # dedicated Weaviate collection (hybrid vector + BM25 search).
    # Entirely independent of `email_storage_backend`, which governs the
    # *cluster synthesis* pipeline's own data, not the chatbot's.
    chatbot_pg_table: str = Field(
        default="bu_emails",
        alias="CHATBOT_PG_TABLE",
        validation_alias="CHATBOT_PG_TABLE",
    )
    # Kept for backward compatibility with .env files that set it.
    # Weaviate is now the only retrieval backend; this is always treated
    # as True regardless of the configured value.
    chatbot_use_weaviate: bool = Field(
        default=True,
        alias="CHATBOT_USE_WEAVIATE",
        validation_alias="CHATBOT_USE_WEAVIATE",
    )
    # When true, the email pipeline re-embeds the chatbot's source rows
    # into Weaviate at the end of every run (see chatbot/refresh.py), so
    # freshly fetched mail is answerable without restarting the backend.
    # Startup ingestion only runs when the collection is empty, so without
    # this the chatbot would stay frozen on its first-boot snapshot.
    # Turn off to avoid the per-run embedding cost and reindex manually
    # via POST /chatbot/index instead.
    chatbot_auto_reindex: bool = Field(
        default=True,
        alias="CHATBOT_AUTO_REINDEX",
        validation_alias="CHATBOT_AUTO_REINDEX",
    )
    # Dedicated Weaviate collection name for chatbot message-level vectors.
    # Kept separate from `weaviate_email_collection_prefix` (clusters/
    # meetings namespaces used by the email cluster pipeline) so the two
    # features never share or collide on the same collection.
    chatbot_weaviate_collection: str = Field(
        default="ChatbotEmailMessage",
        alias="CHATBOT_WEAVIATE_COLLECTION",
        validation_alias="CHATBOT_WEAVIATE_COLLECTION",
    )
    # Retrieval tuning: unique message-level candidates considered before
    # grouping into threads, and the number of top threads sent to the LLM.
    chatbot_retrieval_candidates: int = Field(
        default=25,
        alias="CHATBOT_RETRIEVAL_CANDIDATES",
        validation_alias="CHATBOT_RETRIEVAL_CANDIDATES",
    )
    chatbot_top_k: int = Field(
        default=3,
        alias="CHATBOT_TOP_K",
        validation_alias="CHATBOT_TOP_K",
    )
    chatbot_llm_model: str = Field(
        default="llama3b-fast:latest",
        alias="CHATBOT_LLM_MODEL",
        validation_alias="CHATBOT_LLM_MODEL",
    )
    chatbot_max_tokens: int = Field(
        default=256,
        alias="CHATBOT_MAX_TOKENS",
        validation_alias="CHATBOT_MAX_TOKENS",
    )
    chatbot_llm_num_ctx: int = Field(
        default=4096,
        alias="CHATBOT_LLM_NUM_CTX",
        validation_alias="CHATBOT_LLM_NUM_CTX",
    )
    # Sampling / runtime options sent to Ollama for chatbot answers. Both the
    # non-streaming and streaming paths read them through one helper
    # (chatbot.service._chat_request). temperature 0 + repeat_penalty 1.0 keep
    # figures copied digit for digit (no sampling noise, no penalty on repeated
    # digits).
    chatbot_temperature: float = Field(
        default=0.0,
        alias="CHATBOT_TEMPERATURE",
        validation_alias="CHATBOT_TEMPERATURE",
    )
    chatbot_repeat_penalty: float = Field(
        default=1.0,
        alias="CHATBOT_REPEAT_PENALTY",
        validation_alias="CHATBOT_REPEAT_PENALTY",
    )
    # CPU threads for generation: set to the server's PHYSICAL core count.
    chatbot_num_thread: int = Field(
        default=8,
        alias="CHATBOT_NUM_THREAD",
        validation_alias="CHATBOT_NUM_THREAD",
    )
    # Seconds Ollama keeps the model loaded after a request (-1 = forever).
    chatbot_keep_alive: int = Field(
        default=-1,
        alias="CHATBOT_KEEP_ALIVE",
        validation_alias="CHATBOT_KEEP_ALIVE",
    )
    # Plain-text file holding the chatbot system prompt. A relative path is
    # resolved against the backend root. The file is re-read when it changes,
    # so edits apply without a restart. A missing/empty file is an error (no
    # silent fallback to a built-in prompt).
    chatbot_system_prompt_path: str = Field(
        default="chatbot/system_prompt.txt",
        alias="CHATBOT_SYSTEM_PROMPT_PATH",
        validation_alias="CHATBOT_SYSTEM_PROMPT_PATH",
    )
    # Model routing. With CHATBOT_MODEL_ROUTING off (default) every question
    # uses CHATBOT_LLM_MODEL, exactly as before. With it on, questions that ask
    # for a number (matched by CHATBOT_NUMERIC_QUESTION_PATTERN) go to the
    # accurate model and every other question goes to the fast model.
    chatbot_model_routing: bool = Field(
        default=False,
        alias="CHATBOT_MODEL_ROUTING",
        validation_alias="CHATBOT_MODEL_ROUTING",
    )
    chatbot_llm_model_fast: str = Field(
        default="llama3b-fast:latest",
        alias="CHATBOT_LLM_MODEL_FAST",
        validation_alias="CHATBOT_LLM_MODEL_FAST",
    )
    chatbot_llm_model_accurate: str = Field(
        default="llama8b-fast:latest",
        alias="CHATBOT_LLM_MODEL_ACCURATE",
        validation_alias="CHATBOT_LLM_MODEL_ACCURATE",
    )
    # Regex (case-insensitive) that marks a question as asking for a number.
    chatbot_numeric_question_pattern: str = Field(
        default=r"\bhow\s+(much|many)\b|\bwhat\s+(percentage|percent)\b",
        alias="CHATBOT_NUMERIC_QUESTION_PATTERN",
        validation_alias="CHATBOT_NUMERIC_QUESTION_PATTERN",
    )
    # Character budget for the compact, deduplicated RAG context sent to the
    # chatbot LLM (chatbot.retriever.build_context). ~4000 chars ≈ 1000 tokens;
    # with the short system prompt the whole prompt stays near 1,000-1,500
    # tokens. Editable via CHATBOT_CONTEXT_MAX_CHARS.
    chatbot_context_max_chars: int = Field(
        default=4000,
        alias="CHATBOT_CONTEXT_MAX_CHARS",
        validation_alias="CHATBOT_CONTEXT_MAX_CHARS",
    )
    # Max number of retrieved threads that build_context may put in the prompt
    # (best-ranked threads that share a real keyword with the question). This
    # only trims what is SENT to the LLM; retrieval itself is unchanged.
    chatbot_context_top_threads: int = Field(
        default=2,
        alias="CHATBOT_CONTEXT_TOP_THREADS",
        validation_alias="CHATBOT_CONTEXT_TOP_THREADS",
    )
    # Share of the context budget reserved for table rows (never dropped by the
    # noise filter, length rule, dedup or scoring). Rows are cut only when they
    # exceed this share, and a WARNING with the thread subject is logged.
    chatbot_context_table_fraction: float = Field(
        default=0.5,
        alias="CHATBOT_CONTEXT_TABLE_FRACTION",
        validation_alias="CHATBOT_CONTEXT_TABLE_FRACTION",
    )
    # Comma-separated extra words ignored when matching question words to
    # emails (on top of the built-in stopword/generic-word lists).
    chatbot_context_ignore_words: str = Field(
        default="",
        alias="CHATBOT_CONTEXT_IGNORE_WORDS",
        validation_alias="CHATBOT_CONTEXT_IGNORE_WORDS",
    )
    # Hard ceiling on the assembled user message (context + question), in
    # characters. Above it the context is rebuilt once with a smaller budget.
    chatbot_prompt_char_ceiling: int = Field(
        default=6000,
        alias="CHATBOT_PROMPT_CHAR_CEILING",
        validation_alias="CHATBOT_PROMPT_CHAR_CEILING",
    )
    # Factor applied to the context budget on that single rebuild.
    chatbot_prompt_shrink_factor: float = Field(
        default=0.6,
        alias="CHATBOT_PROMPT_SHRINK_FACTOR",
        validation_alias="CHATBOT_PROMPT_SHRINK_FACTOR",
    )
    # Log a WARNING when Ollama reports more prompt tokens than this.
    chatbot_prompt_warn_tokens: int = Field(
        default=2500,
        alias="CHATBOT_PROMPT_WARN_TOKENS",
        validation_alias="CHATBOT_PROMPT_WARN_TOKENS",
    )
    # Debug aid: when true, the chatbot dumps the EXACT final prompt (system +
    # context + question) and the RAW LLM answer (before the hallucination
    # regex / NOT_FOUND replacement) to `chatbot_debug_prompt_path`,
    # overwriting it each request, and logs which threads were retrieved
    # (subject + rank) plus how many sentences build_context kept per thread.
    # Off by default - a diagnostic switch, not for production traffic.
    chatbot_debug_prompt: bool = Field(
        default=False,
        alias="CHATBOT_DEBUG_PROMPT",
        validation_alias="CHATBOT_DEBUG_PROMPT",
    )
    # Destination file for the CHATBOT_DEBUG_PROMPT dump (overwritten each
    # request). Only used when chatbot_debug_prompt is true.
    chatbot_debug_prompt_path: str = Field(
        default=r"C:\Users\C59250\Desktop\last_prompt.txt",
        alias="CHATBOT_DEBUG_PROMPT_PATH",
        validation_alias="CHATBOT_DEBUG_PROMPT_PATH",
    )
    # ---- Redis (chatbot chat-history persistence) ----
    # Nice-to-have: if unreachable, chat keeps working without history.
    # Empty host disables history persistence outright (no connection
    # attempts made).
    redis_host: str = Field(
        default="",
        alias="REDIS_HOST",
        validation_alias="REDIS_HOST",
    )
    redis_port: int = Field(
        default=7480,
        alias="REDIS_PORT",
        validation_alias="REDIS_PORT",
    )
    redis_db: int = Field(
        default=0,
        alias="REDIS_DB",
        validation_alias="REDIS_DB",
    )
    redis_password: str = Field(
        default="",
        alias="REDIS_PASSWORD",
        validation_alias="REDIS_PASSWORD",
    )
    redis_ssl: bool = Field(
        default=False,
        alias="REDIS_SSL",
        validation_alias="REDIS_SSL",
    )
    # Project/key namespace so this app never reads or overwrites another
    # project's keys on a shared Redis. All keys are prefixed with this.
    redis_key_prefix: str = Field(
        default="rbl_email:",
        alias="REDIS_KEY_PREFIX",
        validation_alias="REDIS_KEY_PREFIX",
    )

    # ---- Ollama (local open-source models) ----
    # Generic localhost default; the real shared Ollama host must be set
    # in .env. No deployment-specific IP is hardcoded in source.
    ollama_host: str = "http://localhost:11434"
    ollama_vision_model: str = "llama3.2-vision:11b"
    ollama_text_model: str = ""  # falls back to ollama_vision_model when empty
    ollama_request_timeout_seconds: int = 600
    ollama_temperature: float = 0.2
    # Context window (in tokens) for both the text/vision chat model and
    # the embedding model. Ollama's default is 2048 tokens, which silently
    # truncates earlier emails in a cluster synthesis prompt. 16k holds a
    # full cluster (bodies + MoM attachments) in one call.
    ollama_num_ctx: int = Field(
        default=16384,
        alias="OLLAMA_NUM_CTX",
        validation_alias="OLLAMA_NUM_CTX",
    )
    # Max tokens the chat model may emit per call. 0 = Ollama default
    # (unlimited). Keep this modest for short tagging calls.
    ollama_num_predict: int = Field(
        default=768,
        alias="OLLAMA_NUM_PREDICT",
        validation_alias="OLLAMA_NUM_PREDICT",
    )
    # Cluster synthesis must list every material action item from MoMs,
    # so it needs a higher generation cap than topic-tagging calls.
    ollama_cluster_synth_num_predict: int = Field(
        default=2048,
        alias="OLLAMA_CLUSTER_SYNTH_NUM_PREDICT",
        validation_alias="OLLAMA_CLUSTER_SYNTH_NUM_PREDICT",
    )
    # ---- Ollama in-flight heartbeat logging toggle ----
    # While any single Ollama call is running, `core.ollama_client.client`
    # emits a WARNING heartbeat line every `ollama_heartbeat_interval_seconds`
    # so operators watching the backend terminal can tell a long call is
    # still progressing (not stuck). This is NOT a poll of the Ollama
    # server - it only tracks this backend's own in-flight call. When
    # False (or OLLAMA_HEARTBEAT_ENABLED=false in `.env`), no heartbeat
    # lines are logged and the call is simply awaited quietly. Default
    # True preserves the existing behavior.
    ollama_heartbeat_enabled: bool = Field(
        default=True,
        alias="OLLAMA_HEARTBEAT_ENABLED",
        validation_alias="OLLAMA_HEARTBEAT_ENABLED",
    )
    # How often (seconds) the in-flight heartbeat above logs while a
    # single Ollama call is still running. Only consulted when
    # `ollama_heartbeat_enabled` is True. Editable via
    # OLLAMA_HEARTBEAT_INTERVAL_SECONDS in `.env`.
    ollama_heartbeat_interval_seconds: float = 15.0

    # ---- EML file storage ----
    # Relative to BACKEND_ROOT. Used by the chatbot / eml summariser,
    # and by GRAPH_AUTH_MODE=local as the offline Outlook mailbox.
    eml_storage_dir: str = "temp_mail_data"
    # Dummy Outlook calendar used only when GRAPH_AUTH_MODE=local.
    # Relative to BACKEND_ROOT unless an absolute path is given.
    local_calendar_path: str = "data/local_calendar.json"
    # Past meeting occurrences for local mode (days_ago offsets). Merged
    # into calendarView responses so the cluster pipeline can match mails
    # to historical meetings without Graph API access.
    local_calendar_history_path: str = "data/local_calendar_history.json"
    # Fallback anchor tokens when no calendar occurrence matches (comma-separated).
    email_cluster_anchor_tokens_raw: str = Field(
        default="minutes,mom,notes,note,action,actions",
        alias="EMAIL_CLUSTER_ANCHOR_TOKENS",
        validation_alias="EMAIL_CLUSTER_ANCHOR_TOKENS",
    )

    # ---- Concurrency limits ----
    max_concurrent_graph_calls: int = 5
    max_concurrent_ollama_calls: int = 2
    # Separate, usually-stricter cap on how many BU pipeline messages are
    # processed concurrently (each message with attachments issues its own
    # get_attachments + one get_attachment_content call against Graph).
    # Microsoft Graph enforces a PER-MAILBOX concurrency limit on top of
    # this app's own MAX_CONCURRENT_GRAPH_CALLS - the shared mailbox used
    # here ("ApplicationThrottled"/"MailboxConcurrency limit") can throttle
    # well before the process-wide Graph semaphore is exhausted, especially
    # since every matched message's attachment calls land on that SAME
    # mailbox. Keeping this lower than MAX_CONCURRENT_GRAPH_CALLS reduces
    # how many simultaneous calls ever hit that one mailbox at once.
    bu_pipeline_max_concurrent_messages: int = 3
    # Number of queue-draining worker coroutines used to synthesize email
    # clusters in parallel during a pipeline run (core.workers pool). This
    # controls how many CLUSTERS are summarized concurrently; it does NOT
    # change clustering/anchoring (those run once, before synthesis). The
    # real ceiling on simultaneous Ollama calls remains
    # MAX_CONCURRENT_OLLAMA_CALLS above — more workers simply keep that
    # semaphore saturated instead of idling between clusters. 1 = the old
    # fully-sequential behaviour. Tune together with MAX_CONCURRENT_OLLAMA_CALLS.
    email_pipeline_cluster_workers: int = Field(
        default=2,
        alias="EMAIL_PIPELINE_CLUSTER_WORKERS",
        validation_alias="EMAIL_PIPELINE_CLUSTER_WORKERS",
    )

    # How many EMAILS are processed concurrently through the relevance /
    # single-extraction stage (per-email context extraction + meeting-match +
    # gate embedding). Like the cluster-synthesis pool this only governs how
    # many items are in flight; the real cap on simultaneous Ollama calls is
    # still MAX_CONCURRENT_OLLAMA_CALLS. Default 1 = fully sequential, which is
    # the safe default for a locally-hosted Ollama host; raise it only when the
    # model host has spare concurrency.
    email_pipeline_relevance_workers: int = Field(
        default=1,
        alias="EMAIL_PIPELINE_RELEVANCE_WORKERS",
        validation_alias="EMAIL_PIPELINE_RELEVANCE_WORKERS",
    )

    # ---- Retry / backoff ----
    retry_max_attempts: int = 3
    retry_base_delay_seconds: float = 1.0

    # ---- Calendar polling (frontend) ----
    # Interval in seconds for polling shared calendar data in real-time
    calendar_polling_interval_seconds: int = 300  # 5 minutes
    # Master switch for the frontend's real-time calendar polling. The
    # calendar agent has NO server-side background loop (meetings are
    # fetched on-demand per request); "calendar polling" is the SPA
    # re-hitting the calendar endpoints on the interval above. When False
    # (or CALENDAR_POLLING_ENABLED=false in `.env`), this flag is exposed
    # to the frontend (via GET /agents/calendar/polling-config) so the
    # SPA can stop its timer and only fetch calendar data once/on demand.
    # Default True preserves the existing behavior.
    calendar_polling_enabled: bool = Field(
        default=True,
        alias="CALENDAR_POLLING_ENABLED",
        validation_alias="CALENDAR_POLLING_ENABLED",
    )

    # ---- Calendar timezone workaround ----
    # Originally assumed the shared mailbox's calendar events were IST
    # wall-clock values mislabeled as UTC (see git history for the old
    # rationale). That assumption was disproven by comparing real
    # Outlook display times against the raw Graph `dateTime` values:
    # a raw value of "12:30" shows up in Outlook's own UI around 18:00
    # (12:30 + 5:30) - i.e. the raw value genuinely IS UTC, and Outlook
    # (and this app's frontend) correctly shift it to IST for display.
    # Keep this False so `dateTime` is parsed as true UTC. Only flip it
    # back on (or via CALENDAR_SOURCE_IS_MISLABELED_UTC=true in .env) if
    # the mailbox's behavior changes and raw values start matching IST
    # wall-clock directly again.
    calendar_source_is_mislabeled_utc: bool = False

    # ---- OneDrive / SharePoint document sync ----
    # Master switch for the scheduled SharePoint -> document library sync.
    # When true, the daily scheduler pulls PDFs from the configured
    # SharePoint/OneDrive shared folder and ingests them via the existing
    # documents pipeline.
    onedrive_sync_enabled: bool = Field(
        default=False,
        alias="ONEDRIVE_SYNC_ENABLED",
        validation_alias="ONEDRIVE_SYNC_ENABLED",
    )
    # SharePoint/OneDrive sharing URL shared by the business. This is the
    # browser URL you get when someone shares a folder via OneDrive/Teams/
    # SharePoint (e.g. "https://rblbank-my.sharepoint.com/shared?id=...").
    # The sync resolves this via Graph's /shares API to a driveItem, so
    # NO Files.Read.All permission is needed — only Sites.Selected with
    # the underlying site granted access by an admin.
    sharepoint_site: str = Field(
        default="",
        alias="SHAREPOINT_SITE",
        validation_alias="SHAREPOINT_SITE",
    )
    # Multiple SharePoint folders to sync (pipe-separated). Each URL is
    # synced in order. If set, SHAREPOINT_SITE above is ignored.
    # Example: "https://...folder1|https://...folder2"
    sharepoint_sites: str = Field(
        default="",
        alias="SHAREPOINT_SITES",
        validation_alias="SHAREPOINT_SITES",
    )

    @property
    def sharepoint_sites_list(self) -> list[str]:
        """All SharePoint folder URLs to sync, parsed from SHAREPOINT_SITES
        (pipe-separated) or falling back to the single SHAREPOINT_SITE."""
        if self.sharepoint_sites.strip():
            return [u.strip() for u in self.sharepoint_sites.split("|") if u.strip()]
        if self.sharepoint_site.strip():
            return [self.sharepoint_site.strip()]
        return []
    # Comma-separated file extensions to sync. Only matching files are
    # downloaded; everything else is ignored.
    onedrive_file_extensions: str = Field(
        default=".pdf,.pptx",
        alias="ONEDRIVE_FILE_EXTENSIONS",
        validation_alias="ONEDRIVE_FILE_EXTENSIONS",
    )
    # Hour (0-23, IST) and minute for the daily scheduled sync. The sync
    # runs once at this time every day when onedrive_sync_enabled=true.
    onedrive_sync_schedule_hour: int = Field(
        default=2,
        alias="ONEDRIVE_SYNC_SCHEDULE_HOUR",
        validation_alias="ONEDRIVE_SYNC_SCHEDULE_HOUR",
    )
    onedrive_sync_schedule_minute: int = Field(
        default=0,
        alias="ONEDRIVE_SYNC_SCHEDULE_MINUTE",
        validation_alias="ONEDRIVE_SYNC_SCHEDULE_MINUTE",
    )
    # Temp directory for downloaded files before ingestion (relative to
    # backend/). Cleaned up after each file is ingested.
    onedrive_temp_dir: str = Field(
        default="data/onedrive_downloads",
        alias="ONEDRIVE_TEMP_DIR",
        validation_alias="ONEDRIVE_TEMP_DIR",
    )
    # Whether to use the vision model during PDF ingestion (chart reading).
    # Passed through to library.add_pdf(). False = faster but no chart data.
    onedrive_use_vision: bool = Field(
        default=True,
        alias="ONEDRIVE_USE_VISION",
        validation_alias="ONEDRIVE_USE_VISION",
    )
    # Local folder (relative to backend/) scanned instead of SharePoint when
    # GRAPH_AUTH_MODE=local. Drop PDFs here for offline development.
    onedrive_local_dir: str = Field(
        default="data/local_onedrive",
        alias="ONEDRIVE_LOCAL_DIR",
        validation_alias="ONEDRIVE_LOCAL_DIR",
    )

    @property
    def onedrive_extensions_tuple(self) -> tuple[str, ...]:
        """Parsed file extensions as a lowercase tuple for matching."""
        return tuple(
            ext.strip().lower()
            for ext in self.onedrive_file_extensions.split(",")
            if ext.strip()
        )

    @property
    def resolved_onedrive_temp_dir(self) -> "Path":
        """Absolute path to the OneDrive download temp directory."""
        p = Path(self.onedrive_temp_dir)
        if not p.is_absolute():
            p = BACKEND_ROOT / p
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def resolved_onedrive_local_dir(self) -> "Path":
        """Absolute path to the local OneDrive fallback folder."""
        p = Path(self.onedrive_local_dir)
        if not p.is_absolute():
            p = BACKEND_ROOT / p
        p.mkdir(parents=True, exist_ok=True)
        return p

    # ---- Observability (Langfuse) ----
    langfuse_enabled: bool = False
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_host: str = "https://cloud.langfuse.com"

    # ---- Logging ----
    log_level: str = "INFO"
    log_dir: str = "logs"
    log_max_bytes: int = 5 * 1024 * 1024
    log_backup_count: int = 5

    # ---- BU departments (email clustering + calendar->cluster mapping) ----
    # The FIXED set of business-unit departments the whole system
    # standardises on. Cluster department is assigned by nearest
    # dictionary prototype (summary + DEPARTMENT_TAGGING_EXAMPLE_PHRASES),
    # with a token-overlap tie-break on close scores — not by the
    # synthesis LLM. The calendar agent maps a meeting to a cluster
    # using the stored department name. Edit here (or via BU_DEPARTMENTS
    # in `.env`) to add/remove/rename a department or reword its
    # summary - no code change required.
    #
    # Format: a JSON array of {"name": ..., "summary": ...} objects. The
    # `summary` is a short description of what the department covers; it
    # is embedded into the department centroid AND shown to the synthesis
    # LLM as scope context (the LLM does not retag). Shared product words
    # (CASA, CA/SA, deposits, TAT) must be qualified here (retail vs
    # wholesale, cards vs corporate) so overlapping vocabulary does not
    # collapse two departments into one embedding blob.
    bu_departments_raw: str = Field(
        default=(
            '['
            '{"name": "BB&RL", "view": "Retail", '
            '"summary": "Retail branch banking and retail liabilities only: branch network, '
            'retail CASA, retail CA/SA targets, retail term deposits, branch deposit mobilisation '
            'and branch operations. Not wholesale CA/SA, not the wholesale or corporate book."}, '
            '{"name": "Credit Cards", "view": "CRO", '
            '"summary": "Credit cards business only: card acquisition and activation, spends and interchange, '
            'cards portfolio, EMI, rewards, delinquency and collections specific to cards. '
            'Not corporate credit TAT or wholesale underwriting."}, '
            '{"name": "Retail Advances - Excluding Credit Cards", "view": "RetailAssets", '
            '"summary": "Retail lending other than credit cards: home loans, LAP, personal loans, '
            'secured and unsecured retail assets, disbursements, retail GNPA and collections excluding cards."}, '
            '{"name": "Wholesale Banking", "view": "Wholesale", '
            '"summary": "Wholesale and corporate banking: large/mid corporate relationships, '
            'wholesale deposits, wholesale CASA, wholesale CA/SA targets, WB CA/SA, '
            'transaction banking, trade and treasury client coverage, '
            'funding cost and NIM assumptions on the wholesale book, corporate credit TAT, '
            'underwriting, pipeline and sanctions."}'
            ']'
        ),
        alias="BU_DEPARTMENTS",
        validation_alias="BU_DEPARTMENTS",
    )
    # Embedding model used for relevance scoring, clustering, and
    # meeting->cluster matching. Must be pulled on the Ollama host.
    embedding_model: str = Field(
        default="nomic-embed-text",
        alias="EMBEDDING_MODEL",
        validation_alias="EMBEDDING_MODEL",
    )
    # Dimensionality of the vectors produced by `embedding_model`. Stored
    # alongside every embedding (embedding_dim column + Weaviate restore)
    # so the vector width is NEVER hardcoded anywhere — it is read from
    # here. nomic-embed-text emits 768-d vectors; change this if the model
    # changes. This is metadata only: it does not reshape or truncate any
    # vector, it records the width the model is expected to produce so a
    # mismatch can be detected and logged rather than silently stored.
    embedding_dim: int = Field(
        default=768,
        alias="EMBEDDING_DIM",
        validation_alias="EMBEDDING_DIM",
    )
    # ---- Vector-store backend toggle (email agent only) ----
    # TEMPORARY toggle. Selects which vector store the EMAIL CLUSTER
    # pipeline (agent_email) uses for its cluster/meeting embeddings:
    #   "weaviate" (default) -> the bank's shared Weaviate deployment
    #                           (agent_email/weaviate_vector_store.py).
    #   "faiss"              -> a local on-disk FAISS index under
    #                           `resolved_faiss_index_dir` (core/faiss/).
    # The chatbot is NOT affected by this (it always uses Weaviate). This
    # exists only so the local FAISS path can be exercised without a
    # Weaviate server; it is meant to be removed later by deleting
    # `core/faiss/`, `agent_email/vector_store_backend.py`, this field and
    # `faiss_index_dir`, and reverting the dispatcher import back to
    # `weaviate_vector_store`. Keep this list exactly two values so a typo
    # fails loudly at startup rather than silently picking a backend.
    vector_store_backend: str = Field(
        default="weaviate",
        alias="VECTOR_STORE_BACKEND",
        validation_alias="VECTOR_STORE_BACKEND",
    )
    # Folder (relative to BACKEND_ROOT unless absolute) that holds the local
    # FAISS index files + id maps, one per namespace. Only used when
    # VECTOR_STORE_BACKEND="faiss"; ignored for the Weaviate path. Kept
    # under `core/faiss/` so the whole temporary FAISS backend lives in one
    # place and is trivial to delete.
    faiss_index_dir: str = Field(
        default="core/faiss/index_data",
        alias="FAISS_INDEX_DIR",
        validation_alias="FAISS_INDEX_DIR",
    )
    # Extra example phrases per department, JSON object of
    # {"<department name>": ["phrase", ...]}. Each phrase is a prototype
    # the cluster is scored against (max cosine), AND is averaged into
    # the relevance-filter centroid. Qualify shared product words
    # (CASA / CA/SA / deposits) with the owning book (retail vs wholesale)
    # rather than leaving them bare. Empty object = summary-only seeds.
    # Editable via DEPARTMENT_TAGGING_EXAMPLE_PHRASES.
    department_tagging_example_phrases_raw: str = Field(
        default=(
            '{'
            '"BB&RL": ["retail CASA deposits", "branch banking", "branch term deposit mobilisation", '
            '"retail savings and current account growth", "virtual RM VRM", "retail liability book", '
            '"retail CA/SA targets", "retail term deposits", "retail deposit base", '
            '"branch CASA mobilisation", "branch deposit growth"], '
            '"Credit Cards": ["credit card spends", "card acquisition and activation", '
            '"interchange income", "card EMI", "card rewards", "cards delinquency and collections"], '
            '"Retail Advances - Excluding Credit Cards": ["home loan disbursement", "loan against property LAP", '
            '"personal loan", "microfinance JLG MFI collections", "retail GNPA", "secured and unsecured retail assets"], '
            '"Wholesale Banking": ["corporate advances", "WB wholesale banking", "large and mid corporate relationships", '
            '"transaction banking", "trade finance", "treasury client coverage", "wholesale deposits", '
            '"wholesale CASA", "wholesale CA/SA targets", "WB CA/SA targets", "WB CASA targets", '
            '"wholesale current and savings accounts", '
            '"cost of funds on the WB book", "blended cost of funds", '
            '"corporate credit turnaround time TAT", "corporate credit underwriting", '
            '"corporate proposal exception cases", "corporate credit pipeline and sanctions"]'
            '}'
        ),
        alias="DEPARTMENT_TAGGING_EXAMPLE_PHRASES",
        validation_alias="DEPARTMENT_TAGGING_EXAMPLE_PHRASES",
    )

    @model_validator(mode="after")
    def _require_graph_secrets_unless_local(self):
        """
        Graph credentials are only required when the app will actually
        talk to Microsoft. Local/offline mode reads `.eml` files from
        `eml_storage_dir` and must start without Azure secrets.
        """
        if self.graph_auth_mode.lower() == "local":
            return self
        missing = [
            name
            for name, value in (
                ("TENANT_ID", self.tenant_id),
                ("CLIENT_ID", self.client_id),
                ("CLIENT_SECRET", self.client_secret),
            )
            if not (value or "").strip()
        ]
        if missing:
            raise ValueError(
                "Missing required Graph credentials when "
                f"GRAPH_AUTH_MODE={self.graph_auth_mode!r}: {', '.join(missing)}"
            )
        return self

    @model_validator(mode="after")
    def _validate_graph_base_url(self):
        """
        Structurally validate the configured Microsoft Graph base URL at
        startup (CWE-918 / SSRF defence). This pins the ONLY origin an
        outbound Graph request can reach to a single, config-supplied,
        structurally-parsed value - a bad GRAPH_BASE_URL fails loudly here
        instead of silently letting requests resolve to an unexpected
        scheme/host/port. The heavy lifting lives in
        `core.graph_client.url_safety.validate_graph_base_url` so the exact
        same validation is reused by the endpoint builder and is unit
        tested in isolation. Normalised (trailing slash stripped) in place.

        Validated in every mode, including `local`: the field has a safe
        default and validating it unconditionally keeps the invariant
        simple (the trusted base is always well-formed) with no runtime
        cost in offline mode.
        """
        # Imported lazily to avoid a config <-> graph_client import cycle at
        # module load time.
        from core.graph_client.url_safety import UnsafeGraphUrlError, validate_graph_base_url

        try:
            normalised = validate_graph_base_url(self.graph_base_url)
        except UnsafeGraphUrlError as exc:
            raise ValueError(f"Invalid GRAPH_BASE_URL: {exc}") from exc
        object.__setattr__(self, "graph_base_url", normalised)
        return self

    @model_validator(mode="after")
    def _normalize_vector_store_backend(self):
        """
        Normalize `vector_store_backend` to lowercase and reject any value
        other than the two supported backends. A typo in VECTOR_STORE_BACKEND
        must fail loudly at startup rather than silently defaulting to one
        backend. (Temporary toggle — see the field definition.)
        """
        allowed = {"weaviate", "faiss"}
        value = (self.vector_store_backend or "").strip().lower()
        if value not in allowed:
            raise ValueError(
                f"VECTOR_STORE_BACKEND={self.vector_store_backend!r} is invalid; "
                f"must be one of {sorted(allowed)}"
            )
        object.__setattr__(self, "vector_store_backend", value)
        return self

    @model_validator(mode="after")
    def _resolve_postgres_dsn(self):
        """
        Build `postgres_dsn` from AWS Secrets Manager credentials plus the
        non-secret POSTGRES_HOST / POSTGRES_PORT / POSTGRES_OPTIONS.

        PostgreSQL is considered "enabled" only when POSTGRES_HOST is set
        (mirroring the previous "empty DSN = disabled, use JSON fallback"
        contract). When no host is configured, `postgres_dsn` stays empty
        and NO AWS call is made — keeping local/offline runs free of AWS.

        When a host IS configured, the user/password/db are fetched from the
        secret and URL-encoded into a standard asyncpg DSN. Any DSN provided
        via .env is intentionally overwritten so credentials can never leak
        in from the environment. Failures here (missing/empty keys, AWS
        access errors, invalid JSON) raise and fail startup loudly rather
        than silently falling back to a credential-less connection.

        The password is never logged.

        Escape hatch: when POSTGRES_HOST is NOT set but a `postgres_dsn` was
        supplied directly (e.g. a test monkeypatching POSTGRES_DSN, or a
        local override), that value is kept as-is and NO AWS call is made.
        The committed `.env` never carries credentials this way — the
        production path is always "POSTGRES_HOST set -> fetch creds from
        Secrets Manager".
        """
        host = (self.postgres_host or "").strip()
        if not host:
            # No host configured: keep any explicitly-provided DSN (tests /
            # local override), otherwise Postgres stays disabled (JSON-file
            # fallback). No Secrets Manager call in this branch.
            return self

        dsn = _build_postgres_dsn_from_secret(
            secret_name=self.aws_secret_name,
            region_name=self.aws_region,
            host=host,
            port=self.postgres_port,
            options=(self.postgres_options or "").strip(),
        )
        object.__setattr__(self, "postgres_dsn", dsn)
        return self

    @property
    def email_backend_is_postgres(self) -> bool:
        """
        True when the email agent's three-layer storage (PostgreSQL +
        Weaviate + Redis) is active. There is no CSV backend anymore, so
        this is simply "is a PostgreSQL DSN configured". When False (no
        DSN), the email schema/startup wiring is skipped.
        """
        return bool((self.postgres_dsn or "").strip())

    @property
    def redis_enabled(self) -> bool:
        """
        True when a Redis read cache is configured (REDIS_HOST set). When
        False the cache layer is a transparent no-op and reads go straight
        to Weaviate/PostgreSQL — Redis is disposable, so disabling it only
        costs cache hits, never correctness.
        """
        return bool((self.redis_host or "").strip())

    @property
    def is_local_mail_mode(self) -> bool:
        """True when mail/calendar should be served from local files, not Graph API."""
        return self.graph_auth_mode.lower() == "local"

    @property
    def resolved_eml_storage_dir(self) -> Path:
        """Absolute path to the `.eml` mailbox folder, creating it if needed."""
        eml_path = Path(self.eml_storage_dir)
        if not eml_path.is_absolute():
            eml_path = BACKEND_ROOT / eml_path
        eml_path.mkdir(parents=True, exist_ok=True)
        return eml_path

    @property
    def resolved_local_calendar_path(self) -> Path:
        """Absolute path to the local-mode dummy calendar JSON."""
        calendar_path = Path(self.local_calendar_path)
        if not calendar_path.is_absolute():
            calendar_path = BACKEND_ROOT / calendar_path
        return calendar_path

    @property
    def resolved_local_calendar_history_path(self) -> Path:
        """Absolute path to the local-mode past-occurrence calendar JSON."""
        calendar_path = Path(self.local_calendar_history_path)
        if not calendar_path.is_absolute():
            calendar_path = BACKEND_ROOT / calendar_path
        return calendar_path

    @property
    def email_cluster_anchor_tokens(self) -> frozenset[str]:
        """Post-meeting subject tokens used as fallback anchor detection."""
        return frozenset(
            token.strip().lower()
            for token in self.email_cluster_anchor_tokens_raw.split(",")
            if token.strip()
        )

    @property
    def meeting_anchor_mom_body_keywords(self) -> tuple[str, ...]:
        """Lowercase MoM/minutes keywords for subject/body detection."""
        return tuple(
            token.strip().lower()
            for token in self.meeting_anchor_mom_body_keywords_raw.split(",")
            if token.strip()
        )

    @property
    def resolved_email_fetch_lookback_days(self) -> int:
        """Days of mail history to fetch before each pipeline run."""
        if self.email_pipeline_fetch_received_after_days > 0:
            return self.email_pipeline_fetch_received_after_days
        return self.meeting_history_lookback_days

    @property
    def effective_text_model(self) -> str:
        """
        Return the model name to use for pure text-generation tasks
        (summarization, classification, brief writing). Falls back to the
        vision model when a dedicated text model has not been configured,
        matching the current "one model for everything" setup while still
        allowing a future split without any code changes.
        """
        return self.ollama_text_model or self.ollama_vision_model

    @property
    def bu_departments(self) -> list[Department]:
        """
        Parse `bu_departments_raw` (a JSON array of {"name", "summary"})
        into `Department` objects. Malformed/empty entries are skipped;
        a completely unparseable value falls back to an empty list so a
        bad `.env` edit degrades gracefully (clustering then simply has
        no department constraints) rather than crashing startup.
        """
        try:
            parsed = json.loads(self.bu_departments_raw)
        except (json.JSONDecodeError, TypeError):
            return []
        if not isinstance(parsed, list):
            return []
        departments: list[Department] = []
        for entry in parsed:
            if not isinstance(entry, dict):
                continue
            name = str(entry.get("name", "")).strip()
            if not name:
                continue
            departments.append(Department(
                name=name,
                summary=str(entry.get("summary", "")).strip(),
                view=str(entry.get("view", "")).strip(),
            ))
        return departments

    @property
    def bu_department_names(self) -> list[str]:
        """The canonical department names only (order preserved), for prompt constraints and validation."""
        return [d.name for d in self.bu_departments]

    @property
    def department_view_map(self) -> dict[str, str]:
        """
        Map each department name (lowercased) to its frontend sidebar/
        route view key (e.g. "BB&RL" -> "Retail"). Used by the calendar
        agent to set `CalendarEvent.matched_view` to a value the frontend
        can route on. Departments with no configured `view` are omitted.
        """
        return {d.name.lower(): d.view for d in self.bu_departments if d.view}

    @property
    def mom_attachment_keywords(self) -> list[str]:
        """Parse `mom_attachment_keywords_raw` (comma-separated) into a clean list of trimmed keywords."""
        return [kw.strip() for kw in self.mom_attachment_keywords_raw.split(",") if kw.strip()]

    @property
    def department_tagging_example_phrases(self) -> dict[str, list[str]]:
        """
        Parse `department_tagging_example_phrases_raw` (a JSON object of
        {department name -> [phrase, ...]}) into a dict. Malformed/empty
        input degrades gracefully to an empty dict, so a bad `.env` edit
        just means summary-only centroids rather than a startup crash.
        Only string phrases under known-shaped entries are kept.
        """
        try:
            parsed = json.loads(self.department_tagging_example_phrases_raw)
        except (json.JSONDecodeError, TypeError):
            return {}
        if not isinstance(parsed, dict):
            return {}
        out: dict[str, list[str]] = {}
        for name, phrases in parsed.items():
            if not isinstance(name, str) or not isinstance(phrases, list):
                continue
            clean = [str(p).strip() for p in phrases if str(p).strip()]
            if clean:
                out[name.strip()] = clean
        return out

    @property
    def resolved_bu_email_data_dir(self) -> Path:
        """Return the absolute path to the email agent's local data directory, creating it if needed."""
        data_path = BACKEND_ROOT / self.bu_email_data_dir
        data_path.mkdir(parents=True, exist_ok=True)
        return data_path

    def weaviate_collection_name(self, namespace: str) -> str:
        """
        Return the Weaviate collection name for a given email-agent vector
        namespace (``"clusters"`` or ``"meetings"``), e.g.
        ``"EmailAgentClusters"``. Keeps the two namespaces in separate
        collections so they never collide, mirroring the old per-namespace
        FAISS index files.
        """
        return f"{self.weaviate_email_collection_prefix}{namespace.capitalize()}"

    @property
    def resolved_faiss_index_dir(self) -> Path:
        """
        Absolute path to the local FAISS index directory, creating it if
        needed. Only meaningful when VECTOR_STORE_BACKEND="faiss". Relative
        to BACKEND_ROOT unless an absolute path is configured. (Temporary —
        part of the removable FAISS backend under `core/faiss/`.)
        """
        index_path = Path(self.faiss_index_dir)
        if not index_path.is_absolute():
            index_path = BACKEND_ROOT / index_path
        index_path.mkdir(parents=True, exist_ok=True)
        return index_path

    @property
    def resolved_log_dir(self) -> Path:
        """Return the absolute path to the log directory, creating it if needed."""
        log_path = BACKEND_ROOT / self.log_dir
        log_path.mkdir(parents=True, exist_ok=True)
        return log_path


def _build_postgres_dsn_from_secret(
    *,
    secret_name: str,
    region_name: str,
    host: str,
    port: int,
    options: str = "",
) -> str:
    """
    Fetch ``POSTGRES_USER`` / ``POSTGRES_PASSWORD`` / ``POSTGRES_DB`` from
    the AWS Secrets Manager JSON secret and assemble a standard asyncpg
    DSN:  ``postgresql://<user>:<password>@<host>:<port>/<db>[?<options>]``.

    User, password and db are percent-encoded so special characters (``@``,
    ``:``, ``/`` …) in a credential can never corrupt the DSN. The password
    is never logged. Raises ``core.secrets_manager.SecretsManagerError`` on
    any AWS access error, a missing secret, invalid JSON, or missing/empty
    required keys — surfacing a clear, secret-free failure at startup.

    Imported locally so modules that only need other settings (and tests
    that stub the DSN) do not pull in boto3 unnecessarily.
    """
    from core.secrets_manager import SecretsManagerClient

    client = SecretsManagerClient(secret_name=secret_name, region_name=region_name)
    creds = client.get_postgres_credentials()

    user = quote(creds["user"], safe="")
    password = quote(creds["password"], safe="")
    db = quote(creds["db"], safe="")

    dsn = f"postgresql://{user}:{password}@{host}:{port}/{db}"
    if options:
        dsn = f"{dsn}?{options}"
    return dsn


@lru_cache
def get_settings() -> Settings:
    """
    Return a process-wide cached `Settings` instance.

    Cached via `lru_cache` so `.env` is parsed once per process. Tests
    that need a different configuration should call
    `get_settings.cache_clear()` after monkeypatching environment
    variables or the `.env` file location.
    """
    return Settings()
