"""
Pipeline metrics collection and reporting for the email clustering pipeline.

Tracks all pipeline stages and metrics:
- Mailbox fetch, normalization, deduplication
- Attachment retrieval and parsing
- Candidate retrieval and relevance scoring
- Embedding generation
- Clustering
- Cluster summarization
- Action extraction
- Meeting synthesis
- LLM call stats (calls, tokens in/out, attempts)
- Embedding stats
"""

from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any, Optional
import logging
import json

from core.config import get_settings

logger = logging.getLogger("agent_email.pipeline_metrics")


@dataclass
class PipelineMetrics:
    """Aggregated pipeline metrics for a single run."""
    
    run_id: str
    timestamp_start: str
    timestamp_end: Optional[str] = None
    
    # Mailbox fetch stage
    mailbox_fetch_count: int = 0
    mailbox_fetch_error: Optional[str] = None
    
    # Email normalization
    email_normalized_count: int = 0
    email_normalization_errors: int = 0
    
    # Thread deduplication
    unique_emails_count: int = 0
    duplicate_emails_count: int = 0
    
    # Attachment retrieval
    attachments_found: int = 0
    attachments_missing: int = 0
    attachment_timeout_count: int = 0
    
    # Attachment parsing
    attachments_parsed: int = 0
    attachments_parse_errors: int = 0
    
    # Candidate retrieval
    candidates_fetched: int = 0
    candidates_after_relevance_filter: int = 0
    
    # Embedding generation
    embeddings_generated: int = 0
    embedding_errors: int = 0
    embedding_cache_hits: int = 0
    
    # Clustering
    clusters_generated: int = 0
    clusters_reused: int = 0
    clusters_regenerated: int = 0
    clustering_algorithm: str = ""
    
    # Cluster summarization
    clusters_summarized: int = 0
    clusters_summary_failed: int = 0
    
    # Action extraction
    action_items_extracted: int = 0
    action_extraction_errors: int = 0
    
    # Meeting synthesis
    meetings_matched: int = 0
    meetings_reranked: int = 0
    
    # LLM call statistics
    llm_calls_total: int = 0
    llm_calls_by_type: dict[str, int] = field(default_factory=dict)  # e.g. {"topic_tagging": 5, "synthesis": 3}
    llm_tokens_in: int = 0
    llm_tokens_out: int = 0
    llm_errors: int = 0
    llm_retries_exhausted: int = 0
    
    # Processing summary
    total_duration_seconds: Optional[float] = None
    pipeline_status: str = "in_progress"  # or "completed", "failed"
    pipeline_error: Optional[str] = None


class PipelineMetricsCollector:
    """Singleton collector for pipeline metrics during a run."""
    
    _instance: Optional["PipelineMetricsCollector"] = None
    _current_run_id: Optional[str] = None
    _metrics: Optional[PipelineMetrics] = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    @staticmethod
    def start_run(run_id: str) -> PipelineMetrics:
        """Initialize metrics for a new pipeline run."""
        collector = PipelineMetricsCollector()
        collector._current_run_id = run_id
        collector._metrics = PipelineMetrics(
            run_id=run_id,
            timestamp_start=datetime.now(timezone.utc).isoformat(),
        )
        logger.info("[METRICS] Pipeline run started: run_id=%s", run_id)
        return collector._metrics
    
    @staticmethod
    def get_current() -> Optional[PipelineMetrics]:
        """Get the current run's metrics (or None if no run is active)."""
        collector = PipelineMetricsCollector()
        return collector._metrics
    
    @staticmethod
    def end_run(status: str = "completed", error: Optional[str] = None) -> Optional[PipelineMetrics]:
        """Finalize metrics for current run and return them."""
        collector = PipelineMetricsCollector()
        if collector._metrics is None:
            return None
        
        metrics = collector._metrics
        metrics.timestamp_end = datetime.now(timezone.utc).isoformat()
        metrics.pipeline_status = status
        metrics.pipeline_error = error
        
        # Calculate duration
        try:
            start = datetime.fromisoformat(metrics.timestamp_start.replace("Z", "+00:00"))
            end = datetime.fromisoformat(metrics.timestamp_end.replace("Z", "+00:00"))
            metrics.total_duration_seconds = (end - start).total_seconds()
        except Exception:
            pass
        
        logger.info(
            "[METRICS] Pipeline run completed: run_id=%s status=%s duration=%.2fs "
            "emails=%d clusters=%d llm_calls=%d tokens_in=%d tokens_out=%d",
            metrics.run_id, status, metrics.total_duration_seconds or 0,
            metrics.unique_emails_count, metrics.clusters_generated,
            metrics.llm_calls_total, metrics.llm_tokens_in, metrics.llm_tokens_out,
        )
        
        _log_metrics_report(metrics)
        
        collector._current_run_id = None
        collector._metrics = None
        
        return metrics
    
    @staticmethod
    def record_mailbox_fetch(count: int, error: Optional[str] = None) -> None:
        """Record mailbox fetch stage."""
        m = PipelineMetricsCollector.get_current()
        if m:
            m.mailbox_fetch_count = count
            m.mailbox_fetch_error = error
            logger.info("[METRICS-FETCH] mailbox_count=%d error=%s", count, error or "none")
    
    @staticmethod
    def record_normalization(normalized: int, errors: int) -> None:
        """Record email normalization stage."""
        m = PipelineMetricsCollector.get_current()
        if m:
            m.email_normalized_count = normalized
            m.email_normalization_errors = errors
            logger.info("[METRICS-NORM] normalized=%d errors=%d", normalized, errors)
    
    @staticmethod
    def record_deduplication(unique: int, duplicates: int) -> None:
        """Record thread deduplication stage."""
        m = PipelineMetricsCollector.get_current()
        if m:
            m.unique_emails_count = unique
            m.duplicate_emails_count = duplicates
            logger.info("[METRICS-DEDUP] unique=%d duplicates=%d", unique, duplicates)
    
    @staticmethod
    def record_attachment_retrieval(found: int, missing: int, timeouts: int) -> None:
        """Record attachment retrieval stage."""
        m = PipelineMetricsCollector.get_current()
        if m:
            m.attachments_found = found
            m.attachments_missing = missing
            m.attachment_timeout_count = timeouts
            logger.info(
                "[METRICS-ATTACH-RETR] found=%d missing=%d timeouts=%d",
                found, missing, timeouts,
            )
    
    @staticmethod
    def record_attachment_parsing(parsed: int, errors: int) -> None:
        """Record attachment parsing stage."""
        m = PipelineMetricsCollector.get_current()
        if m:
            m.attachments_parsed = parsed
            m.attachments_parse_errors = errors
            logger.info("[METRICS-ATTACH-PARSE] parsed=%d errors=%d", parsed, errors)
    
    @staticmethod
    def record_candidate_retrieval(fetched: int, after_filter: int) -> None:
        """Record candidate retrieval and relevance filtering stage."""
        m = PipelineMetricsCollector.get_current()
        if m:
            m.candidates_fetched = fetched
            m.candidates_after_relevance_filter = after_filter
            logger.info(
                "[METRICS-CANDIDATE] fetched=%d after_filter=%d filter_rate=%.1f%%",
                fetched, after_filter,
                (after_filter / fetched * 100) if fetched > 0 else 0,
            )
    
    @staticmethod
    def record_embedding(count: int, errors: int, cache_hits: int = 0) -> None:
        """Record embedding generation stage."""
        m = PipelineMetricsCollector.get_current()
        if m:
            m.embeddings_generated = count
            m.embedding_errors = errors
            m.embedding_cache_hits = cache_hits
            logger.info(
                "[METRICS-EMBED] generated=%d errors=%d cache_hits=%d",
                count, errors, cache_hits,
            )
    
    @staticmethod
    def record_clustering(
        generated: int,
        reused: int = 0,
        regenerated: int = 0,
        algorithm: str = "",
    ) -> None:
        """Record clustering stage."""
        m = PipelineMetricsCollector.get_current()
        if m:
            m.clusters_generated = generated
            m.clusters_reused = reused
            m.clusters_regenerated = regenerated
            m.clustering_algorithm = algorithm
            logger.info(
                "[METRICS-CLUSTER] generated=%d reused=%d regenerated=%d algo=%s",
                generated, reused, regenerated, algorithm,
            )
    
    @staticmethod
    def record_summarization(summarized: int, failed: int) -> None:
        """Record cluster summarization stage."""
        m = PipelineMetricsCollector.get_current()
        if m:
            m.clusters_summarized = summarized
            m.clusters_summary_failed = failed
            logger.info("[METRICS-SUMMARY] summarized=%d failed=%d", summarized, failed)
    
    @staticmethod
    def record_action_extraction(extracted: int, errors: int) -> None:
        """Record action extraction stage."""
        m = PipelineMetricsCollector.get_current()
        if m:
            m.action_items_extracted = extracted
            m.action_extraction_errors = errors
            logger.info("[METRICS-ACTIONS] extracted=%d errors=%d", extracted, errors)
    
    @staticmethod
    def record_meeting_synthesis(matched: int, reranked: int) -> None:
        """Record meeting synthesis stage."""
        m = PipelineMetricsCollector.get_current()
        if m:
            m.meetings_matched = matched
            m.meetings_reranked = reranked
            logger.info("[METRICS-MEETING] matched=%d reranked=%d", matched, reranked)
    
    @staticmethod
    def record_llm_call(
        call_type: str,
        tokens_in: int = 0,
        tokens_out: int = 0,
        error: bool = False,
        retry_exhausted: bool = False,
    ) -> None:
        """Record a single LLM call with token count."""
        m = PipelineMetricsCollector.get_current()
        if m:
            m.llm_calls_total += 1
            m.llm_calls_by_type[call_type] = m.llm_calls_by_type.get(call_type, 0) + 1
            m.llm_tokens_in += tokens_in
            m.llm_tokens_out += tokens_out
            if error:
                m.llm_errors += 1
            if retry_exhausted:
                m.llm_retries_exhausted += 1
            
            logger.debug(
                "[METRICS-LLM] type=%s tokens_in=%d tokens_out=%d error=%s",
                call_type, tokens_in, tokens_out, error,
            )


def _log_metrics_report(metrics: PipelineMetrics) -> None:
    """Log a comprehensive metrics report."""
    settings = get_settings()
    metrics_dir = getattr(settings, "resolved_log_dir", "logs")
    
    # Log to console as info-level summary
    logger.info(
        "\n"
        "=== EMAIL PIPELINE METRICS SUMMARY ===\n"
        "Run ID: %s\n"
        "Status: %s\n"
        "Duration: %.2f seconds\n"
        "\n"
        "--- MAILBOX & EMAIL PROCESSING ---\n"
        "Mailbox fetch: %d\n"
        "Emails normalized: %d (errors: %d)\n"
        "Unique emails: %d\n"
        "Duplicate emails: %d\n"
        "\n"
        "--- ATTACHMENTS ---\n"
        "Attachments found: %d\n"
        "Attachments missing: %d\n"
        "Attachment timeouts: %d\n"
        "Attachments parsed: %d (errors: %d)\n"
        "\n"
        "--- CANDIDATES ---\n"
        "Candidates fetched: %d\n"
        "Candidates after relevance filter: %d\n"
        "\n"
        "--- EMBEDDINGS ---\n"
        "Embeddings generated: %d (errors: %d)\n"
        "Embedding cache hits: %d\n"
        "\n"
        "--- CLUSTERING ---\n"
        "Clusters generated: %d\n"
        "Clusters reused: %d\n"
        "Clusters regenerated: %d\n"
        "Clustering algorithm: %s\n"
        "\n"
        "--- SUMMARIZATION ---\n"
        "Clusters summarized: %d\n"
        "Clusters summary failed: %d\n"
        "Action items extracted: %d (errors: %d)\n"
        "\n"
        "--- MEETING SYNTHESIS ---\n"
        "Meetings matched: %d\n"
        "Meetings reranked: %d\n"
        "\n"
        "--- LLM STATISTICS ---\n"
        "Total LLM calls: %d\n"
        "LLM calls by type: %s\n"
        "LLM tokens in: %d\n"
        "LLM tokens out: %d\n"
        "LLM errors: %d\n"
        "LLM retries exhausted: %d\n"
        "=== END METRICS SUMMARY ===\n",
        metrics.run_id,
        metrics.pipeline_status,
        metrics.total_duration_seconds or 0,
        metrics.mailbox_fetch_count,
        metrics.email_normalized_count,
        metrics.email_normalization_errors,
        metrics.unique_emails_count,
        metrics.duplicate_emails_count,
        metrics.attachments_found,
        metrics.attachments_missing,
        metrics.attachment_timeout_count,
        metrics.attachments_parsed,
        metrics.attachments_parse_errors,
        metrics.candidates_fetched,
        metrics.candidates_after_relevance_filter,
        metrics.embeddings_generated,
        metrics.embedding_errors,
        metrics.embedding_cache_hits,
        metrics.clusters_generated,
        metrics.clusters_reused,
        metrics.clusters_regenerated,
        metrics.clustering_algorithm,
        metrics.clusters_summarized,
        metrics.clusters_summary_failed,
        metrics.action_items_extracted,
        metrics.action_extraction_errors,
        metrics.meetings_matched,
        metrics.meetings_reranked,
        metrics.llm_calls_total,
        json.dumps(metrics.llm_calls_by_type),
        metrics.llm_tokens_in,
        metrics.llm_tokens_out,
        metrics.llm_errors,
        metrics.llm_retries_exhausted,
    )
    
    # Optionally write to JSON file for post-processing
    try:
        metrics_file = Path(metrics_dir) / f"pipeline_metrics_{metrics.run_id}.json"
        metrics_file.parent.mkdir(parents=True, exist_ok=True)
        with open(metrics_file, "w") as f:
            json.dump(asdict(metrics), f, indent=2, default=str)
        logger.info("[METRICS-FILE] Wrote metrics to %s", metrics_file)
    except Exception as exc:
        logger.warning("[METRICS-FILE] Failed to write metrics file: %s", exc)


from pathlib import Path
