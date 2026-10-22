"""
agent_email/fact_extraction.py
================================
Deterministic pre-processing layer for extracting and normalizing facts
from tables before LLM summarization.

Extracts all facts from tables into a structured format:
  ENTITY | VALUE | TIME/CONTEXT | STATUS | RELATION | SOURCE_TEXT

Features:
- Generic and deterministic (no hardcoded business-specific terms)
- Preserves table headers and surrounding context
- Normalizes explicit statuses only when supported by source
- Normalizes dates without inferring status
- Detects explicit relationships (e.g., baseline → current, actual → target)
- Preserves original source text for every normalized fact
"""

import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


# =============================================================================
# Valid statuses - only normalize explicit statuses from this list
# =============================================================================
VALID_STATUSES = frozenset({
    "actual",
    "baseline",
    "current",
    "target",
    "forecast",
    "planned",
    "proposed",
    "approved",
    "pending",
    "completed",
    "in_progress",
    "expected",
    "deadline",
    "milestone",
    "benchmark",
    "threshold",
})


# =============================================================================
# Status aliases - normalize these to their canonical form
# =============================================================================
STATUS_ALIASES = {
    "achieved": "actual",
    "achieved_value": "actual",
    "actual_value": "actual",
    "base": "baseline",
    "base_value": "baseline",
    "baseline_value": "baseline",
    "curr": "current",
    "current_value": "current",
    "est": "forecast",
    "estimated": "forecast",
    "estimation": "forecast",
    "expected_value": "forecast",
    "forecast_value": "forecast",
    "tgt": "target",
    "target_value": "target",
    "plan": "planned",
    "planned_value": "planned",
    "prop": "proposed",
    "proposed_value": "proposed",
    "appr": "approved",
    "approv": "approved",
    "approved_value": "approved",
    "pend": "pending",
    "pending_value": "pending",
    "comp": "completed",
    "complete": "completed",
    "completed_value": "completed",
    "fin": "completed",
    "finished": "completed",
    "done": "completed",
    "inpr": "in_progress",
    "in_prog": "in_progress",
    "in_progress_value": "in_progress",
    "exp": "expected",
    "exp_value": "expected",
    "expct": "expected",
    "deadline_value": "deadline",
    "dl": "deadline",
    "dl_value": "deadline",
    "mstn": "milestone",
    "mstone": "milestone",
    "mstn_value": "milestone",
    "bench": "benchmark",
    "benchmark_value": "benchmark",
    "thrsh": "threshold",
    "threshold_value": "threshold",
}


# =============================================================================
# Date patterns for normalization
# =============================================================================
DATE_PATTERNS = [
    # ISO formats
    (r"(\d{4}-\d{2}-\d{2}(?:T\d{2}:\d{2}:\d{2})?(?:\+\d{2}:\d{2}|Z)?)", "iso"),
    # DD/MM/YYYY or MM/DD/YYYY
    (r"(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})", "dmy"),
    # Month name formats
    (r"(\d{1,2}\s+(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+\d{2,4})", "month_name"),
    # Quarter formats
    (r"(Q[1-4]\s*\d{4})", "quarter"),
    # Year only
    (r"(\b\d{4}\b)", "year"),
]


# =============================================================================
# Relationship patterns for detection
# =============================================================================
RELATIONSHIP_PATTERNS = [
    # Explicit arrow patterns
    (r"(\w[\w\s]+?)\s*→\s*(\w[\w\s]+?)(?:\s*:\s*([\d,\.]+%?\s*(?:Cr|crore|bps|Rs\.?|INR|₹)?))?", "arrow"),
    (r"(\w[\w\s]+?)\s*-->\s*(\w[\w\s]+?)(?:\s*:\s*([\d,\.]+%?\s*(?:Cr|crore|bps|Rs\.?|INR|₹)?))?", "arrow_long"),
    (r"(\w[\w\s]+?)\s*-\s*(\w[\w\s]+?)(?:\s*:\s*([\d,\.]+%?\s*(?:Cr|crore|bps|Rs\.?|INR|₹)?))?", "dash"),
    # Text patterns
    (r"(baseline)\s+(?:to|→|-->\s*)\s*(current|actual|target|forecast)", "text_arrow"),
    (r"(previous|prior|last\s+month|last\s+quarter|last\s+year)\s+(?:to|→|-->\s*)\s*(current|actual|target|forecast)", "text_arrow"),
    (r"(actual|actual_value)\s+(?:to|→|-->\s*)\s*(target|tgt|target_value)", "text_arrow"),
    (r"(from|vs|compared\s+to|vs\.)\s+(baseline|previous|prior|last)", "from_to"),
]


@dataclass
class ExtractedFact:
    """Represents a single extracted fact from a table."""
    entity: str
    value: str
    time_context: str
    status: str
    relation: Optional[str] = None
    source_text: str = ""
    table_context: str = ""
    row_index: int = -1
    col_index: int = -1
    original_row: str = ""
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "entity": self.entity,
            "value": self.value,
            "time_context": self.time_context,
            "status": self.status,
            "relation": self.relation,
            "source_text": self.source_text,
            "table_context": self.table_context,
            "row_index": self.row_index,
            "col_index": self.col_index,
            "original_row": self.original_row,
        }
    
    def to_string(self) -> str:
        """Convert to pipe-delimited string format."""
        relation_str = self.relation if self.relation else "unknown"
        return f"{self.entity} | {self.value} | {self.time_context} | {self.status} | {relation_str} | {self.source_text}"


@dataclass
class ExtractedTable:
    """Represents a table with extracted facts."""
    table_index: int
    title: str = ""
    headers: list[str] = field(default_factory=list)
    facts: list[ExtractedFact] = field(default_factory=list)
    raw_text: str = ""
    surrounding_context: str = ""
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "table_index": self.table_index,
            "title": self.title,
            "headers": self.headers,
            "facts": [fact.to_dict() for fact in self.facts],
            "raw_text": self.raw_text,
            "surrounding_context": self.surrounding_context,
        }
    
    def to_markdown(self) -> str:
        """Convert to markdown table format."""
        if not self.facts:
            return f"### {self.title}\n\n(no facts extracted)"
        
        lines = [f"### {self.title}" if self.title else "### Extracted Table"]
        lines.append("")
        
        # Build headers
        headers = ["Entity", "Value", "Time/Context", "Status", "Relation", "Source Text"]
        lines.append("| " + " | ".join(headers) + " |")
        lines.append("|" + "|".join(["---"] * len(headers)) + "|")
        
        # Build rows
        for fact in self.facts:
            row = [
                fact.entity,
                fact.value,
                fact.time_context,
                fact.status,
                fact.relation or "unknown",
                fact.source_text[:50] + "..." if len(fact.source_text) > 50 else fact.source_text,
            ]
            lines.append("| " + " | ".join(row) + " |")
        
        return "\n".join(lines)


@dataclass
class FactExtractionResult:
    """Result of fact extraction from a document."""
    tables: list[ExtractedTable] = field(default_factory=list)
    facts: list[ExtractedFact] = field(default_factory=list)
    raw_text: str = ""
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "tables": [table.to_dict() for table in self.tables],
            "facts": [fact.to_dict() for fact in self.facts],
            "raw_text": self.raw_text,
        }
    
    def to_markdown(self) -> str:
        """Convert to markdown format."""
        sections = []
        
        if self.facts:
            sections.append("## Extracted Facts")
            sections.append("")
            for fact in self.facts:
                sections.append(fact.to_string())
            sections.append("")
        
        for table in self.tables:
            sections.append(table.to_markdown())
            sections.append("")
        
        return "\n".join(sections)


def normalize_status(status_text: str) -> str:
    """
    Normalize a status text to one of the valid statuses.
    
    If the status is not in the valid set or a recognized alias,
    returns "unknown".
    """
    if not status_text:
        return "unknown"
    
    status_lower = status_text.strip().lower()
    
    # Check for exact match in valid statuses
    if status_lower in VALID_STATUSES:
        return status_lower
    
    # Check for aliases
    for alias, normalized in STATUS_ALIASES.items():
        if status_lower.startswith(alias):
            return normalized
    
    # Check if status contains any valid status keyword
    for valid_status in VALID_STATUSES:
        if valid_status in status_lower:
            return valid_status
    
    return "unknown"


def normalize_date(date_text: str) -> str:
    """
    Normalize a date text to a standard format (YYYY-MM-DD when possible).
    
    Returns the original text if the date cannot be normalized.
    Does NOT infer status - unlabeled dates remain "unknown".
    """
    if not date_text:
        return ""
    
    date_text = date_text.strip()
    
    # Try to match patterns and normalize
    for pattern, pattern_type in DATE_PATTERNS:
        match = re.search(pattern, date_text, re.IGNORECASE)
        if match:
            matched_date = match.group(1)
            
            if pattern_type == "iso":
                # Extract date portion from ISO format
                date_match = re.match(r"(\d{4}-\d{2}-\d{2})", matched_date)
                if date_match:
                    return date_match.group(1)
            
            elif pattern_type == "dmy":
                # Try to parse DD/MM/YYYY or MM/DD/YYYY
                parts = re.split(r"[/-]", matched_date)
                if len(parts) == 3:
                    try:
                        d, m, y = parts
                        # Handle 2-digit years
                        if len(y) == 2:
                            y = "20" + y if int(y) < 50 else "19" + y
                        # Try to determine if DD/MM or MM/DD
                        if int(d) > 12:
                            # DD/MM/YYYY format
                            return f"{y}-{m.zfill(2)}-{d.zfill(2)}"
                        elif int(m) > 12:
                            # MM/DD/YYYY format
                            return f"{y}-{d.zfill(2)}-{m.zfill(2)}"
                        else:
                            # Ambiguous, assume DD/MM/YYYY (common format)
                            return f"{y}-{m.zfill(2)}-{d.zfill(2)}"
                    except (ValueError, IndexError):
                        pass
            
            elif pattern_type == "month_name":
                # Parse "15 Jan 2024" format
                month_map = {
                    "jan": "01", "january": "01",
                    "feb": "02", "february": "02",
                    "mar": "03", "march": "03",
                    "apr": "04", "april": "04",
                    "may": "05",
                    "jun": "06", "june": "06",
                    "jul": "07", "july": "07",
                    "aug": "08", "august": "08",
                    "sep": "09", "september": "09",
                    "oct": "10", "october": "10",
                    "nov": "11", "november": "11",
                    "dec": "12", "december": "12",
                }
                match = re.match(r"(\d{1,2})\s+(\w+)\s+(\d{4})", matched_date, re.IGNORECASE)
                if match:
                    day, month, year = match.groups()
                    month_num = month_map.get(month.lower(), "01")
                    return f"{year}-{month_num}-{day.zfill(2)}"
            
            elif pattern_type == "quarter":
                # Parse "Q1 2024" format
                match = re.match(r"Q([1-4])\s*(\d{4})", matched_date, re.IGNORECASE)
                if match:
                    quarter, year = match.groups()
                    # Convert quarter to month range
                    quarter_months = {
                        "1": ("01", "03"),
                        "2": ("04", "06"),
                        "3": ("07", "09"),
                        "4": ("10", "12"),
                    }
                    start_month, end_month = quarter_months[quarter]
                    return f"{year}-{start_month}/Q{quarter}"
            
            elif pattern_type == "year":
                # Just return the year
                return matched_date
    
    return date_text


def detect_relationship(text: str) -> Optional[str]:
    """
    Detect explicit relationships in text (e.g., baseline → current).
    
    Returns the relationship type or None if no relationship detected.
    """
    if not text:
        return None
    
    text_lower = text.lower()
    
    for pattern, pattern_type in RELATIONSHIP_PATTERNS:
        match = re.search(pattern, text_lower, re.IGNORECASE)
        if match:
            groups = match.groups()
            
            if pattern_type in ("arrow", "arrow_long", "dash"):
                # Format: ENTITY1 → ENTITY2 : VALUE
                source = groups[0].strip() if groups[0] else ""
                target = groups[1].strip() if groups[1] else ""
                if source and target:
                    return f"{source} → {target}"
            
            elif pattern_type == "text_arrow":
                # Format: baseline → current/actual/target
                source = groups[0].strip() if groups[0] else "unknown"
                target = groups[1].strip() if groups[1] else "unknown"
                if source and target:
                    return f"{source} → {target}"
            
            elif pattern_type == "from_to":
                # Format: from/versus baseline
                target = groups[1].strip() if groups[1] else "unknown"
                return f"from {target}"
    
    return None


def extract_facts_from_table_row(
    row_index: int,
    cells: list[str],
    headers: list[str],
    table_title: str = "",
    surrounding_context: str = "",
) -> list[ExtractedFact]:
    """
    Extract facts from a single table row.
    
    Args:
        row_index: Index of the row (0-based, excluding header)
        cells: List of cell values in the row
        headers: List of column headers
        table_title: Title of the table
        surrounding_context: Text surrounding the table
    
    Returns:
        List of ExtractedFact objects
    """
    facts = []
    
    if not cells or not headers:
        return facts
    
    # Determine the number of columns to process
    num_cols = min(len(cells), len(headers))
    
    # For each cell, extract as a fact
    for col_index in range(num_cols):
        cell_text = cells[col_index].strip()
        if not cell_text:
            continue
        
        # Determine entity name
        header = headers[col_index].strip() if col_index < len(headers) else ""
        entity = header if header else f"column_{col_index}"
        
        # Try to extract relationship from cell text
        relation = detect_relationship(cell_text)
        
        # Extract status from cell text
        status = normalize_status(cell_text)
        
        # If no explicit status, try to find one in headers or context
        if status == "unknown":
            for h in headers:
                header_status = normalize_status(h)
                if header_status != "unknown":
                    status = header_status
                    break
        
        # If still unknown, try to infer from position (lateral thinking)
        if status == "unknown" and col_index == 0:
            # First column is often the entity name, not a status column
            status = "unknown"
        
        # Create the fact
        fact = ExtractedFact(
            entity=entity,
            value=cell_text,
            time_context="",
            status=status,
            relation=relation,
            source_text=cell_text,
            table_context=table_title,
            row_index=row_index,
            col_index=col_index,
            original_row=" | ".join(cells[:num_cols]),
        )
        
        facts.append(fact)
    
    return facts


def extract_facts_from_table(
    table_text: str,
    table_index: int = 0,
    table_title: str = "",
    surrounding_context: str = "",
) -> ExtractedTable:
    """
    Extract facts from a table represented as text.
    
    Handles both structured table formats (markdown, pipe-delimited)
    and plain tab-delimited text.
    
    Args:
        table_text: Raw text of the table
        table_index: Index of the table in the document
        table_title: Optional title of the table
        surrounding_context: Text surrounding the table
    
    Returns:
        ExtractedTable object with facts
    """
    # Parse the table text into rows
    lines = [line.strip() for line in table_text.split("\n") if line.strip()]
    
    if not lines:
        return ExtractedTable(
            table_index=table_index,
            title=table_title,
            surrounding_context=surrounding_context,
            raw_text=table_text,
        )
    
    # Detect table format and extract headers/rows
    headers = []
    data_rows = []
    
    # Check if first line looks like a header (pipe-delimited or tab-delimited)
    first_line = lines[0]
    
    if "|" in first_line:
        # Pipe-delimited (markdown-style)
        headers = [h.strip() for h in first_line.split("|") if h.strip()]
        
        # Skip separator row if present (contains ---)
        start_idx = 1
        if start_idx < len(lines) and "---" in lines[start_idx]:
            start_idx += 1
        
        for i in range(start_idx, len(lines)):
            row_cells = [c.strip() for c in lines[i].split("|") if c.strip()]
            if row_cells:
                data_rows.append(row_cells)
    
    elif "\t" in first_line:
        # Tab-delimited
        headers = [h.strip() for h in first_line.split("\t") if h.strip()]
        start_idx = 1
        
        for i in range(start_idx, len(lines)):
            row_cells = [c.strip() for c in lines[i].split("\t") if c.strip()]
            if row_cells:
                data_rows.append(row_cells)
    
    else:
        # Plain text table - try to infer structure
        # Assume first line is header
        headers = first_line.split()
        for line in lines[1:]:
            row_cells = line.split()
            if row_cells:
                data_rows.append(row_cells)
    
    # Extract facts from each data row
    facts = []
    for row_idx, row_cells in enumerate(data_rows):
        row_facts = extract_facts_from_table_row(
            row_index=row_idx,
            cells=row_cells,
            headers=headers,
            table_title=table_title,
            surrounding_context=surrounding_context,
        )
        facts.extend(row_facts)
    
    return ExtractedTable(
        table_index=table_index,
        title=table_title,
        headers=headers,
        facts=facts,
        raw_text=table_text,
        surrounding_context=surrounding_context,
    )


def extract_tables_from_text(text: str) -> list[ExtractedTable]:
    """
    Extract all tables from text and convert them to facts.
    
    This is a generic table detection function that looks for
    table-like patterns in text (separated by blank lines, 
    containing pipe or tab delimiters).
    
    Args:
        text: Raw text to extract tables from
    
    Returns:
        List of ExtractedTable objects
    """
    tables = []
    
    # Split text into blocks (separated by 2+ newlines)
    blocks = re.split(r"\n\s*\n", text)
    
    for block_idx, block in enumerate(blocks):
        block = block.strip()
        if not block:
            continue
        
        # Check if block looks like a table
        lines = [line.strip() for line in block.split("\n") if line.strip()]
        
        if not lines or len(lines) < 2:
            continue
        
        # Look for table indicators
        has_pipe = any("|" in line for line in lines)
        has_tab = any("\t" in line for line in lines)
        has_header_pattern = len(lines[0].split("|" if has_pipe else "\t")) > 1
        
        if has_pipe or (has_tab and has_header_pattern):
            table_title = f"Table {block_idx + 1}"
            
            # Try to extract a title from preceding context
            # (this would require access to the original text structure)
            
            extracted = extract_facts_from_table(
                table_text=block,
                table_index=block_idx,
                table_title=table_title,
            )
            
            if extracted.facts:
                tables.append(extracted)
    
    return tables


def format_facts_for_llm(facts_result: FactExtractionResult) -> str:
    """
    Format extracted facts for LLM consumption.
    
    Returns a structured text format that includes:
    - All extracted facts in pipe-delimited format
    - Table context
    - Source text
    """
    if not facts_result.facts:
        return "(no facts extracted from tables)"
    
    sections = []
    
    # Summary section
    sections.append("EXTRACTED FACTS FROM TABLES")
    sections.append("=" * 80)
    sections.append(f"Total facts extracted: {len(facts_result.facts)}")
    sections.append(f"Tables processed: {len(facts_result.tables)}")
    sections.append("")
    
    # Facts section
    sections.append("FACTS:")
    sections.append("")
    
    for fact in facts_result.facts:
        sections.append(fact.to_string())
        if fact.table_context:
            sections.append(f"  [Table: {fact.table_context}]")
        if fact.source_text != fact.value:
            sections.append(f"  [Original: {fact.source_text}]")
        sections.append("")
    
    # Tables section (as markdown for better LLM understanding)
    for table in facts_result.tables:
        sections.append("TABLE AS MARKDOWN:")
        sections.append("-" * 40)
        sections.append(table.to_markdown())
        sections.append("")
    
    return "\n".join(sections)


def extract_and_format_tables(text: str) -> str:
    """
    High-level function: extract tables from text, convert to facts,
    and format for LLM consumption.
    
    Args:
        text: Raw text to process
    
    Returns:
        Formatted string of extracted facts
    """
    tables = extract_tables_from_text(text)
    facts = []
    
    for table in tables:
        facts.extend(table.facts)
    
    result = FactExtractionResult(
        tables=tables,
        facts=facts,
        raw_text=text,
    )
    
    return format_facts_for_llm(result)