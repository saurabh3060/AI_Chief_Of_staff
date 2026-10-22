"""
agent_email/table_extractor.py
================================
Table extraction utilities that preserve structure and relationships
for LLM processing.

Provides functions to extract tables from Word documents and Excel
files while maintaining:
- Row/column relationships
- Header vs data distinction
- Cell位置 information
- Table metadata
"""

import io
from dataclasses import dataclass, field
from typing import Optional

from docx import Document
from openpyxl import load_workbook


@dataclass
class TableCell:
    """Represents a single table cell with its position and content."""
    row_index: int
    col_index: int
    text: str
    row_span: int = 1
    col_span: int = 1
    is_header: bool = False


@dataclass
class TableRow:
    """Represents a table row with its cells."""
    row_index: int
    cells: list[TableCell]


@dataclass
class ExtractedTable:
    """Represents a complete extracted table with metadata."""
    table_index: int
    title: Optional[str] = None
    rows: list[TableRow] = field(default_factory=list)
    headers: list[str] = field(default_factory=list)
    num_rows: int = 0
    num_columns: int = 0
    has_header: bool = False
    
    def to_markdown(self) -> str:
        """Convert table to markdown format for LLM consumption."""
        if not self.rows:
            return ""
        
        lines = []
        if self.title:
            lines.append(f"### {self.title}")
        lines.append("")  # Empty line after title
        
        # Build markdown table
        for row in self.rows:
            cells = []
            for cell in row.cells:
                # Escape pipe characters in cell text
                cell_text = cell.text.replace("|", "\\|") if cell.text else ""
                cells.append(f" {cell_text} ")
            lines.append("|" + "|".join(cells) + "|")
            
            # Add separator row after header
            if row.row_index == 0 and self.has_header:
                separators = []
                for _ in row.cells:
                    separators.append("---")
                lines.append("|" + "|".join(separators) + "|")
        
        return "\n".join(lines)
    
    def to_json(self) -> dict:
        """Convert table to structured JSON format."""
        return {
            "table_index": self.table_index,
            "title": self.title,
            "num_rows": self.num_rows,
            "num_columns": self.num_columns,
            "has_header": self.has_header,
            "headers": self.headers,
            "rows": [
                {
                    "row_index": row.row_index,
                    "cells": [
                        {
                            "row_index": cell.row_index,
                            "col_index": cell.col_index,
                            "text": cell.text,
                            "row_span": cell.row_span,
                            "col_span": cell.col_span,
                            "is_header": cell.is_header
                        }
                        for cell in row.cells
                    ]
                }
                for row in self.rows
            ]
        }
    
    def to_structured_text(self) -> str:
        """Convert table to structured text format with clear separators."""
        if not self.rows:
            return ""
        
        lines = []
        if self.title:
            lines.append(f"TABLE: {self.title}")
        else:
            lines.append(f"TABLE {self.table_index}")
        lines.append(f"Dimensions: {self.num_rows} rows x {self.num_columns} columns")
        lines.append(f"Has Header: {self.has_header}")
        
        if self.headers:
            lines.append(f"Headers: {', '.join(self.headers)}")
        
        lines.append("")  # Empty line before data
        
        for row in self.rows:
            row_str = f"Row {row.row_index}:"
            for cell in row.cells:
                header_tag = " [HEADER]" if cell.is_header else ""
                row_str += f" | Col {cell.col_index}{header_tag}: {cell.text}"
            row_str += " |"
            lines.append(row_str)
        
        return "\n".join(lines)


def _extract_excel_table(sheet, table_index: int) -> ExtractedTable:
    """Extract a single table from an Excel sheet."""
    table = ExtractedTable(table_index=table_index)
    
    # Find the data range
    rows_data = []
    for row_idx, row in enumerate(sheet.iter_rows(values_only=True)):
        if row is None:
            continue
        # Skip completely empty rows
        if all(cell is None for cell in row):
            continue
        
        cell_objects = []
        for col_idx, value in enumerate(row):
            cell = TableCell(
                row_index=row_idx,
                col_index=col_idx,
                text=str(value) if value is not None else ""
            )
            cell_objects.append(cell)
        
        rows_data.append((row_idx, cell_objects))
    
    if not rows_data:
        return table
    
    # Determine if first row is header
    table.num_rows = len(rows_data)
    table.num_columns = len(rows_data[0][1]) if rows_data else 0
    
    # Check if first row contains headers (non-numeric, shorter text)
    first_row = rows_data[0][1]
    potential_headers = []
    for cell in first_row:
        text = cell.text.strip()
        # Heuristic: if cell is short and contains alphabetic text, likely a header
        is_potential_header = bool(text) and len(text) < 30 and any(c.isalpha() for c in text)
        potential_headers.append(is_potential_header)
    
    # If most cells in first row look like headers, treat as header row
    header_ratio = sum(potential_headers) / len(potential_headers) if potential_headers else 0
    table.has_header = header_ratio > 0.5
    
    # Extract headers if present
    if table.has_header and rows_data:
        table.headers = [cell.text.strip() for cell in rows_data[0][1]]
        rows_data = rows_data[1:]  # Remove header row from data rows
    
    # Build row objects
    for row_idx, cells in rows_data:
        row = TableRow(row_index=row_idx, cells=cells)
        table.rows.append(row)
    
    return table


def extract_excel_tables(excel_bytes: bytes) -> list[ExtractedTable]:
    """
    Extract all tables from an Excel workbook, preserving structure.
    
    Returns a list of ExtractedTable objects, each containing:
    - Full row/column structure
    - Header identification
    - Cell-level metadata
    - Methods to convert to markdown, JSON, or structured text
    """
    workbook = load_workbook(io.BytesIO(excel_bytes), data_only=True, read_only=True)
    tables: list[ExtractedTable] = []
    
    for sheet_idx, sheet in enumerate(workbook.worksheets):
        # Extract each contiguous table in the sheet
        tables_in_sheet = _extract_excel_tables_from_sheet(sheet, sheet_idx)
        tables.extend(tables_in_sheet)
    
    return tables


def _extract_excel_tables_from_sheet(sheet, sheet_index: int) -> list[ExtractedTable]:
    """Extract all tables from a single Excel sheet."""
    tables: list[ExtractedTable] = []
    
    # For now, treat entire sheet as one table
    # Could be enhanced to detect multiple tables in one sheet
    table = _extract_excel_table(sheet, table_index=sheet_index)
    if table.rows:
        tables.append(table)
    
    return tables


def _extract_docx_table(document, table_index: int) -> ExtractedTable:
    """Extract a single table from a Word document."""
    table = ExtractedTable(table_index=table_index)
    
    docx_table = document.tables[table_index]
    
    table.num_rows = len(docx_table.rows)
    table.num_columns = len(docx_table.columns)
    
    rows_data = []
    for row_idx, row in enumerate(docx_table.rows):
        cell_objects = []
        for col_idx, cell in enumerate(row.cells):
            # Get cell text
            cell_text = "\n".join(p.text.strip() for p in cell.paragraphs if p.text.strip())
            
            # Check if this looks like a header cell
            # (simplified - could use more sophisticated detection)
            is_header = row_idx == 0
            
            cell = TableCell(
                row_index=row_idx,
                col_index=col_idx,
                text=cell_text,
                is_header=is_header
            )
            cell_objects.append(cell)
        
        rows_data.append((row_idx, cell_objects))
    
    if not rows_data:
        return table
    
    # Extract headers from first row
    if rows_data:
        table.headers = [cell.text.strip() for cell in rows_data[0][1]]
        table.has_header = True
    
    # Build row objects
    for row_idx, cells in rows_data:
        row = TableRow(row_index=row_idx, cells=cells)
        table.rows.append(row)
    
    return table


def extract_docx_tables(docx_bytes: bytes) -> list[ExtractedTable]:
    """
    Extract all tables from a Word document, preserving structure.
    
    Returns a list of ExtractedTable objects with:
    - Full table structure maintained
    - Header identification
    - Row/column relationships preserved
    """
    document = Document(io.BytesIO(docx_bytes))
    tables: list[ExtractedTable] = []
    
    for table_idx, table in enumerate(document.tables):
        extracted = _extract_docx_table(document, table_idx)
        if extracted.rows:
            extracted.title = f"Table {table_idx + 1}"
            tables.append(extracted)
    
    return tables


def format_tables_for_llm(tables: list[ExtractedTable], format: str = "structured") -> str:
    """
    Format extracted tables for LLM consumption.
    
    Args:
        tables: List of ExtractedTable objects
        format: Output format - "structured", "markdown", or "json"
    
    Returns:
        Formatted string suitable for LLM input
    """
    if not tables:
        return "(no tables found)"
    
    sections = []
    
    for table in tables:
        if format == "markdown":
            sections.append(table.to_markdown())
        elif format == "json":
            import json
            sections.append(json.dumps(table.to_json(), indent=2, ensure_ascii=False))
        else:  # structured (default)
            sections.append(table.to_structured_text())
    
    return "\n\n".join(sections)


def format_tables_for_llm_with_context(tables: list[ExtractedTable], 
                                       context: str = "",
                                       format: str = "structured") -> str:
    """
    Format tables with optional contextual preamble for LLM.
    
    Args:
        tables: List of ExtractedTable objects
        context: Optional context description (e.g., "Minutes of Meeting")
        format: Output format - "markdown", "json", or "structured"
    
    Returns:
        Formatted string with context and tables
    """
    sections = []
    
    if context:
        sections.append(f"DOCUMENT CONTEXT: {context}")
        sections.append("")
    
    if not tables:
        sections.append("(no tables found)")
        return "\n".join(sections)
    
    sections.append(f"EXTRACTED TABLES ({len(tables)} found):")
    sections.append("")
    
    for i, table in enumerate(tables, 1):
        if format == "markdown":
            sections.append(table.to_markdown())
        elif format == "json":
            import json
            sections.append(f"Table {i}:")
            sections.append(json.dumps(table.to_json(), indent=2, ensure_ascii=False))
        else:  # structured
            sections.append(table.to_structured_text())
        
        # Add separator between tables
        if i < len(tables):
            sections.append("")
            sections.append("=" * 80)
            sections.append("")
    
    return "\n".join(sections)