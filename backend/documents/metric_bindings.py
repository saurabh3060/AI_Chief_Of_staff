"""Reviewed business metrics that can be derived safely from approved fact series.

These mappings do not approve the original free-text formula. They select an already validated
base metric and a deterministic operation whose provenance remains the underlying PDF facts.
"""

RUNTIME_TEMPLATES = [
    {"source": "CC12.pdf", "page": 3, "metric": "Quarterly sourcing",
     "base_metric": "Monthly total sourcing", "operation": "quarterly_sum"},
    {"source": "CC12.pdf", "page": 3, "metric": "Quarterly channel sourcing",
     "base_metric": "Monthly channel sourcing", "operation": "quarterly_sum"},
    {"source": "CC12.pdf", "page": 3, "metric": "Quarterly product sourcing",
     "base_metric": "Monthly product sourcing", "operation": "quarterly_sum"},
    {"source": "CC12.pdf", "page": 4, "metric": "Quarterly net addition",
     "base_metric": "Monthly net addition", "operation": "quarterly_sum"},
    {"source": "CC12.pdf", "page": 6, "metric": "Quarterly spend",
     "base_metric": "Monthly spend", "operation": "quarterly_sum"},
    {"source": "CC12.pdf", "page": 7, "metric": "Quarterly disbursal",
     "base_metric": "Monthly disbursal", "operation": "quarterly_sum"},
    {"source": "CC12.pdf", "page": 7, "metric": "Quarterly cases",
     "base_metric": "Monthly cases", "operation": "quarterly_sum"},
]
