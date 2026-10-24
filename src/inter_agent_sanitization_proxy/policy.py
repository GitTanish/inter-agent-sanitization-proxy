from __future__ import annotations

from collections.abc import Iterable

from .models import Finding, Severity, ThreatCategory

SEVERITY_ORDER: dict[Severity, int] = {
    Severity.INFO: 0,
    Severity.LOW: 1,
    Severity.MEDIUM: 2,
    Severity.HIGH: 3,
}

CATEGORY_PRIORITY: dict[ThreatCategory, int] = {
    ThreatCategory.PROMPT_INJECTION: 0,
    ThreatCategory.DATA_EXFILTRATION: 1,
}


def severity_rank(severity: Severity) -> int:
    return SEVERITY_ORDER[severity]


def _selection_key(finding: Finding) -> tuple[object, ...]:
    return (
        -severity_rank(finding.severity),
        -CATEGORY_PRIORITY[finding.category],
        finding.detector,
        finding.reason,
        finding.evidence.locator,
        finding.evidence.excerpt or "",
    )


def select_primary_finding(findings: Iterable[Finding]) -> Finding | None:
    candidates = tuple(findings)
    if not candidates:
        return None
    return min(candidates, key=_selection_key)
