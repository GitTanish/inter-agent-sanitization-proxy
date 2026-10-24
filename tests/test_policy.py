from inter_agent_sanitization_proxy.models import (
    EvidenceReference,
    Finding,
    Severity,
    ThreatCategory,
)
from inter_agent_sanitization_proxy.policy import (
    select_primary_finding,
    severity_rank,
)


def make_finding(
    category: ThreatCategory,
    severity: Severity,
    detector: str | None = None,
    reason: str = "reason",
) -> Finding:
    return Finding(
        category=category,
        severity=severity,
        reason=reason,
        detector=detector or category.value,
        evidence=EvidenceReference(locator="message", excerpt="evidence"),
    )


def test_severity_ordering() -> None:
    assert (
        severity_rank(Severity.INFO)
        < severity_rank(Severity.LOW)
        < severity_rank(Severity.MEDIUM)
        < severity_rank(Severity.HIGH)
    )


def test_no_findings_returns_none() -> None:
    assert select_primary_finding([]) is None


def test_highest_severity_wins() -> None:
    low = make_finding(ThreatCategory.PROMPT_INJECTION, Severity.LOW)
    high = make_finding(ThreatCategory.DATA_EXFILTRATION, Severity.HIGH)

    assert select_primary_finding([low, high]) == high


def test_category_breaks_severity_tie_in_favor_of_exfiltration() -> None:
    injection = make_finding(ThreatCategory.PROMPT_INJECTION, Severity.HIGH)
    exfiltration = make_finding(ThreatCategory.DATA_EXFILTRATION, Severity.HIGH)

    assert select_primary_finding([injection, exfiltration]) == exfiltration
    assert select_primary_finding([exfiltration, injection]) == exfiltration


def test_selection_is_deterministic_for_equal_findings() -> None:
    first = make_finding(
        ThreatCategory.PROMPT_INJECTION,
        Severity.MEDIUM,
        detector="alpha",
        reason="same",
    )
    second = make_finding(
        ThreatCategory.PROMPT_INJECTION,
        Severity.MEDIUM,
        detector="beta",
        reason="same",
    )

    assert select_primary_finding([first, second]) == first
    assert select_primary_finding([second, first]) == first


def test_findings_may_be_provided_as_a_generator() -> None:
    finding = make_finding(ThreatCategory.PROMPT_INJECTION, Severity.HIGH)

    assert select_primary_finding(item for item in [finding]) == finding
