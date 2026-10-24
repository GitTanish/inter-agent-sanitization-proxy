from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from inter_agent_sanitization_proxy.models import (
    AuditRecord,
    EvidenceReference,
    Finding,
    InspectionResult,
    InspectionStatus,
    Message,
    Severity,
    ThreatCategory,
    Verdict,
)


def make_finding(
    category: ThreatCategory = ThreatCategory.PROMPT_INJECTION,
    severity: Severity = Severity.HIGH,
) -> Finding:
    return Finding(
        category=category,
        severity=severity,
        reason="test reason",
        detector="prompt_injection",
        evidence=EvidenceReference(locator="message", excerpt="test evidence"),
    )


def make_result(**overrides) -> InspectionResult:
    values = {
        "inspection_status": InspectionStatus.INSPECTED,
        "verdict": Verdict.NO_THREAT_DETECTED,
        "message_hash": "a" * 64,
        "prompt_version": {
            "prompt_injection": "v1",
            "data_exfiltration": "v1",
        },
        "duration_ms": 1.25,
    }
    values.update(overrides)
    return InspectionResult(**values)


def test_message_accepts_text_and_structured_content() -> None:
    text_message = Message(
        sender="planner",
        recipient="executor",
        content="plain text",
        metadata={"trace_id": "t1"},
        message_id="m1",
    )
    object_message = Message(
        sender="planner",
        recipient="executor",
        content={"task": "summarize", "items": [1, 2]},
    )
    array_message = Message(
        sender="planner",
        recipient="executor",
        content=["one", {"two": 2}],
    )

    assert text_message.content == "plain text"
    assert object_message.content == {"task": "summarize", "items": [1, 2]}
    assert array_message.content == ["one", {"two": 2}]


@pytest.mark.parametrize(
    "content",
    [
        42,
        None,
        {"nested": {1, 2}},
        [float("nan")],
        {"bad": object()},
    ],
)
def test_message_rejects_non_json_content(content) -> None:
    with pytest.raises(ValidationError):
        Message(sender="planner", recipient="executor", content=content)


def test_message_rejects_invalid_metadata_and_identifiers() -> None:
    with pytest.raises(ValidationError):
        Message(
            sender="planner",
            recipient="executor",
            content="hello",
            metadata={"bad": {1, 2}},
        )
    with pytest.raises(ValidationError):
        Message(sender="", recipient="executor", content="hello")
    with pytest.raises(ValidationError):
        Message(sender="planner", recipient="executor", content="hello", message_id="")


def test_message_is_frozen() -> None:
    message = Message(sender="planner", recipient="executor", content="hello")

    with pytest.raises(ValidationError):
        message.sender = "changed"


def test_finding_requires_bounded_evidence_and_reason() -> None:
    with pytest.raises(ValidationError):
        EvidenceReference(locator="message", excerpt="x" * 201)
    with pytest.raises(ValidationError):
        Finding(
            category=ThreatCategory.PROMPT_INJECTION,
            severity=Severity.HIGH,
            reason="   ",
            detector="prompt_injection",
            evidence=EvidenceReference(locator="message"),
        )
    with pytest.raises(ValidationError):
        Finding(
            category=ThreatCategory.PROMPT_INJECTION,
            severity=Severity.HIGH,
            reason="reason",
            detector="prompt_injection",
            evidence=None,
        )


def test_clean_inspection_result_is_valid() -> None:
    result = make_result()

    assert result.inspection_status is InspectionStatus.INSPECTED
    assert result.verdict is Verdict.NO_THREAT_DETECTED
    assert result.primary_finding is None
    assert result.failure_reason is None


def test_threat_result_requires_a_primary_finding() -> None:
    with pytest.raises(ValidationError):
        make_result(verdict=Verdict.THREAT_DETECTED)

    finding = make_finding()
    result = make_result(
        verdict=Verdict.THREAT_DETECTED,
        primary_finding=finding,
    )
    assert result.primary_finding == finding


def test_no_threat_result_cannot_include_a_finding() -> None:
    with pytest.raises(ValidationError):
        make_result(primary_finding=make_finding())


def test_uninspected_result_requires_unknown_verdict_and_reason() -> None:
    result = make_result(
        inspection_status=InspectionStatus.UNINSPECTED,
        verdict=Verdict.UNKNOWN,
        failure_reason="prompt_injection=timeout",
    )
    assert result.failure_reason == "prompt_injection=timeout"

    with pytest.raises(ValidationError):
        make_result(
            inspection_status=InspectionStatus.UNINSPECTED,
            verdict=Verdict.UNKNOWN,
        )
    with pytest.raises(ValidationError):
        make_result(
            inspection_status=InspectionStatus.UNINSPECTED,
            verdict=Verdict.NO_THREAT_DETECTED,
            failure_reason="provider_error",
        )
    with pytest.raises(ValidationError):
        make_result(
            inspection_status=InspectionStatus.UNINSPECTED,
            verdict=Verdict.UNKNOWN,
            failure_reason="  ",
        )


def test_inspected_result_rejects_unknown_or_failure() -> None:
    with pytest.raises(ValidationError):
        make_result(verdict=Verdict.UNKNOWN, failure_reason="timeout")
    with pytest.raises(ValidationError):
        make_result(failure_reason="timeout")


def test_inspection_result_requires_sha256_hash() -> None:
    with pytest.raises(ValidationError):
        make_result(message_hash="not-a-hash")


def test_uninspected_result_may_carry_partial_finding() -> None:
    result = make_result(
        inspection_status=InspectionStatus.UNINSPECTED,
        verdict=Verdict.UNKNOWN,
        primary_finding=make_finding(),
        failure_reason="data_exfiltration=timeout",
    )

    assert result.primary_finding is not None


def test_audit_record_validation() -> None:
    record = AuditRecord(
        timestamp=datetime(2026, 1, 2, tzinfo=timezone.utc),
        message_id="m1",
        sender="planner",
        recipient="executor",
        message_hash="b" * 64,
        inspection_status=InspectionStatus.INSPECTED,
        verdict=Verdict.NO_THREAT_DETECTED,
        duration_ms=3.0,
        prompt_version={"prompt_injection": "v1"},
        metadata_keys=["trace_id"],
    )
    assert record.metadata_keys == ["trace_id"]

    with pytest.raises(ValidationError):
        AuditRecord(
            timestamp=datetime(2026, 1, 2),
            sender="planner",
            recipient="executor",
            message_hash="b" * 64,
            inspection_status=InspectionStatus.INSPECTED,
            verdict=Verdict.NO_THREAT_DETECTED,
            duration_ms=3.0,
        )
    with pytest.raises(ValidationError):
        AuditRecord(
            timestamp=datetime(2026, 1, 2, tzinfo=timezone.utc),
            sender="planner",
            recipient="executor",
            message_hash="b" * 64,
            inspection_status=InspectionStatus.INSPECTED,
            verdict=Verdict.THREAT_DETECTED,
            primary_category=ThreatCategory.PROMPT_INJECTION,
            duration_ms=3.0,
        )
