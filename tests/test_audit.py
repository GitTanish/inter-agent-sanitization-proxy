from datetime import datetime, timezone
import json

import pytest

from inter_agent_sanitization_proxy.audit import (
    AuditLog,
    canonical_message,
    hash_message,
)
from inter_agent_sanitization_proxy.models import (
    EvidenceReference,
    Finding,
    InspectionResult,
    InspectionStatus,
    Message,
    Severity,
    ThreatCategory,
    Verdict,
)


def make_message(content=None) -> Message:
    return Message(
        sender="planner",
        recipient="executor",
        content=content or "Please summarize the deployment notes.",
        metadata={"trace_id": "sensitive-trace-value", "source": "test"},
        message_id="m1",
    )


def make_finding() -> Finding:
    return Finding(
        category=ThreatCategory.PROMPT_INJECTION,
        severity=Severity.HIGH,
        reason="sensitive reason that must not be logged",
        detector="prompt_injection",
        evidence=EvidenceReference(
            locator="message",
            excerpt="sensitive evidence that must not be logged",
        ),
    )


def make_result(message: Message, **overrides) -> InspectionResult:
    values = {
        "inspection_status": InspectionStatus.INSPECTED,
        "verdict": Verdict.NO_THREAT_DETECTED,
        "message_hash": hash_message(message),
        "prompt_version": {
            "prompt_injection": "v1",
            "data_exfiltration": "v1",
        },
        "duration_ms": 12.5,
    }
    values.update(overrides)
    return InspectionResult(**values)


def test_message_hash_is_stable_sha256() -> None:
    message = make_message()
    same_message = Message(
        sender="planner",
        recipient="executor",
        content={
            "source": "test",
            "trace_id": "sensitive-trace-value",
        },
        message_id="m1",
    )
    reordered_message = Message(
        sender="planner",
        recipient="executor",
        content="Please summarize the deployment notes.",
        metadata={"source": "test", "trace_id": "sensitive-trace-value"},
        message_id="m1",
    )

    digest = hash_message(message)

    assert len(digest) == 64
    assert int(digest, 16) >= 0
    assert digest == hash_message(reordered_message)
    assert digest != hash_message(same_message)
    assert canonical_message(message).startswith("{")


def test_records_clean_outcome_without_raw_content() -> None:
    message = make_message()
    audit = AuditLog(
        clock=lambda: datetime(2026, 1, 2, 3, 4, 5, tzinfo=timezone.utc)
    )
    result = make_result(message)

    record = audit.record(message, result)

    assert record.timestamp == datetime(
        2026, 1, 2, 3, 4, 5, tzinfo=timezone.utc
    )
    assert record.message_id == "m1"
    assert record.sender == "planner"
    assert record.recipient == "executor"
    assert record.message_hash == result.message_hash
    assert record.inspection_status is InspectionStatus.INSPECTED
    assert record.verdict is Verdict.NO_THREAT_DETECTED
    assert record.primary_category is None
    assert record.primary_severity is None
    assert record.primary_detector is None
    assert record.failure_reason is None
    assert record.duration_ms == 12.5
    assert record.prompt_version == {
        "prompt_injection": "v1",
        "data_exfiltration": "v1",
    }
    assert record.metadata_keys == ["source", "trace_id"]

    payload = audit.to_jsonl()
    assert "Please summarize the deployment notes." not in payload
    assert "sensitive-trace-value" not in payload
    assert "trace_id" in payload


def test_records_flagged_outcome_without_reason_or_evidence() -> None:
    message = make_message()
    finding = make_finding()
    audit = AuditLog()
    result = make_result(
        message,
        verdict=Verdict.THREAT_DETECTED,
        primary_finding=finding,
    )

    record = audit.record(message, result)
    payload = audit.to_jsonl()

    assert record.primary_category is ThreatCategory.PROMPT_INJECTION
    assert record.primary_severity is Severity.HIGH
    assert record.primary_detector == "prompt_injection"
    assert "sensitive reason" not in payload
    assert "sensitive evidence" not in payload
    assert "Please summarize" not in payload


def test_records_uninspected_outcome() -> None:
    message = make_message()
    audit = AuditLog()
    result = make_result(
        message,
        inspection_status=InspectionStatus.UNINSPECTED,
        verdict=Verdict.UNKNOWN,
        failure_reason="prompt_injection=timeout",
    )

    record = audit.record(message, result)

    assert record.inspection_status is InspectionStatus.UNINSPECTED
    assert record.verdict is Verdict.UNKNOWN
    assert record.failure_reason == "prompt_injection=timeout"
    assert record.primary_category is None


def test_rejects_result_with_wrong_hash() -> None:
    message = make_message()
    result = make_result(message, message_hash="a" * 64)
    audit = AuditLog()

    with pytest.raises(ValueError):
        audit.record(message, result)


def test_records_returns_snapshot_and_clear_works() -> None:
    message = make_message()
    audit = AuditLog()

    audit.record(message, make_result(message))
    snapshot = audit.records
    audit.record(message, make_result(message))

    assert len(snapshot) == 1
    assert len(audit) == 2

    audit.clear()

    assert len(audit) == 0
    assert audit.to_jsonl() == ""


def test_naive_clock_is_normalized_to_utc() -> None:
    message = make_message()
    audit = AuditLog(clock=lambda: datetime(2026, 5, 6, 7, 8, 9))

    record = audit.record(message, make_result(message))

    assert record.timestamp.tzinfo is timezone.utc
    assert record.timestamp.isoformat() == "2026-05-06T07:08:09+00:00"


def test_write_jsonl_round_trips(tmp_path) -> None:
    message = make_message()
    audit = AuditLog()
    audit.record(message, make_result(message))

    path = tmp_path / "audit.jsonl"
    audit.write_jsonl(path)

    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    data = json.loads(lines[0])
    assert data["sender"] == "planner"
    assert data["inspection_status"] == "inspected"
    assert data["verdict"] == "no_threat_detected"
    assert data["metadata_keys"] == ["source", "trace_id"]
    assert "content" not in data
    assert "reason" not in data
