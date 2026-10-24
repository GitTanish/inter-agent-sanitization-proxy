from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

from inter_agent_sanitization_proxy.audit import AuditLog, hash_message
from inter_agent_sanitization_proxy.classifier import (
    LLMClassifier,
    TransientProviderError,
)
from inter_agent_sanitization_proxy.detectors import (
    CategoryDetector,
    load_prompt,
)
from inter_agent_sanitization_proxy.models import (
    InspectionStatus,
    Message,
    Severity,
    ThreatCategory,
    Verdict,
)
from inter_agent_sanitization_proxy.proxy import inspect_message

CLEAN = json.dumps({"finding": None})


def finding_response(
    category: str,
    severity: str = "high",
    reason: str = "test finding",
    evidence: str = "bounded evidence",
) -> str:
    return json.dumps(
        {
            "finding": {
                "category": category,
                "severity": severity,
                "reason": reason,
                "evidence": evidence,
            }
        }
    )


def make_message() -> Message:
    return Message(
        sender="planner",
        recipient="executor",
        content="Please summarize the deployment notes.",
        metadata={"trace_id": "trace-1"},
        message_id="m1",
    )


class ScenarioProvider:
    provider_name = "fake"
    model = "fake-v1"

    def __init__(self, responder):
        self.responder = responder
        self.calls = []
        self.attempts = {"prompt_injection": 0, "data_exfiltration": 0}

    def complete(self, *, system_prompt, user_prompt, timeout):
        category = (
            "data_exfiltration"
            if "data_exfiltration" in system_prompt
            else "prompt_injection"
        )
        self.attempts[category] += 1
        self.calls.append(
            {
                "category": category,
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
                "timeout": timeout,
                "attempt": self.attempts[category],
            }
        )
        return self.responder(category, self.attempts[category])


def test_clean_message_is_inspected_and_not_modified() -> None:
    provider = ScenarioProvider(lambda category, attempt: CLEAN)
    classifier = LLMClassifier(provider)
    message = make_message()
    original_dump = message.model_dump()

    result = inspect_message(message, classifier)

    assert result.inspection_status is InspectionStatus.INSPECTED
    assert result.verdict is Verdict.NO_THREAT_DETECTED
    assert result.primary_finding is None
    assert result.failure_reason is None
    assert result.duration_ms >= 0
    assert result.message_hash == hash_message(message)
    assert result.prompt_version == {
        "prompt_injection": "v1",
        "data_exfiltration": "v1",
    }
    assert len(provider.calls) == 2
    assert provider.calls[0]["system_prompt"] == load_prompt(
        ThreatCategory.PROMPT_INJECTION
    ).text
    assert provider.calls[1]["system_prompt"] == load_prompt(
        ThreatCategory.DATA_EXFILTRATION
    ).text
    assert message.model_dump() == original_dump
    assert "content" not in result.model_dump()


def test_clean_and_uninspected_outcomes_are_distinct() -> None:
    clean_provider = ScenarioProvider(lambda category, attempt: CLEAN)
    clean_result = inspect_message(
        make_message(),
        LLMClassifier(clean_provider),
    )

    failing_provider = ScenarioProvider(
        lambda category, attempt: (_ for _ in ()).throw(
            TransientProviderError("offline")
        )
    )
    uninspected_result = inspect_message(
        make_message(),
        LLMClassifier(failing_provider),
    )

    assert clean_result.inspection_status is InspectionStatus.INSPECTED
    assert clean_result.verdict is Verdict.NO_THREAT_DETECTED
    assert uninspected_result.inspection_status is InspectionStatus.UNINSPECTED
    assert uninspected_result.verdict is Verdict.UNKNOWN
    assert uninspected_result.failure_reason is not None


def test_threat_result_includes_bounded_primary_finding() -> None:
    provider = ScenarioProvider(
        lambda category, attempt: (
            finding_response("prompt_injection", "high", "override attempt")
            if category == "prompt_injection"
            else finding_response("data_exfiltration", "medium", "upload attempt")
        )
    )
    classifier = LLMClassifier(provider)

    result = inspect_message(make_message(), classifier)

    assert result.inspection_status is InspectionStatus.INSPECTED
    assert result.verdict is Verdict.THREAT_DETECTED
    assert result.failure_reason is None
    assert result.primary_finding is not None
    assert result.primary_finding.category is ThreatCategory.PROMPT_INJECTION
    assert result.primary_finding.severity is Severity.HIGH
    assert result.primary_finding.reason == "override attempt"
    assert len(result.primary_finding.evidence.excerpt or "") <= 200


def test_primary_finding_uses_deterministic_tie_break() -> None:
    provider = ScenarioProvider(
        lambda category, attempt: finding_response(category, "high")
    )

    result = inspect_message(make_message(), LLMClassifier(provider))

    assert result.verdict is Verdict.THREAT_DETECTED
    assert result.primary_finding is not None
    assert result.primary_finding.category is ThreatCategory.DATA_EXFILTRATION


def test_invalid_output_marks_inspection_incomplete_without_retry() -> None:
    provider = ScenarioProvider(
        lambda category, attempt: (
            "not-json" if category == "prompt_injection" else CLEAN
        )
    )
    classifier = LLMClassifier(provider)

    result = inspect_message(make_message(), classifier)

    assert result.inspection_status is InspectionStatus.UNINSPECTED
    assert result.verdict is Verdict.UNKNOWN
    assert result.primary_finding is None
    assert result.failure_reason == "prompt_injection=invalid_output"
    assert result.prompt_version == {
        "prompt_injection": "v1",
        "data_exfiltration": "v1",
    }
    assert provider.attempts["prompt_injection"] == 1
    assert provider.attempts["data_exfiltration"] == 1


def test_transient_provider_failure_is_retried_once() -> None:
    def responder(category, attempt):
        if category == "prompt_injection" and attempt == 1:
            raise TransientProviderError("temporary")
        return CLEAN

    provider = ScenarioProvider(responder)
    classifier = LLMClassifier(provider)

    result = inspect_message(make_message(), classifier)

    assert result.inspection_status is InspectionStatus.INSPECTED
    assert result.verdict is Verdict.NO_THREAT_DETECTED
    assert result.failure_reason is None
    assert provider.attempts["prompt_injection"] == 2
    assert provider.attempts["data_exfiltration"] == 1
    assert len(provider.calls) == 3


def test_exhausted_retry_marks_inspection_incomplete() -> None:
    def responder(category, attempt):
        if category == "prompt_injection":
            raise TransientProviderError("offline")
        return CLEAN

    provider = ScenarioProvider(responder)
    classifier = LLMClassifier(provider)

    result = inspect_message(make_message(), classifier)

    assert result.inspection_status is InspectionStatus.UNINSPECTED
    assert result.verdict is Verdict.UNKNOWN
    assert result.failure_reason == "prompt_injection=provider_error"
    assert result.primary_finding is None
    assert provider.attempts["prompt_injection"] == 2
    assert provider.attempts["data_exfiltration"] == 1


def test_partial_finding_is_kept_when_other_detector_fails() -> None:
    def responder(category, attempt):
        if category == "prompt_injection":
            return finding_response("prompt_injection", "high", "override attempt")
        raise TransientProviderError("offline")

    provider = ScenarioProvider(responder)
    classifier = LLMClassifier(provider)

    result = inspect_message(make_message(), classifier)

    assert result.inspection_status is InspectionStatus.UNINSPECTED
    assert result.verdict is Verdict.UNKNOWN
    assert result.failure_reason == "data_exfiltration=provider_error"
    assert result.primary_finding is not None
    assert result.primary_finding.category is ThreatCategory.PROMPT_INJECTION


def test_detectors_share_one_overall_deadline() -> None:
    clock = [0.0]

    class DeadlineProvider:
        provider_name = "fake"
        model = "fake-v1"

        def __init__(self):
            self.calls = []

        def complete(self, *, system_prompt, user_prompt, timeout):
            self.calls.append(system_prompt)
            clock[0] = 100.0
            return CLEAN

    provider = DeadlineProvider()
    classifier = LLMClassifier(
        provider,
        timeout_seconds=10.0,
        clock=lambda: clock[0],
    )

    result = inspect_message(make_message(), classifier)

    assert result.inspection_status is InspectionStatus.UNINSPECTED
    assert result.verdict is Verdict.UNKNOWN
    assert result.primary_finding is None
    assert result.failure_reason == (
        "data_exfiltration=timeout;prompt_injection=timeout"
    )
    assert len(provider.calls) == 1


def test_audits_out_of_band_result() -> None:
    audit = AuditLog(clock=lambda: datetime(2026, 1, 2, tzinfo=timezone.utc))
    provider = ScenarioProvider(lambda category, attempt: CLEAN)

    result = inspect_message(
        make_message(),
        LLMClassifier(provider),
        audit=audit,
    )

    assert len(audit) == 1
    record = audit.records[0]
    assert record.message_hash == result.message_hash
    assert record.inspection_status is InspectionStatus.INSPECTED
    assert record.verdict is Verdict.NO_THREAT_DETECTED
    assert "Please summarize the deployment notes." not in audit.to_jsonl()


def test_detectors_must_cover_both_categories_exactly_once() -> None:
    classifier = LLMClassifier(
        ScenarioProvider(lambda category, attempt: CLEAN)
    )
    only_injection = CategoryDetector(
        classifier=classifier,
        prompt=load_prompt(ThreatCategory.PROMPT_INJECTION),
    )

    with pytest.raises(ValueError):
        inspect_message(
            make_message(),
            classifier,
            detectors=[only_injection],
        )


def test_rejects_non_message_input() -> None:
    classifier = LLMClassifier(
        ScenarioProvider(lambda category, attempt: CLEAN)
    )

    with pytest.raises(TypeError):
        inspect_message("hello", classifier)  # type: ignore[arg-type]
