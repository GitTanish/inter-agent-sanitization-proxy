from __future__ import annotations

import time
from collections.abc import Sequence

from .audit import AuditLog, hash_message
from .classifier import LLMClassifier
from .detectors import (
    CategoryDetector,
    PromptLoadError,
    default_detectors,
)
from .models import (
    FailureReason,
    InspectionResult,
    InspectionStatus,
    Message,
    ThreatCategory,
    Verdict,
)
from .normalizer import Normalizer
from .policy import select_primary_finding


def _elapsed_ms(start: float) -> float:
    return round(max(0.0, (time.monotonic() - start) * 1000.0), 3)


def _validate_detectors(detectors: Sequence[CategoryDetector]) -> None:
    if len(detectors) != len(ThreatCategory):
        raise ValueError("detectors must cover both threat categories exactly once")
    categories = {detector.category for detector in detectors}
    if categories != set(ThreatCategory):
        raise ValueError("detectors must cover both threat categories exactly once")


def _audit(
    audit: AuditLog | None,
    message: Message,
    result: InspectionResult,
) -> None:
    if audit is not None:
        audit.record(message, result)


def inspect_message(
    message: Message,
    classifier: LLMClassifier,
    *,
    detectors: Sequence[CategoryDetector] | None = None,
    normalizer: Normalizer | None = None,
    audit: AuditLog | None = None,
) -> InspectionResult:
    if not isinstance(message, Message):
        raise TypeError("message must be a Message")

    start = time.monotonic()
    digest = hash_message(message)
    normalizer_instance = normalizer if normalizer is not None else Normalizer()
    normalized = normalizer_instance.normalize_message(message)

    if detectors is None:
        try:
            detector_list = default_detectors(classifier)
        except PromptLoadError:
            result = InspectionResult(
                inspection_status=InspectionStatus.UNINSPECTED,
                verdict=Verdict.UNKNOWN,
                primary_finding=None,
                message_hash=digest,
                prompt_version={},
                duration_ms=_elapsed_ms(start),
                failure_reason=f"prompts={FailureReason.PROMPT_ERROR.value}",
            )
            _audit(audit, message, result)
            return result
    else:
        detector_list = tuple(detectors)
        _validate_detectors(detector_list)

    deadline = classifier.new_deadline()
    outcomes = [
        detector.inspect(normalized=normalized, deadline=deadline)
        for detector in detector_list
    ]
    prompt_versions = {
        outcome.category.value: outcome.prompt_version for outcome in outcomes
    }

    failures = [outcome for outcome in outcomes if not outcome.completed]
    findings = [
        outcome.finding
        for outcome in outcomes
        if outcome.finding is not None
    ]
    primary = select_primary_finding(findings)

    if failures:
        failure_reason = ";".join(
            f"{outcome.detector}={outcome.failure_reason.value}"
            for outcome in sorted(failures, key=lambda outcome: outcome.detector)
        )
        result = InspectionResult(
            inspection_status=InspectionStatus.UNINSPECTED,
            verdict=Verdict.UNKNOWN,
            primary_finding=primary,
            message_hash=digest,
            prompt_version=prompt_versions,
            duration_ms=_elapsed_ms(start),
            failure_reason=failure_reason,
        )
    elif primary is not None:
        result = InspectionResult(
            inspection_status=InspectionStatus.INSPECTED,
            verdict=Verdict.THREAT_DETECTED,
            primary_finding=primary,
            message_hash=digest,
            prompt_version=prompt_versions,
            duration_ms=_elapsed_ms(start),
        )
    else:
        result = InspectionResult(
            inspection_status=InspectionStatus.INSPECTED,
            verdict=Verdict.NO_THREAT_DETECTED,
            message_hash=digest,
            prompt_version=prompt_versions,
            duration_ms=_elapsed_ms(start),
        )

    _audit(audit, message, result)
    return result
