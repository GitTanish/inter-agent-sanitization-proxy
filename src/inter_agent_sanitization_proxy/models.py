from __future__ import annotations

import json
from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class ThreatCategory(str, Enum):
    PROMPT_INJECTION = "prompt_injection"
    DATA_EXFILTRATION = "data_exfiltration"


class Severity(str, Enum):
    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class InspectionStatus(str, Enum):
    INSPECTED = "inspected"
    UNINSPECTED = "uninspected"


class Verdict(str, Enum):
    THREAT_DETECTED = "threat_detected"
    NO_THREAT_DETECTED = "no_threat_detected"
    UNKNOWN = "unknown"


class FailureReason(str, Enum):
    TIMEOUT = "timeout"
    PROVIDER_ERROR = "provider_error"
    INVALID_OUTPUT = "invalid_output"
    PROMPT_ERROR = "prompt_error"
    INTERNAL_ERROR = "internal_error"


MessageContent = str | dict[str, Any] | list[Any]


def _ensure_json_serializable(value: Any, field_name: str) -> Any:
    try:
        json.dumps(value, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be JSON-serializable") from exc
    return value


class Message(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    sender: str = Field(min_length=1)
    recipient: str = Field(min_length=1)
    content: MessageContent
    metadata: dict[str, Any] = Field(default_factory=dict)
    message_id: str | None = Field(default=None, min_length=1)

    @field_validator("content")
    @classmethod
    def _validate_content(cls, value: MessageContent) -> MessageContent:
        return _ensure_json_serializable(value, "content")

    @field_validator("metadata")
    @classmethod
    def _validate_metadata(cls, value: dict[str, Any]) -> dict[str, Any]:
        return _ensure_json_serializable(value, "metadata")


class EvidenceReference(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    locator: str = Field(min_length=1, max_length=200)
    excerpt: str | None = Field(default=None, max_length=200)


class Finding(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    category: ThreatCategory
    severity: Severity
    reason: str = Field(min_length=1, max_length=1000)
    detector: str = Field(min_length=1, max_length=100)
    evidence: EvidenceReference

    @field_validator("reason", "detector")
    @classmethod
    def _strip_non_empty(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("value must not be blank")
        return stripped


class InspectionResult(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    inspection_status: InspectionStatus
    verdict: Verdict
    primary_finding: Finding | None = None
    message_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    prompt_version: dict[str, str] = Field(default_factory=dict)
    duration_ms: float = Field(ge=0)
    failure_reason: str | None = None

    @field_validator("prompt_version")
    @classmethod
    def _validate_prompt_versions(cls, value: dict[str, str]) -> dict[str, str]:
        for category, version in value.items():
            if not category or not version:
                raise ValueError("prompt version keys and values must be non-empty")
        return value

    @field_validator("failure_reason")
    @classmethod
    def _validate_failure_reason(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("failure_reason must not be blank")
        return value

    @model_validator(mode="after")
    def _validate_outcome(self) -> InspectionResult:
        if self.inspection_status is InspectionStatus.INSPECTED:
            if self.verdict is Verdict.UNKNOWN:
                raise ValueError("inspected results cannot have an unknown verdict")
            if self.failure_reason is not None:
                raise ValueError("inspected results cannot have a failure reason")
            if (
                self.verdict is Verdict.THREAT_DETECTED
                and self.primary_finding is None
            ):
                raise ValueError("threat_detected requires a primary finding")
            if (
                self.verdict is Verdict.NO_THREAT_DETECTED
                and self.primary_finding is not None
            ):
                raise ValueError("no_threat_detected cannot include a finding")
        else:
            if self.verdict is not Verdict.UNKNOWN:
                raise ValueError("uninspected results must have an unknown verdict")
            if self.failure_reason is None:
                raise ValueError("uninspected results require a failure reason")
        return self


class AuditRecord(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    timestamp: datetime
    message_id: str | None = None
    sender: str
    recipient: str
    message_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    inspection_status: InspectionStatus
    verdict: Verdict
    primary_category: ThreatCategory | None = None
    primary_severity: Severity | None = None
    primary_detector: str | None = None
    failure_reason: str | None = None
    duration_ms: float = Field(ge=0)
    prompt_version: dict[str, str] = Field(default_factory=dict)
    metadata_keys: list[str] = Field(default_factory=list)

    @field_validator("timestamp")
    @classmethod
    def _require_timezone(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("timestamp must be timezone-aware")
        return value

    @model_validator(mode="after")
    def _validate_primary_pair(self) -> AuditRecord:
        if (self.primary_category is None) != (self.primary_severity is None):
            raise ValueError(
                "primary_category and primary_severity must be set together"
            )
        return self
