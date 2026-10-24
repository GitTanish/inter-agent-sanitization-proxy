from __future__ import annotations

import hashlib
import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from .models import AuditRecord, InspectionResult, Message


def canonical_message(message: Message) -> str:
    if not isinstance(message, Message):
        raise TypeError("message must be a Message")
    return json.dumps(
        message.model_dump(mode="json"),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def hash_message(message: Message) -> str:
    canonical = canonical_message(message)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class AuditLog:
    def __init__(
        self,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self._records: list[AuditRecord] = []
        self._lock = threading.RLock()
        self._clock = clock or (lambda: datetime.now(timezone.utc))

    def record(self, message: Message, result: InspectionResult) -> AuditRecord:
        digest = hash_message(message)
        if digest != result.message_hash:
            raise ValueError("result message hash does not match the message")

        now = self._clock()
        if now.tzinfo is None or now.utcoffset() is None:
            now = now.replace(tzinfo=timezone.utc)
        else:
            now = now.astimezone(timezone.utc)

        finding = result.primary_finding
        record = AuditRecord(
            timestamp=now,
            message_id=message.message_id,
            sender=message.sender,
            recipient=message.recipient,
            message_hash=digest,
            inspection_status=result.inspection_status,
            verdict=result.verdict,
            primary_category=finding.category if finding is not None else None,
            primary_severity=finding.severity if finding is not None else None,
            primary_detector=finding.detector if finding is not None else None,
            failure_reason=result.failure_reason,
            duration_ms=result.duration_ms,
            prompt_version=dict(result.prompt_version),
            metadata_keys=sorted(message.metadata),
        )

        with self._lock:
            self._records.append(record)

        return record

    @property
    def records(self) -> tuple[AuditRecord, ...]:
        with self._lock:
            return tuple(self._records)

    def __len__(self) -> int:
        with self._lock:
            return len(self._records)

    def clear(self) -> None:
        with self._lock:
            self._records.clear()

    def to_jsonl(self) -> str:
        return "".join(
            json.dumps(
                record.model_dump(mode="json"),
                ensure_ascii=False,
                sort_keys=True,
            )
            + "\n"
            for record in self.records
        )

    def write_jsonl(self, path: str | Path) -> None:
        Path(path).write_text(self.to_jsonl(), encoding="utf-8")
