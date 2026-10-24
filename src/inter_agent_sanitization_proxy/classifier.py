from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Callable, Protocol
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .models import (
    EvidenceReference,
    Finding,
    Severity,
    ThreatCategory,
)
from .normalizer import NormalizedMessage


@dataclass(frozen=True, slots=True)
class Prompt:
    category: ThreatCategory
    detector_id: str
    version: str
    text: str

    def __post_init__(self) -> None:
        if not self.detector_id or not self.version or not self.text:
            raise ValueError("prompt fields must be non-empty")


class Provider(Protocol):
    def complete(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        timeout: float,
    ) -> str:
        ...


class ProviderError(RuntimeError):
    pass


class TransientProviderError(ProviderError):
    pass


class PermanentProviderError(ProviderError):
    pass


class ClassifierTimeoutError(RuntimeError):
    pass


class InvalidModelOutput(RuntimeError):
    pass


class HTTPChatProvider:
    provider_name = "openai-compatible-http"

    def __init__(
        self,
        *,
        endpoint: str,
        model: str,
        api_key: str | None = None,
        temperature: float = 0.0,
    ) -> None:
        if not endpoint.strip():
            raise ValueError("endpoint must be non-empty")
        if not model.strip():
            raise ValueError("model must be non-empty")
        self.endpoint = endpoint
        self.model = model
        self.api_key = api_key
        self.temperature = temperature

    def complete(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        timeout: float,
    ) -> str:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": self.temperature,
        }
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        request = Request(
            self.endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )

        try:
            with urlopen(request, timeout=timeout) as response:
                raw_body = response.read()
        except HTTPError as exc:
            if exc.code == 429 or 500 <= exc.code <= 599:
                raise TransientProviderError(
                    f"model API returned HTTP {exc.code}"
                ) from exc
            raise PermanentProviderError(
                f"model API returned HTTP {exc.code}"
            ) from exc
        except (URLError, TimeoutError, OSError) as exc:
            raise TransientProviderError("model API request failed") from exc

        try:
            body = json.loads(raw_body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise PermanentProviderError(
                "model API returned an invalid JSON response"
            ) from exc

        try:
            content = body["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise PermanentProviderError(
                "model API response did not contain message content"
            ) from exc

        if not isinstance(content, str):
            raise PermanentProviderError(
                "model API message content was not text"
            )

        return content


class LLMClassifier:
    def __init__(
        self,
        provider: Provider,
        *,
        timeout_seconds: float = 20.0,
        max_retries: int = 1,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if max_retries not in (0, 1):
            raise ValueError("at most one retry is supported")
        if not callable(getattr(provider, "complete", None)):
            raise TypeError("provider must implement complete()")
        self.provider = provider
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self._clock = clock

    def new_deadline(self) -> float:
        return self._clock() + self.timeout_seconds

    def classify(
        self,
        *,
        normalized: NormalizedMessage,
        prompt: Prompt,
        deadline: float | None = None,
    ) -> Finding | None:
        if deadline is None:
            deadline = self.new_deadline()

        user_prompt = self._build_user_prompt(normalized)
        attempts = 0
        last_error: Exception | None = None

        while True:
            remaining = deadline - self._clock()
            if remaining <= 0:
                raise ClassifierTimeoutError(
                    "inspection deadline exceeded"
                ) from last_error

            attempts += 1
            try:
                raw = self.provider.complete(
                    system_prompt=prompt.text,
                    user_prompt=user_prompt,
                    timeout=remaining,
                )
            except (TimeoutError, TransientProviderError) as exc:
                last_error = exc
            except ProviderError:
                raise
            except Exception as exc:
                raise ProviderError("provider call failed") from exc
            else:
                if self._clock() > deadline:
                    raise ClassifierTimeoutError(
                        "inspection deadline exceeded"
                    ) from last_error
                return parse_classifier_output(raw, prompt=prompt)

            if attempts > self.max_retries:
                raise ProviderError("provider failed after retry") from last_error

    @staticmethod
    def _build_user_prompt(normalized: NormalizedMessage) -> str:
        message = normalized.original
        context = {
            "message_id": message.message_id,
            "sender": message.sender,
            "recipient": message.recipient,
            "metadata": normalized.metadata,
            "content": normalized.content,
            "content_format": normalized.content_format,
        }
        return json.dumps(
            context,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )


def _strip_markdown_fence(text: str) -> str:
    if not text.startswith("```"):
        return text
    lines = text.splitlines()
    if not lines or not lines[0].startswith("```"):
        return text
    lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()


def parse_classifier_output(raw: str, *, prompt: Prompt) -> Finding | None:
    if not isinstance(raw, str):
        raise InvalidModelOutput("classifier output must be text")

    text = _strip_markdown_fence(raw.strip())
    if not text:
        raise InvalidModelOutput("classifier output was empty")

    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise InvalidModelOutput("classifier output was not valid JSON") from exc

    if not isinstance(data, dict) or set(data) != {"finding"}:
        raise InvalidModelOutput(
            "classifier output must contain exactly one finding field"
        )

    payload = data["finding"]
    if payload is None:
        return None
    if not isinstance(payload, dict):
        raise InvalidModelOutput("finding must be an object or null")

    allowed_keys = {"category", "severity", "reason", "evidence"}
    required_keys = {"category", "severity", "reason"}
    if not required_keys <= set(payload) or not set(payload) <= allowed_keys:
        raise InvalidModelOutput("finding fields do not match the response schema")

    category = payload["category"]
    if not isinstance(category, str) or category != prompt.category.value:
        raise InvalidModelOutput(
            "finding category does not match the detector category"
        )

    raw_severity = payload["severity"]
    if not isinstance(raw_severity, str):
        raise InvalidModelOutput("finding severity must be text")
    try:
        severity = Severity(raw_severity.strip().lower())
    except ValueError as exc:
        raise InvalidModelOutput("finding severity is not recognized") from exc

    reason = payload["reason"]
    if not isinstance(reason, str) or not reason.strip():
        raise InvalidModelOutput("finding reason must be non-empty text")
    reason = reason.strip()[:500]

    raw_evidence = payload.get("evidence")
    if raw_evidence is not None and not isinstance(raw_evidence, str):
        raise InvalidModelOutput("finding evidence must be text or null")
    evidence_excerpt = (
        raw_evidence.strip()[:200]
        if isinstance(raw_evidence, str) and raw_evidence.strip()
        else reason[:200]
    )

    return Finding(
        category=prompt.category,
        severity=severity,
        reason=reason,
        detector=prompt.detector_id,
        evidence=EvidenceReference(
            locator="message",
            excerpt=evidence_excerpt,
        ),
    )
