from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .classifier import (
    ClassifierTimeoutError,
    InvalidModelOutput,
    LLMClassifier,
    Prompt,
    ProviderError,
)
from .models import FailureReason, Finding, ThreatCategory
from .normalizer import NormalizedMessage

PROMPT_DIR = Path(__file__).with_name("prompts")


class PromptLoadError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class PromptSpec:
    category: ThreatCategory
    detector_id: str
    version: str
    filename: str


PROMPT_SPECS: tuple[PromptSpec, ...] = (
    PromptSpec(
        category=ThreatCategory.PROMPT_INJECTION,
        detector_id="prompt_injection",
        version="v1",
        filename="prompt_injection_v1.txt",
    ),
    PromptSpec(
        category=ThreatCategory.DATA_EXFILTRATION,
        detector_id="data_exfiltration",
        version="v1",
        filename="data_exfiltration_v1.txt",
    ),
)

DEFAULT_PROMPT_VERSIONS: dict[str, str] = {
    spec.category.value: spec.version for spec in PROMPT_SPECS
}


@dataclass(frozen=True, slots=True)
class DetectorOutcome:
    detector: str
    category: ThreatCategory
    prompt_version: str
    finding: Finding | None = None
    failure_reason: FailureReason | None = None

    def __post_init__(self) -> None:
        if self.failure_reason is not None and self.finding is not None:
            raise ValueError("a failed detector cannot report a finding")

    @property
    def completed(self) -> bool:
        return self.failure_reason is None


def load_prompt(category: ThreatCategory) -> Prompt:
    spec = next(
        (candidate for candidate in PROMPT_SPECS if candidate.category is category),
        None,
    )
    if spec is None:
        raise PromptLoadError(f"no prompt is configured for {category.value}")

    path = PROMPT_DIR / spec.filename
    try:
        text = path.read_text(encoding="utf-8").strip()
    except OSError as exc:
        raise PromptLoadError(
            f"could not read prompt file for {category.value}"
        ) from exc

    if not text:
        raise PromptLoadError(f"prompt file for {category.value} is empty")

    return Prompt(
        category=spec.category,
        detector_id=spec.detector_id,
        version=spec.version,
        text=text,
    )


class CategoryDetector:
    def __init__(self, classifier: LLMClassifier, prompt: Prompt) -> None:
        self.classifier = classifier
        self.prompt = prompt

    @property
    def detector_id(self) -> str:
        return self.prompt.detector_id

    @property
    def category(self) -> ThreatCategory:
        return self.prompt.category

    def inspect(
        self,
        normalized: NormalizedMessage,
        deadline: float,
    ) -> DetectorOutcome:
        try:
            finding = self.classifier.classify(
                normalized=normalized,
                prompt=self.prompt,
                deadline=deadline,
            )
        except ClassifierTimeoutError:
            return self._failure(FailureReason.TIMEOUT)
        except InvalidModelOutput:
            return self._failure(FailureReason.INVALID_OUTPUT)
        except ProviderError:
            return self._failure(FailureReason.PROVIDER_ERROR)
        except Exception:
            return self._failure(FailureReason.INTERNAL_ERROR)

        if finding is not None:
            if (
                not isinstance(finding, Finding)
                or finding.category is not self.prompt.category
            ):
                return self._failure(FailureReason.INVALID_OUTPUT)

        return DetectorOutcome(
            detector=self.detector_id,
            category=self.category,
            prompt_version=self.prompt.version,
            finding=finding,
        )

    def _failure(self, reason: FailureReason) -> DetectorOutcome:
        return DetectorOutcome(
            detector=self.detector_id,
            category=self.category,
            prompt_version=self.prompt.version,
            failure_reason=reason,
        )


def default_detectors(
    classifier: LLMClassifier,
) -> tuple[CategoryDetector, ...]:
    return tuple(
        CategoryDetector(classifier=classifier, prompt=load_prompt(spec.category))
        for spec in PROMPT_SPECS
    )
