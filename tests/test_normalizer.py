import pytest

from inter_agent_sanitization_proxy.models import Message
from inter_agent_sanitization_proxy.normalizer import Normalizer


def test_nfkc_normalizes_fullwidth_characters() -> None:
    assert Normalizer().normalize_text("Ｉｇｎｏｒｅ　ｍｅ") == "Ignore me"


def test_removes_zero_width_and_bidi_controls() -> None:
    assert (
        Normalizer().normalize_text("ig\u200bnore\u202eevil\u200f\u061c")
        == "ignoreevil"
    )


def test_strips_control_characters_but_keeps_newlines() -> None:
    assert Normalizer().normalize_text("a\x00b\x07c\nd\x7fe") == "abc\nde"


def test_collapses_whitespace_and_excess_blank_lines() -> None:
    text = "  hello   world \n\n\n\nsecond\t\tline  "
    assert Normalizer().normalize_text(text) == "hello world\n\nsecond line"


def test_normalize_text_rejects_non_string_input() -> None:
    with pytest.raises(TypeError):
        Normalizer().normalize_text(b"bytes")  # type: ignore[arg-type]


def test_normalizes_text_message_and_preserves_original() -> None:
    message = Message(
        sender="planner",
        recipient="executor",
        content="  Ｉｇｎｏｒｅ \u200b this   message.  ",
        metadata={"trace": "  t1  "},
        message_id="m1",
    )
    normalized = Normalizer().normalize_message(message)

    assert normalized.content == "Ignore this message."
    assert normalized.metadata == {"trace": "t1"}
    assert normalized.content_format == "text"
    assert normalized.original is message
    assert message.content == "  Ｉｇｎｏｒｅ \u200b this   message.  "
    assert message.metadata == {"trace": "  t1  "}


def test_normalizes_structured_message_recursively() -> None:
    content = {
        "title": "  Ｒｅｐｏｒｔ  ",
        "items": ["Ａ  Ｂ", {"note": "line\u200bone"}],
        "count": 2,
        "enabled": True,
        "missing": None,
    }
    message = Message(
        sender="planner",
        recipient="executor",
        content=content,
        metadata={"source": "  tool  "},
    )
    normalized = Normalizer().normalize_message(message)

    assert normalized.content == {
        "title": "Report",
        "items": ["A B", {"note": "lineone"}],
        "count": 2,
        "enabled": True,
        "missing": None,
    }
    assert normalized.metadata == {"source": "tool"}
    assert normalized.content_format == "structured"
    assert message.content["title"] == "  Ｒｅｐｏｒｔ  "
    assert message.metadata["source"] == "  tool  "


def test_preserves_structured_keys() -> None:
    message = Message(
        sender="planner",
        recipient="executor",
        content={"Keep Key": " value "},
    )
    normalized = Normalizer().normalize_message(message)

    assert normalized.content == {"Keep Key": "value"}


def test_normalizes_structured_array() -> None:
    message = Message(
        sender="planner",
        recipient="executor",
        content=["  first  ", ["  second  "]],
    )
    normalized = Normalizer().normalize_message(message)

    assert normalized.content == ["first", ["second"]]
    assert normalized.content_format == "structured"


def test_normalize_message_rejects_non_message_input() -> None:
    with pytest.raises(TypeError):
        Normalizer().normalize_message("hello")  # type: ignore[arg-type]
