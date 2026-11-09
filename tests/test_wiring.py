"""How the app builds its LLM dependencies from settings. Offline.

Settings are built with _env_file=None and LLM variables removed from the
environment, so nothing here reads the local .env or a real key.
"""

from collections.abc import Iterator
from typing import Any

import pytest
from pydantic import ValidationError

from specsmith import main
from specsmith.config import Settings
from specsmith.domain.errors import LLMNotConfiguredError
from specsmith.llm.anthropic import AnthropicProvider
from specsmith.llm.fake import FakeProvider
from specsmith.llm.retry import RetryingProvider

LLM_ENV = (
    "ANTHROPIC_API_KEY",
    "LLM_MODEL",
    "LLM_TIMEOUT_SECONDS",
    "PROMPT_VERSION",
    "MAX_EXTRACTION_RETRIES",
    "MAX_EXTRACTIONS_PER_DAY",
    "LLM_MAX_RETRIES",
    "LLM_BACKOFF_BASE_SECONDS",
    "LLM_BACKOFF_MAX_SECONDS",
)


def settings(**values: Any) -> Settings:
    return Settings(  # type: ignore[call-arg]
        _env_file=None, database_url="postgresql+psycopg://u:p@127.0.0.1:1/x", **values
    )


@pytest.fixture(autouse=True)
def isolated(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    for name in LLM_ENV:
        monkeypatch.delenv(name, raising=False)
    main.get_provider.cache_clear()
    yield
    main.get_provider.cache_clear()


def use(monkeypatch: pytest.MonkeyPatch, configured: Settings) -> None:
    monkeypatch.setattr(main, "get_settings", lambda: configured)


def test_llm_settings_have_the_documented_defaults() -> None:
    s = settings()

    assert s.anthropic_api_key is None
    assert (s.llm_model, s.llm_timeout_seconds) == ("claude-sonnet-5-5", 30)
    assert (s.prompt_version, s.max_extraction_retries) == ("v3", 2)


@pytest.mark.parametrize(
    "values",
    [
        {"max_extraction_retries": -1},
        {"llm_timeout_seconds": 0},
        {"llm_max_retries": -1},
        {"llm_backoff_base_seconds": 0},
        {"llm_backoff_base_seconds": 10, "llm_backoff_max_seconds": 5},
    ],
)
def test_out_of_range_llm_settings_are_rejected(values: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        settings(**values)


def test_api_key_is_kept_out_of_reprs() -> None:
    s = settings(anthropic_api_key="sk-test-not-real")

    assert "sk-test-not-real" not in repr(s)
    assert "sk-test-not-real" not in str(s.model_dump())


def test_provider_without_a_key_is_not_configured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    use(monkeypatch, settings())

    with pytest.raises(LLMNotConfiguredError):
        main.get_provider()


def test_provider_is_anthropic_built_from_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    use(
        monkeypatch,
        settings(
            anthropic_api_key="sk-test-not-real",
            llm_model="claude-opus-5-5",
            llm_timeout_seconds=12,
        ),
    )

    provider = main.get_provider()

    assert isinstance(provider, RetryingProvider)
    inner = provider._inner
    assert isinstance(inner, AnthropicProvider)
    assert provider.model == inner.model == "claude-opus-5-5"
    assert inner._client.timeout == 12
    assert inner._client.max_retries == 0


def test_provider_backoff_comes_from_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    use(
        monkeypatch,
        settings(
            anthropic_api_key="sk-test-not-real",
            llm_max_retries=4,
            llm_backoff_base_seconds=0.5,
            llm_backoff_max_seconds=3,
        ),
    )

    provider = main.get_provider()

    assert isinstance(provider, RetryingProvider)
    assert (provider._max_retries, provider._base_delay, provider._max_delay) == (
        4,
        0.5,
        3,
    )


def test_llm_backoff_defaults() -> None:
    s = settings()

    assert (
        s.llm_max_retries,
        s.llm_backoff_base_seconds,
        s.llm_backoff_max_seconds,
    ) == (2, 1, 8)


def test_provider_is_built_once_per_process(monkeypatch: pytest.MonkeyPatch) -> None:
    use(monkeypatch, settings(anthropic_api_key="sk-test-not-real"))

    assert main.get_provider() is main.get_provider()


def test_extractor_uses_prompt_version_and_retries_from_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    use(monkeypatch, settings(max_extraction_retries=5))
    provider = FakeProvider([])

    extractor = main.get_extractor(provider)

    assert extractor._provider is provider
    assert extractor._prompt.version == "v3"
    assert extractor._max_retries == 5


def test_daily_extraction_limit_defaults_to_20_and_comes_from_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert settings().max_extractions_per_day == 20

    use(monkeypatch, settings(max_extractions_per_day=7))

    assert main.get_daily_extraction_limit() == 7


def test_daily_extraction_limit_must_be_positive() -> None:
    with pytest.raises(ValidationError):
        settings(max_extractions_per_day=0)
