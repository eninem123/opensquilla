from __future__ import annotations

import pytest

from opensquilla.provider.selector import ModelSelector, ProviderConfig, SelectorConfig


class _Plugin:
    def __init__(self, chain: list[ProviderConfig]) -> None:
        self.chain = chain
        self.failures: list[Exception] = []

    def failover_hook(self, primary_failure: Exception) -> list[ProviderConfig]:
        self.failures.append(primary_failure)
        return list(self.chain)


def test_selector_failure_fallback_uses_plugin_chain_before_static_fallbacks() -> None:
    failure = RuntimeError("primary failed")
    plugin = _Plugin(
        [
            ProviderConfig(
                provider="deepseek",
                model="deepseek-chat",
            )
        ]
    )
    selector = ModelSelector(
        SelectorConfig(
            primary=ProviderConfig(provider="openai", model="gpt-5.4-mini"),
            fallbacks=[ProviderConfig(provider="anthropic", model="claude-sonnet-4.5")],
        ),
        plugin=plugin,
    )

    provider = selector.next_fallback_after_failure(failure)

    assert plugin.failures == [failure]
    assert selector.current_config.provider == "deepseek"
    assert selector.current_config.model == "deepseek-chat"
    assert getattr(provider, "_provider_kind") == "deepseek"


def test_selector_failure_fallback_treats_empty_plugin_chain_as_no_fallback() -> None:
    plugin = _Plugin([])
    selector = ModelSelector(
        SelectorConfig(
            primary=ProviderConfig(provider="openai", model="gpt-5.4-mini"),
            fallbacks=[ProviderConfig(provider="anthropic", model="claude-sonnet-4.5")],
        ),
        plugin=plugin,
    )

    with pytest.raises(IndexError, match="No fallback chain available"):
        selector.next_fallback_after_failure(RuntimeError("primary failed"))

    assert selector.current_config.provider == "openai"
