import json
from pathlib import Path
from typing import Final

import pytest
import respx

import litellm
from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

_DEFAULT_API_BASE: Final = "https://inference.jetfern.ai/v1"
_CHAT_COMPLETIONS_URL: Final = f"{_DEFAULT_API_BASE}/chat/completions"
_MODEL: Final = "deepseek-v4.1-flash"
_CHAT_COMPLETION: Final = {
    "id": "chatcmpl_jetfern",
    "object": "chat.completion",
    "created": 1_790_000_000,
    "model": _MODEL,
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "Hello from JetFern", "reasoning_content": "Say hi"},
            "finish_reason": "stop",
        }
    ],
    "usage": {"prompt_tokens": 4, "completion_tokens": 3, "total_tokens": 7},
}
_TOOL_CALL_COMPLETION: Final = {
    **_CHAT_COMPLETION,
    "choices": [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "get_weather", "arguments": '{"city": "Paris"}'},
                    }
                ],
            },
            "finish_reason": "tool_calls",
        }
    ],
}
_WEATHER_TOOL: Final = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
    },
}
JETFERN_MODELS: Final = tuple(sorted(name for name in litellm.model_cost if name.startswith("jetfern/")))


def test_jetfern_is_a_registered_provider():
    assert litellm.LlmProviders.JETFERN.value == "jetfern"
    assert "jetfern" in litellm.provider_list
    assert "jetfern" in litellm.constants.openai_compatible_providers
    assert _DEFAULT_API_BASE in litellm.constants.openai_compatible_endpoints


def test_jetfern_provider_resolution(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("JETFERN_API_KEY", "jetfern-test-key")
    monkeypatch.delenv("JETFERN_API_BASE", raising=False)

    model, provider, api_key, api_base = get_llm_provider(
        model=f"jetfern/{_MODEL}",
        custom_llm_provider=None,
        api_base=None,
        api_key=None,
    )

    assert model == _MODEL
    assert provider == "jetfern"
    assert api_key == "jetfern-test-key"
    assert api_base == _DEFAULT_API_BASE


def test_jetfern_provider_keeps_explicit_credentials(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("JETFERN_API_KEY", "jetfern-env-key")
    monkeypatch.setenv("JETFERN_API_BASE", "https://jetfern.env.example/v1")

    _, provider, api_key, api_base = get_llm_provider(
        model=f"jetfern/{_MODEL}",
        custom_llm_provider=None,
        api_base="https://jetfern.internal.example/v1",
        api_key="jetfern-explicit-key",
    )

    assert provider == "jetfern"
    assert api_key == "jetfern-explicit-key"
    assert api_base == "https://jetfern.internal.example/v1"


def test_jetfern_api_base_env_overrides_default(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("JETFERN_API_KEY", "jetfern-env-key")
    monkeypatch.setenv("JETFERN_API_BASE", "https://jetfern.env.example/v1")

    _, provider, _, api_base = get_llm_provider(
        model=f"jetfern/{_MODEL}",
        custom_llm_provider=None,
        api_base=None,
        api_key=None,
    )

    assert provider == "jetfern"
    assert api_base == "https://jetfern.env.example/v1"


def test_jetfern_api_base_autodetects_provider(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("JETFERN_API_KEY", "jetfern-env-key")

    model, provider, api_key, api_base = get_llm_provider(
        model=_MODEL,
        custom_llm_provider=None,
        api_base=_DEFAULT_API_BASE,
        api_key=None,
    )

    assert model == _MODEL
    assert provider == "jetfern"
    assert api_key == "jetfern-env-key"
    assert api_base == _DEFAULT_API_BASE


@pytest.mark.parametrize("model", JETFERN_MODELS)
def test_jetfern_model_cost_and_capabilities(model: str):
    from litellm.cost_calculator import cost_per_token
    from litellm.utils import (
        supports_function_calling,
        supports_prompt_caching,
        supports_reasoning,
        supports_response_schema,
        supports_vision,
    )

    entry: Final = litellm.model_cost[model]
    prompt_cost, completion_cost = cost_per_token(
        model=model,
        prompt_tokens=1_000_000,
        completion_tokens=1_000_000,
        custom_llm_provider="jetfern",
    )
    cached_prompt_cost, _ = cost_per_token(
        model=model,
        prompt_tokens=1_000_000,
        completion_tokens=0,
        cache_read_input_tokens=1_000_000,
        custom_llm_provider="jetfern",
    )
    model_info: Final = litellm.get_model_info(model)

    assert prompt_cost == pytest.approx(entry["input_cost_per_token"] * 1_000_000)
    assert completion_cost == pytest.approx(entry["output_cost_per_token"] * 1_000_000)
    assert cached_prompt_cost == pytest.approx(entry["cache_read_input_token_cost"] * 1_000_000)
    assert model_info["max_tokens"] == entry["max_output_tokens"]
    assert entry["max_output_tokens"] <= entry["max_input_tokens"]
    assert model_info["litellm_provider"] == "jetfern"
    assert supports_function_calling(model) is entry["supports_function_calling"]
    assert supports_reasoning(model) is entry["supports_reasoning"]
    assert supports_prompt_caching(model) is entry.get("supports_prompt_caching", False)
    assert supports_response_schema(model) is entry.get("supports_response_schema", False)
    assert supports_vision(model) is ("image" in entry["supported_modalities"])


@pytest.mark.usefixtures("local_model_cost_map")
def test_jetfern_wildcard_expands_to_every_jetfern_row():
    from litellm.proxy.auth.model_checks import get_known_models_from_wildcard

    assert JETFERN_MODELS
    assert set(get_known_models_from_wildcard("jetfern/*")) == set(JETFERN_MODELS)


def test_jetfern_model_missing_from_cost_map_uses_provider_defaults():
    from litellm.utils import supports_function_calling, supports_reasoning

    assert supports_function_calling("jetfern/a-model-not-in-the-cost-map") is True
    assert supports_reasoning("jetfern/a-model-not-in-the-cost-map") is True


def test_jetfern_backup_registry_mirrors_cost_map():
    package_root: Final = Path(litellm.__file__).parent
    cost_map: Final = json.loads((package_root.parent / "model_prices_and_context_window.json").read_text())
    backup: Final = json.loads((package_root / "model_prices_and_context_window_backup.json").read_text())
    jetfern_entries: Final = {name: entry for name, entry in cost_map.items() if name.startswith("jetfern/")}

    assert tuple(sorted(jetfern_entries)) == JETFERN_MODELS
    assert jetfern_entries == {name: backup[name] for name in jetfern_entries}


def test_jetfern_is_available_in_add_model_form():
    fields_path: Final = Path(litellm.__file__).parent / "proxy" / "public_endpoints" / "provider_create_fields.json"
    providers: Final = json.loads(fields_path.read_text())
    jetfern: Final = next(provider for provider in providers if provider["litellm_provider"] == "jetfern")

    assert jetfern["provider"] == "JETFERN"
    assert jetfern["default_model_placeholder"] in JETFERN_MODELS
    assert {field["key"]: field["required"] for field in jetfern["credential_fields"]} == {
        "api_base": False,
        "api_key": True,
    }


def test_jetfern_supported_endpoints_match_in_both_registries():
    backup_path: Final = Path(litellm.__file__).parent / "provider_endpoints_support_backup.json"
    root_path: Final = Path(litellm.__file__).parent.parent / "provider_endpoints_support.json"

    backup: Final = json.loads(backup_path.read_text())["providers"]["jetfern"]["endpoints"]
    root: Final = json.loads(root_path.read_text())["providers"]["jetfern"]["endpoints"]

    assert backup == root
    assert root["chat_completions"] is True
    assert root["embeddings"] is False


def test_jetfern_chat_completion_request():
    with respx.mock() as upstream:
        route: Final = upstream.post(_CHAT_COMPLETIONS_URL).respond(200, json=_CHAT_COMPLETION)
        response: Final = litellm.completion(
            model=f"jetfern/{_MODEL}",
            messages=[{"role": "user", "content": "Say hello"}],
            max_completion_tokens=64,
            api_key="jetfern-test-key",
        )

    request: Final = route.calls.last.request
    body: Final = json.loads(request.content)
    assert route.call_count == 1
    assert str(request.url) == _CHAT_COMPLETIONS_URL
    assert request.headers["authorization"] == "Bearer jetfern-test-key"
    assert body["model"] == _MODEL
    assert body["messages"] == [{"role": "user", "content": "Say hello"}]
    assert body["max_completion_tokens"] == 64
    assert response.choices[0].message.content == "Hello from JetFern"
    assert response.choices[0].message.reasoning_content == "Say hi"


def test_jetfern_tool_call_round_trip():
    with respx.mock() as upstream:
        route: Final = upstream.post(_CHAT_COMPLETIONS_URL).respond(200, json=_TOOL_CALL_COMPLETION)
        response: Final = litellm.completion(
            model=f"jetfern/{_MODEL}",
            messages=[{"role": "user", "content": "Weather in Paris?"}],
            tools=[_WEATHER_TOOL],
            api_key="jetfern-test-key",
        )

    body: Final = json.loads(route.calls.last.request.content)
    assert body["tools"] == [_WEATHER_TOOL]
    tool_call: Final = response.choices[0].message.tool_calls[0]
    assert tool_call.function.name == "get_weather"
    assert json.loads(tool_call.function.arguments) == {"city": "Paris"}
    assert response.choices[0].finish_reason == "tool_calls"
