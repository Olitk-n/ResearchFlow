import json
import re
from dataclasses import dataclass, field
from typing import Any
from uuid import UUID

from litellm import acompletion, completion_cost

PROVIDER_PRESETS = {
    "openai": "https://api.openai.com/v1",
    "anthropic": "https://api.anthropic.com",
    "gemini": "https://generativelanguage.googleapis.com",
    "openrouter": "https://openrouter.ai/api/v1",
    "deepseek": "https://api.deepseek.com",
    "mimo": "https://api.xiaomimimo.com/v1",
    "qwen": "https://dashscope.aliyuncs.com/compatible-mode/v1",
    "kimi": "https://api.moonshot.cn/v1",
    "glm": "https://open.bigmodel.cn/api/paas/v4",
    "minimax": "https://api.minimax.chat/v1",
    "openai_compatible": None,
}
MODEL_PRICE_DEFAULTS = {
    ("mimo", "mimo-v2.5-pro"): (0.435, 0.87),
}


@dataclass(slots=True)
class LLMConfig:
    provider: str
    model: str
    api_key: str
    base_url: str | None = None
    budget_limit_usd: float | None = None
    spent_usd: float = 0.0
    input_price_per_million_usd: float | None = None
    output_price_per_million_usd: float | None = None
    last_call_cost_usd: float | None = None
    config_id: UUID | None = None
    usage_records: list[dict[str, Any]] = field(default_factory=list)


class ModelBudgetExceeded(RuntimeError):
    pass


def litellm_model(config: LLMConfig) -> str:
    if config.provider in {"openai", "anthropic", "gemini", "openrouter", "deepseek"}:
        return f"{config.provider}/{config.model}"
    return f"openai/{config.model}"


async def complete_json(
    config: LLMConfig,
    system: str,
    prompt: str,
    schema_hint: dict[str, Any],
    max_tokens: int = 3000,
    purpose: str = "unspecified",
) -> dict[str, Any]:
    if config.budget_limit_usd is not None and (
        config.budget_limit_usd <= 0 or config.spent_usd >= config.budget_limit_usd
    ):
        raise ModelBudgetExceeded("model task budget is exhausted")
    try:
        response = await acompletion(
            model=litellm_model(config),
            api_key=config.api_key,
            api_base=config.base_url or PROVIDER_PRESETS.get(config.provider),
            messages=[
                {"role": "system", "content": system},
                {
                    "role": "user",
                    "content": f"{prompt}\nReturn JSON matching:\n{json.dumps(schema_hint)}",
                },
            ],
            response_format={"type": "json_object"},
            temperature=0.2,
            max_tokens=max_tokens,
        )
    except Exception as exc:
        config.usage_records.append(
            {
                "purpose": purpose,
                "status": "failed",
                "error_type": type(exc).__name__,
            }
        )
        raise
    usage = getattr(response, "usage", None)
    input_tokens = getattr(usage, "prompt_tokens", None)
    output_tokens = getattr(usage, "completion_tokens", None)
    try:
        call_cost = float(completion_cost(completion_response=response))
    except Exception:
        call_cost = None
    if call_cost is None and input_tokens is not None and output_tokens is not None:
        prices = MODEL_PRICE_DEFAULTS.get((config.provider, config.model))
        input_price = config.input_price_per_million_usd
        output_price = config.output_price_per_million_usd
        if input_price is None and prices:
            input_price = prices[0]
        if output_price is None and prices:
            output_price = prices[1]
        if input_price is not None and output_price is not None:
            call_cost = (
                input_tokens * input_price + output_tokens * output_price
            ) / 1_000_000
    config.last_call_cost_usd = call_cost
    if call_cost is not None:
        config.spent_usd += call_cost
    config.usage_records.append(
        {
            "purpose": purpose,
            "status": "completed",
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cost_usd": call_cost,
        }
    )
    content = response.choices[0].message.content or ""
    content = re.sub(r"^```(?:json)?\s*|\s*```$", "", content.strip())
    return json.loads(content)


async def probe_model(config: LLMConfig) -> dict[str, Any]:
    result = await complete_json(
        config,
        system="Return a minimal capability probe. Do not add prose.",
        prompt="Return ok=true and the exact string researchflow.",
        schema_hint={"ok": True, "echo": "researchflow"},
        max_tokens=200,
        purpose="model_probe",
    )
    return {
        "reachable": result.get("ok") is True,
        "structured_output": result.get("echo") == "researchflow",
        "tools": None,
        "long_context": None,
    }
