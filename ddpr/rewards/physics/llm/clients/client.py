"""Default synchronous model client for physics assessments."""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ddsr_bench.generation.client import ChatResponse, ClientName

from ...schemas import CompletionResult
from .bedrock import run_bedrock_responses
from .config import ClientConfig
from .openai import GenerationConfig, OpenAIChatBackend, OpenAIChatBackendConfig


def complete_judge(
    messages: list[dict[str, str]], config: ClientConfig
) -> CompletionResult:
    if config.backend == "ddsr":
        return DdsrBackend(name=config.provider).complete(messages, config)
    if config.backend == "bedrock":
        run = run_bedrock_responses(
            messages=messages,
            model=config.model,
            region=config.region,
            max_tokens=config.max_tokens,
            timeout_seconds=config.timeout_seconds,
            reasoning_effort=config.reasoning_effort,
            json_mode=True,
            max_attempts=config.http_max_attempts,
            initial_backoff_seconds=config.http_backoff_seconds,
        )
        return CompletionResult(
            content=run.content,
            reasoning_content=run.reasoning_content,
            response_id=run.response_id,
            finish_reason=run.finish_reason,
            usage=run.usage,
        )
    backend = OpenAIChatBackend(
        OpenAIChatBackendConfig(
            base_url=config.base_url,
            api_key_env=config.api_key_env,
            max_attempts=config.http_max_attempts,
            initial_backoff_seconds=config.http_backoff_seconds,
        )
    )
    return backend.complete(
        messages,
        GenerationConfig(
            model=config.model,
            max_tokens=config.max_tokens,
            timeout_seconds=config.timeout_seconds,
            temperature=0.0,
            extra_body={
                **config.extra_body,
                "response_format": {"type": "json_object"},
            },
        ),
    )


def parse_completion(response: ChatResponse) -> CompletionResult:
    """Retain visible output, separate reasoning and available request metadata."""
    response_id = response.raw.get("id")
    return CompletionResult(
        content=response.content,
        reasoning_content=response.reasoning,
        model=response.model,
        response_id=response_id if isinstance(response_id, str) else None,
        finish_reason=response.finish_reason,
        usage=response.usage,
        latency_seconds=response.latency,
    )


@dataclass(frozen=True, slots=True)
class DdsrBackend:
    """Use a DDSR-Bench provider with the judge's model, endpoint and token budget.

    The dispatcher selects name from ClientConfig.provider. DDSR-Bench owns
    retry policy; http_backoff_seconds is not used. Arbitrary extra_body settings
    are unsupported. Each call creates and closes its client on the same loop.
    """

    name: ClientName = "vllm"

    def complete(
        self, messages: list[dict[str, str]], config: ClientConfig
    ) -> CompletionResult:
        if config.extra_body:
            raise ValueError("DdsrBackend does not support extra_body")
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self._complete(messages, config))
        # Synchronous callers may already be inside an event loop. Construct the
        # client in the worker as well, so no HTTP connections cross loop boundaries.
        with ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(
                lambda: asyncio.run(self._complete(messages, config))
            ).result()

    async def _complete(
        self, messages: list[dict[str, str]], config: ClientConfig
    ) -> CompletionResult:
        from ddsr_bench.generation.client import CLIENTS, Sampling, read_api_key

        async with asyncio.timeout(config.timeout_seconds):
            async with CLIENTS[self.name](
                base_url=config.base_url,
                sampling=Sampling(
                    model=config.model,
                    max_tokens=config.max_tokens,
                    reasoning_effort=config.reasoning_effort or None,
                ),
                timeout=config.timeout_seconds,
                retries=config.http_max_attempts - 1,
                api_key=read_api_key(config.api_key_env),
            ) as client:
                response = await client.chat(
                    tuple(dict(message) for message in messages)
                )
        result = parse_completion(response)
        return result.model_copy(update={"provider": f"ddsr-bench/{self.name}"})
