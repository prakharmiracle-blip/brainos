"""
BrainOS — LLM Client
Supports: OpenRouter (default) | OpenAI | Together AI | Anthropic
Switch via LLM_PROVIDER env var.

OpenRouter is the recommended default — one API key, 200+ models,
many free tiers. See: https://openrouter.ai/models
"""

from __future__ import annotations

from src.config import settings
from src.utils.logger import log


class LLMClient:
    """Unified LLM interface."""

    def __init__(self):
        self.provider = settings.llm_provider
        self._client = self._build_client()
        log.info(f"LLM provider: {self.provider} | model: {settings.llm_model}")

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def chat(self, system: str, user: str, max_tokens: int = 1024) -> str:
        """Send a chat message and return the text response."""
        try:
            if self.provider == "openrouter":
                return self._openrouter_chat(system, user, max_tokens)
            elif self.provider == "openai":
                return self._openai_chat(system, user, max_tokens)
            elif self.provider == "together":
                return self._together_chat(system, user, max_tokens)
            elif self.provider == "anthropic":
                return self._anthropic_chat(system, user, max_tokens)
            else:
                raise ValueError(
                    f"Unknown LLM_PROVIDER: '{self.provider}'. "
                    "Choose: openrouter | openai | together | anthropic"
                )
        except Exception as e:
            log.error(f"LLM call failed [{self.provider}]: {e}")
            raise

    # ------------------------------------------------------------------
    # Private — client builders
    # ------------------------------------------------------------------

    def _build_client(self):
        if self.provider == "openrouter":
            from openai import OpenAI
            if not settings.openrouter_api_key:
                raise ValueError(
                    "OPENROUTER_API_KEY is not set in .env. "
                    "Get a free key at https://openrouter.ai/keys"
                )
            return OpenAI(
                api_key=settings.openrouter_api_key,
                base_url=settings.openrouter_base_url,
            )

        elif self.provider == "openai":
            from openai import OpenAI
            return OpenAI(api_key=settings.openai_api_key)

        elif self.provider == "together":
            from openai import OpenAI
            return OpenAI(
                api_key=settings.together_api_key,
                base_url="https://api.together.xyz/v1",
            )

        elif self.provider == "anthropic":
            import anthropic
            return anthropic.Anthropic(api_key=settings.anthropic_api_key)

        return None

    # ------------------------------------------------------------------
    # Private — provider chat methods
    # ------------------------------------------------------------------

    def _openrouter_chat(self, system: str, user: str, max_tokens: int) -> str:
        """
        OpenRouter uses the OpenAI SDK format but with extra headers
        for rankings/attribution on openrouter.ai dashboard.
        """
        model = settings.llm_model or "mistralai/mistral-7b-instruct:free"

        response = self._client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": user},
            ],
            max_tokens=max_tokens,
            temperature=0.3,
            extra_headers={
                "HTTP-Referer": settings.openrouter_site_url,
                "X-Title":      settings.openrouter_site_name,
            },
        )
        return response.choices[0].message.content.strip()

    def _openai_chat(self, system: str, user: str, max_tokens: int) -> str:
        model = settings.llm_model or "gpt-4o-mini"
        response = self._client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": user},
            ],
            max_tokens=max_tokens,
            temperature=0.3,
        )
        return response.choices[0].message.content.strip()

    def _together_chat(self, system: str, user: str, max_tokens: int) -> str:
        model = settings.llm_model or "mistralai/Mixtral-8x7B-Instruct-v0.1"
        response = self._client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": user},
            ],
            max_tokens=max_tokens,
            temperature=0.3,
        )
        return response.choices[0].message.content.strip()

    def _anthropic_chat(self, system: str, user: str, max_tokens: int) -> str:
        import anthropic
        model = settings.llm_model or "claude-3-haiku-20240307"
        response = self._client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return response.content[0].text.strip()


# Singleton
llm_client = LLMClient()
