"""
Enhanced Groq Client for Haystack — RotatableGroqGenerator.

Wraps OpenAIChatGenerator with Groq's base URL and adds:
1. API Key Rotation
2. Rate Limit (429) handling
3. TPM (Tokens Per Minute) limit handling
4. Exponential Backoff with jitter
5. Batch buffering with cooldown
"""

import time
import random
import logging
from typing import List, Any, Dict, Optional

from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.dataclasses import ChatMessage
from haystack.utils import Secret

logger = logging.getLogger(__name__)

GROQ_BASE_URL = "https://api.groq.com/openai/v1"


class RotatableGroqGenerator:
    """
    Haystack-compatible Groq generator with key rotation and rate limiting.

    Exposes a `run(messages=...)` interface matching Haystack ChatGenerators.
    """

    def __init__(
        self,
        api_keys: List[str],
        model: str = "llama-3.3-70b-versatile",
        buffer_time: float = 1.5,
        batch_size: int = 10,
        batch_cooldown: float = 30.0,
        tpm_limit: int = 5500,
        tpm_window: float = 60.0,
    ):
        if not api_keys:
            raise ValueError("No Groq API keys provided in configuration")

        self.api_keys = api_keys
        self.model = model
        self.current_key_index = 0
        self.buffer_time = buffer_time
        self.batch_size = batch_size
        self.batch_cooldown = batch_cooldown
        self.tpm_limit = tpm_limit
        self.tpm_window = tpm_window

        self._request_count = 0
        self._token_usage: List[tuple] = []  # [(timestamp, tokens)]
        self._last_rotation: float = 0
        self._current_generator: Optional[OpenAIChatGenerator] = None

        self._rotate_client()

    def _rotate_client(self):
        """Switch to the next API key and create a new generator."""
        key = self.api_keys[self.current_key_index]
        masked_key = f"{key[:4]}...{key[-4:]}" if len(key) > 8 else "********"
        logger.info(f"Rotating to Groq Key Index {self.current_key_index} ({masked_key})")

        self._current_generator = OpenAIChatGenerator(
            api_key=Secret.from_token(key),
            model=self.model,
            api_base_url=GROQ_BASE_URL,
            generation_kwargs={"temperature": 0.1},
        )
        self._current_generator.warm_up()

        self.current_key_index = (self.current_key_index + 1) % len(self.api_keys)
        self._last_rotation = time.time()
        self._token_usage = []

    def _estimate_tokens(self, messages: List[ChatMessage]) -> int:
        """Rough token estimate: ~4 chars per token."""
        total_chars = sum(len(str(msg.text)) for msg in messages)
        return int(total_chars / 4) + 100

    def _get_current_tpm(self) -> int:
        """Calculate current tokens per minute usage."""
        current_time = time.time()
        cutoff = current_time - self.tpm_window
        self._token_usage = [(ts, t) for ts, t in self._token_usage if ts > cutoff]
        return sum(t for _, t in self._token_usage)

    def _track_token_usage(self, estimated_tokens: int):
        self._token_usage.append((time.time(), estimated_tokens))

    def _should_wait_for_tpm(self, estimated_tokens: int) -> Optional[float]:
        current_tpm = self._get_current_tpm()
        if current_tpm + estimated_tokens > self.tpm_limit:
            if self._token_usage:
                oldest_time = self._token_usage[0][0]
                time_passed = time.time() - oldest_time
                return max(0, self.tpm_window - time_passed + 1)
        return None

    def run(self, messages: List[ChatMessage], **kwargs) -> Dict[str, Any]:
        """Execute with retry, matching Haystack generator interface.

        Returns:
            Dict with "replies" key containing list of ChatMessage.
        """
        max_retries = 10
        attempt = 0

        while attempt < max_retries:
            try:
                estimated_tokens = self._estimate_tokens(messages)

                # Check TPM limit
                tpm_wait = self._should_wait_for_tpm(estimated_tokens)
                if tpm_wait:
                    logger.warning(
                        f"TPM limit approaching ({self._get_current_tpm()}/{self.tpm_limit}). "
                        f"Waiting {tpm_wait:.1f}s..."
                    )
                    time.sleep(tpm_wait)

                # Batch cooldown
                if self._request_count > 0 and self._request_count % self.batch_size == 0:
                    logger.info(
                        f"Groq Batch limit ({self.batch_size}) reached. "
                        f"Cooling down for {self.batch_cooldown}s..."
                    )
                    time.sleep(self.batch_cooldown)
                else:
                    time.sleep(self.buffer_time)

                # Execute
                result = self._current_generator.run(messages=messages, **kwargs)

                self._request_count += 1
                self._track_token_usage(estimated_tokens)
                return result

            except Exception as e:
                error_msg = str(e).lower()

                is_rate_limit = "429" in error_msg or "rate limit" in error_msg
                is_tpm_limit = "tokens per minute" in error_msg or "tpm" in error_msg
                is_tool_error = "tool_use_failed" in error_msg

                if is_tool_error:
                    logger.warning(f"Groq tool-call format error (attempt {attempt + 1}/3): {e}")
                    if attempt >= 2:
                        raise
                    time.sleep(1.0)
                elif is_rate_limit or is_tpm_limit:
                    limit_type = "TPM" if is_tpm_limit else "Rate"
                    logger.warning(f"Groq {limit_type} Limit detected. Error: {e}")
                    self._rotate_client()
                    base_wait = 5 if is_tpm_limit else 2
                    wait_time = base_wait * (2 ** min(attempt, 4)) + random.uniform(0, 2)
                    logger.info(f"Waiting {wait_time:.1f}s before retry with new key...")
                    time.sleep(wait_time)
                else:
                    wait_time = (2 ** attempt) + random.uniform(0, 1)
                    logger.error(f"Groq Error: {e}. Retrying in {wait_time:.1f}s...")
                    time.sleep(wait_time)

                attempt += 1

        raise Exception(
            f"Max retries ({max_retries}) exceeded for Groq API. "
            f"Last error may indicate persistent rate limiting or API issues."
        )

    def get_stats(self) -> Dict[str, Any]:
        """Get current usage statistics."""
        return {
            "current_key_index": self.current_key_index,
            "total_requests": self._request_count,
            "current_tpm": self._get_current_tpm(),
            "tpm_limit": self.tpm_limit,
            "time_since_rotation": time.time() - self._last_rotation,
        }
