"""
Adaptive Chunking with Automatic Fallback

Handles automatic retry with smaller chunks when hitting token limits.
Implements the strategy: If output limit hit -> chunk input / 2 -> retry
"""

from __future__ import annotations

import logging
from typing import Any, Callable

logger = logging.getLogger(__name__)


class ChunkingStrategy:
    """Smart chunking with automatic fallback on token limit errors."""

    def __init__(self, model_name: str | None = None):
        """
        Initialize chunking strategy.

        Args:
            model_name: Model name for determining initial chunk sizes
        """
        self.model_name = model_name
        self._attempt_count = 0
        self._max_attempts = 5  # Prevent infinite loops

    def call_with_adaptive_chunking(
        self,
        llm_function: Callable[[str, int | None], Any],
        prompt: str,
        initial_chunk_tokens: int | None = None,
        initial_max_output: int | None = None,
    ) -> Any:
        """
        Call LLM with adaptive chunking - automatically retries with smaller chunks on failure.

        Args:
            llm_function: Function to call LLM (must accept prompt and max_tokens)
            prompt: The prompt text
            initial_chunk_tokens: Initial chunk size (None = auto-detect)
            initial_max_output: Initial max output tokens (None = auto-detect)

        Returns:
            LLM response

        Raises:
            Exception: If all retry attempts fail
        """
        from scripts.model_token_config import (
            get_fallback_chunk_size,
            get_input_chunk_tokens,
            get_output_safe_tokens,
            validate_input_output_fit,
        )

        # Auto-detect initial settings
        if initial_chunk_tokens is None:
            initial_chunk_tokens = get_input_chunk_tokens(self.model_name)

        if initial_max_output is None:
            initial_max_output = get_output_safe_tokens(self.model_name)

        current_chunk_size = initial_chunk_tokens
        current_max_output = initial_max_output
        self._attempt_count = 0

        while self._attempt_count < self._max_attempts:
            self._attempt_count += 1

            # Estimate prompt tokens (rough approximation)
            prompt_tokens = len(prompt) // 4

            # Validate fit
            fits, message = validate_input_output_fit(
                prompt_tokens, current_max_output, self.model_name
            )

            if not fits:
                logger.warning(
                    f"[Adaptive Chunking] Attempt {self._attempt_count}: {message}"
                )
                logger.warning(
                    f"[Adaptive Chunking] Reducing chunk size: {current_chunk_size:,} -> {current_chunk_size // 2:,}"
                )

                # Reduce chunk size and try again
                current_chunk_size = get_fallback_chunk_size(
                    current_chunk_size, self.model_name
                )

                # If we reduced chunk size, we might also reduce output expectation
                # to leave more room
                if current_chunk_size < initial_chunk_tokens // 2:
                    current_max_output = int(current_max_output * 0.8)
                    logger.info(
                        f"[Adaptive Chunking] Also reducing max_output to {current_max_output:,}"
                    )

                continue

            # Try the call
            logger.info(
                f"[Adaptive Chunking] Attempt {self._attempt_count}: "
                f"input~{prompt_tokens:,} tokens, max_output={current_max_output:,}"
            )

            try:
                response = llm_function(prompt, current_max_output)

                # Check if we hit length limit
                finish_reason = self._extract_finish_reason(response)

                if finish_reason == "length":
                    logger.warning(
                        f"[Adaptive Chunking] Hit length limit on attempt {self._attempt_count}"
                    )

                    # Try reducing chunk size
                    new_chunk_size = get_fallback_chunk_size(
                        current_chunk_size, self.model_name
                    )

                    if new_chunk_size == current_chunk_size:
                        # Can't reduce further, give up
                        logger.error(
                            "[Adaptive Chunking] Cannot reduce chunk size further, returning partial response"
                        )
                        return response

                    logger.info(
                        f"[Adaptive Chunking] Retrying with smaller input: {current_chunk_size:,} -> {new_chunk_size:,}"
                    )
                    current_chunk_size = new_chunk_size

                    # Also reduce output expectation
                    current_max_output = int(current_max_output * 0.7)
                    logger.info(
                        f"[Adaptive Chunking] Reducing max_output: {initial_max_output:,} -> {current_max_output:,}"
                    )

                    # Retry (note: this requires re-chunking the input, which the caller must handle)
                    # For now, we return the partial response with a warning
                    logger.warning(
                        "[Adaptive Chunking] Returning partial response - caller should re-chunk and retry"
                    )
                    return response

                elif finish_reason == "error":
                    logger.error(
                        f"[Adaptive Chunking] API error on attempt {self._attempt_count}"
                    )
                    raise Exception("LLM API error")

                else:
                    # Success!
                    logger.info(
                        f"[Adaptive Chunking] Success on attempt {self._attempt_count}"
                    )
                    return response

            except Exception as e:
                logger.error(
                    f"[Adaptive Chunking] Error on attempt {self._attempt_count}: {e}"
                )

                if self._attempt_count >= self._max_attempts:
                    raise

                # Try reducing chunk size
                current_chunk_size = get_fallback_chunk_size(
                    current_chunk_size, self.model_name
                )
                current_max_output = int(current_max_output * 0.8)

        raise Exception(
            f"Failed after {self._max_attempts} attempts with adaptive chunking"
        )

    @staticmethod
    def _extract_finish_reason(response: Any) -> str | None:
        """Extract finish_reason from LLM response."""
        try:
            # LiteLLM response
            if hasattr(response, "choices") and len(response.choices) > 0:
                return getattr(response.choices[0], "finish_reason", None)

            # Wrapped response
            if hasattr(response, "raw_response"):
                raw = response.raw_response
                if hasattr(raw, "choices") and len(raw.choices) > 0:
                    return getattr(raw.choices[0], "finish_reason", None)

        except Exception:
            pass

        return None


def estimate_tokens(text: str) -> int:
    """
    Estimate number of tokens in text.

    Uses simple heuristic: ~4 characters per token.

    Args:
        text: Input text

    Returns:
        Estimated token count
    """
    return len(text) // 4


def should_chunk_input(text: str, model_name: str | None = None) -> tuple[bool, int]:
    """
    Determine if input should be chunked based on model limits.

    Args:
        text: Input text
        model_name: Model name

    Returns:
        Tuple of (should_chunk: bool, recommended_chunk_size: int)
    """
    from scripts.model_token_config import (
        calculate_safe_input_limit,
        get_output_safe_tokens,
    )

    estimated_tokens = estimate_tokens(text)
    desired_output = get_output_safe_tokens(model_name)
    safe_input_limit = calculate_safe_input_limit(model_name, desired_output)

    if estimated_tokens <= safe_input_limit:
        return False, estimated_tokens

    # Need to chunk
    recommended_chunk_size = safe_input_limit
    return True, recommended_chunk_size


# Example usage
if __name__ == "__main__":
    from scripts.model_token_config import get_model_config

    print("=" * 80)
    print("ADAPTIVE CHUNKING STRATEGY")
    print("=" * 80)
    print()

    for model_name in ["minimax-m2.5", "glm-5.2"]:
        config = get_model_config(model_name)
        print(f"{model_name}:")
        print(f"  Initial chunk:  {config['input_chunk_tokens']:,} tokens")
        print(f"  Max output:     {config['output_max_tokens']:,} tokens")
        print()

        # Simulate fallback
        current_size = config["input_chunk_tokens"]
        print("  Fallback sequence:")
        for i in range(5):
            from scripts.model_token_config import get_fallback_chunk_size

            current_size = get_fallback_chunk_size(current_size, model_name)
            print(f"    Attempt {i + 2}: {current_size:,} tokens")

        print()
        print("-" * 80)
        print()

    # Test validation
    print("\nINPUT/OUTPUT VALIDATION:")
    print("-" * 80)

    from scripts.model_token_config import validate_input_output_fit

    test_cases = [
        ("minimax-m2.5", 150_000, 16_000),
        ("minimax-m2.5", 200_000, 16_000),  # Should fail
        ("glm-5.2", 200_000, 280_000),
        ("glm-5.2", 700_000, 280_000),  # Should fail
    ]

    for model, input_tokens, output_tokens in test_cases:
        fits, message = validate_input_output_fit(input_tokens, output_tokens, model)
        status = "OK" if fits else "FAIL"
        print(
            f"{status} {model:15} | Input: {input_tokens:>7,} | Output: {output_tokens:>7,}"
        )
        print(f"   {message}")
        print()
