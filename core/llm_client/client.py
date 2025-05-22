from typing import Protocol, runtime_checkable

@runtime_checkable
class LLMClient(Protocol):
    def __init__(self, model: str = "llama3.2", temperature: float = 0.7, max_tokens: int = -1):
        """Initialize the LLM client with model, temperature, and max tokens."""

    def chat(self, prompt: str) -> str:
        """Generate a response from the LLM given a prompt."""