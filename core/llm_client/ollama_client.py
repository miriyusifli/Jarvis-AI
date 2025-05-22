from ollama import chat
from ollama import ChatResponse
from typing import Optional


class OllamaClient:
    def __init__(
        self,
        model: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        system_message: str = "",
    ):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.system_message = system_message

    def chat(self, prompt: str) -> str:
        response: ChatResponse = chat(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": self.system_message,
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            options={
                "temperature": self.temperature,
                "num_predict": self.max_tokens if self.max_tokens else -1,
            },
        )
        return response.message.content
