from ollama import chat
from ollama import ChatResponse
from typing import Optional

class OllamaClient:
    def __init__(self, model: str = "llama2", temperature: float = 0.7, max_tokens: Optional[int] = None):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def generate(self, prompt: str) -> str:
        response: ChatResponse = chat(
            model=self.model,
            messages=[{
                'role': 'user',
                'content': prompt,
            }],
            options={
                'temperature': self.temperature,
                'num_predict': self.max_tokens if self.max_tokens else -1,
            }
        )
        return response.message.content
