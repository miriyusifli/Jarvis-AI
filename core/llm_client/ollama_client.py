from ollama import chat
from ollama import ChatResponse
from typing import Optional


class OllamaClient:
    def __init__(
        self,
        model: str = "llama3.2",
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def chat(self, prompt: str) -> str:
        response: ChatResponse = chat(
            model=self.model,
            messages=[
                {
                    ##TODO try to get system message from config
                    "role": "system",
                    "content": """
                        Your personal name is Jarvis.  You are a helpful AI assistant serves for Miri with access to his personal information. Your primary goals are:
                            - Provide accurate and helpful responses using the available information
                            - Protect sensitive personal information while being helpful
                            - Use a friendly and professional tone
                            - When accessing personal data, only share information that is directly relevant to the query
                    """,
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
