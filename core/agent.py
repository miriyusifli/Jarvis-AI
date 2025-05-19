from typing import Any, Dict, Iterator, List, Optional
from core.ollama_client import OllamaClient
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage
from langchain_core.messages.ai import UsageMetadata
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from pydantic import Field


class CustomAgent(BaseChatModel):
    """
    Custom chat agent model that processes input messages
    and generates responses. This class can be extended
    to connect to real LLM APIs, local models, or toolkits.
    """

    ollama_client: object = OllamaClient()

    # Agent name for identification and logging
    agent_name: str = Field(default=None)

    # Optional prefix prepended to every response (e.g., "Jarvis: ")
    response_prefix: Optional[str] = Field(default=None)

    # Maximum allowed response length (characters)
    max_response_length: Optional[int] = Field(default=None)

    def __init__(self, agent_name: str, **kwargs):
        super().__init__(**kwargs)
        self.agent_name = agent_name
        self.response_prefix = f"{self.agent_name}: "

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """
        Generate a complete chat response based on input messages.

        Args:
            messages: List of BaseMessage objects representing the conversation history.
            stop: List of stop tokens to terminate generation (unused here).
            run_manager: Callback manager for streaming tokens (unused here).
            kwargs: Additional parameters (ignored).

        Returns:
            ChatResult containing generated AIMessage response.
        """

        # Take the last message in the conversation (user input)
        last_message = messages[-1]
        user_input = last_message.content

        # Compose the response: prepend prefix if set, then append user input
        response_text = (self.response_prefix or "") + self.ollama_client.generate(user_input)


        # Enforce maximum response length limit if specified
        if self.max_response_length:
            response_text = response_text[: self.max_response_length]

        # Construct AIMessage with content and metadata about usage & agent info
        ai_message = AIMessage(
            content=response_text,
            additional_kwargs={},  # Additional payload, empty here
            response_metadata={
                "agent_name": self.agent_name,
            }
        )

        # Wrap the AIMessage in a ChatGeneration and then ChatResult for LangChain
        generation = ChatGeneration(message=ai_message)
        return ChatResult(generations=[generation])

    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        """
        Stream the generated response token-by-token.

        Here tokens are simulated as single characters for demonstration.

        Args:
            messages: List of BaseMessage objects representing conversation.
            stop: List of stop tokens (ignored).
            run_manager: Callback manager to notify about new tokens.
            kwargs: Additional parameters (ignored).

        Yields:
            ChatGenerationChunk objects for each character/token generated.
        """

        # Get last user message content
        last_message = messages[-1]
        user_input = last_message.content

        # Compose full response text (with prefix and length limit)
        response_text = (self.response_prefix or "") + user_input
        if self.max_response_length:
            response_text = response_text[: self.max_response_length]

        # Iterate over each character in the response text to simulate streaming
        for char in response_text:
    

            # Create a ChatGenerationChunk containing this token
            chunk = ChatGenerationChunk(
                message=AIMessageChunk(content=char)
            )

            # If a callback manager is provided, notify it of the new token
            if run_manager:
                run_manager.on_llm_new_token(char, chunk=chunk)

            # Yield the chunk for streaming consumption
            yield chunk

        # After all tokens, yield a final chunk with metadata but no content
        final_chunk = ChatGenerationChunk(
            message=AIMessageChunk(
                content="",
                response_metadata={"agent_name": self.agent_name},
            )
        )
        if run_manager:
            run_manager.on_llm_new_token("", chunk=final_chunk)
        yield final_chunk

    @property
    def _llm_type(self) -> str:
        # Identifier string describing the model type
        return "custom-chat-agent"

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        # Parameters used for identifying the model in logs/monitoring
        return {
            "agent_name": self.agent_name
        }
    
