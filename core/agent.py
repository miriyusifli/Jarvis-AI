from typing import Any, Dict, Iterator, List, Optional
from langchain_core.callbacks import CallbackManagerForLLMRun
from core.llm_client.client import LLMClient
from core.store.vector_store import VectorStore
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    SystemMessage,
)
from langchain_core.messages.ai import UsageMetadata
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from pydantic import Field


class CustomAgent(BaseChatModel):
    """
    Custom chat agent model that processes input messages
    and generates responses using RAG for enhanced context.
    """

    llm_client: LLMClient = Field(default=None)
    vector_store: VectorStore = Field(default=None)

    # Agent name for identification and logging
    agent_name: str = Field(default=None)

    # Maximum allowed response length (characters)
    max_response_length: Optional[int] = Field(default=None)

    def __init__(
        self,
        agent_name: str,
        llm_client: LLMClient,
        vector_store: VectorStore,
        max_response_length: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.agent_name = agent_name
        self.max_response_length = max_response_length
        self.llm_client = llm_client
        self.vector_store = vector_store

    def _get_context(self, query: str) -> str:
        """Retrieve relevant context from the vector store"""
        relevant_docs = self.vector_store.similarity_search(query)
        if not relevant_docs:
            return ""

        context_str = "\n\n".join([doc["content"] for doc in relevant_docs])
        return f"\nRelevant context:\n{context_str}\n\nBased on this context, "

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:

        # Take the last message in the conversation (user input)
        last_message = messages[-1]
        user_input = last_message.content

        # Get relevant context from vector store
        context = self._get_context(user_input)

        # Combine context with user input for enhanced response
        enhanced_input = f"{context}{user_input}"

        # Generate response with context-enhanced input
        response_text = self.agent_name + self.llm_client.chat(enhanced_input)

        # Enforce maximum response length limit if specified
        if self.max_response_length:
            response_text = response_text[: self.max_response_length]

        # Construct AIMessage with content and metadata about usage & agent info
        ai_message = AIMessage(
            content=response_text,
            additional_kwargs={},  # Additional payload, empty here
            response_metadata={
                "agent_name": self.agent_name,
            },
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
        response_text = self.agent_name + user_input
        if self.max_response_length:
            response_text = response_text[: self.max_response_length]

        # Iterate over each character in the response text to simulate streaming
        for char in response_text:

            # Create a ChatGenerationChunk containing this token
            chunk = ChatGenerationChunk(message=AIMessageChunk(content=char))

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
        return {"agent_name": self.agent_name}
