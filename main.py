import yaml
from pathlib import Path
from core.agent import CustomAgent
from core.llm_client.ollama_client import OllamaClient
from core.store.vector_store import VectorStore
from core.store.chroma_vector_store import ChromaVectorStore
from core.util.document_loader import DocumentLoader


def load_config():
    config_path = Path(__file__).parent / "config.yaml"
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_rag_data(vector_store: VectorStore):
    """Load RAG data into the vector store"""
    loader = DocumentLoader()
    data_dir = Path(__file__).parent / "data"

    # Load all supported files from the data directory
    for file_path in data_dir.glob("**/*"):
        if file_path.suffix.lower() in [".txt", ".md", ".pdf"]:
            try:
                chunks = loader.load_and_split(str(file_path))
                metadata = [{"source": str(file_path)} for _ in chunks]
                vector_store.add_texts(chunks, metadata)
            except Exception as e:
                print(f"Error loading {file_path}: {e}")

    # Persist the vector store
    vector_store.persist()


def main():
    # Load configuration
    config = load_config()

    # Create Ollama client with config
    llm_client = OllamaClient(
        model=config["llm"]["model"],
        temperature=config["llm"]["temperature"],
        max_tokens=config["llm"]["max_tokens"],
        system_message=config["agent"]["system_message"],
    )

    # Initialize vector store
    vector_store = ChromaVectorStore()

    # Load rag data into vector store only if it's empty
    if vector_store.is_empty():
        print("Vector store is empty. Loading RAG data...")
        load_rag_data(vector_store)
    else:
        print("Using existing vector store data...")

    # Create agent with config and vector store
    agent = CustomAgent(
        agent_name=config["agent"]["name"],
        max_response_length=config["agent"].get("max_response_length"),
        llm_client=llm_client,
        vector_store=vector_store,
    )

    print(f"\nWelcome to {config['agent']['name']}! Your personal AI assistant.")
    print("Type 'exit' or 'quit' to end the conversation.\n")

    from langchain_core.messages import HumanMessage

    while True:
        user_input = input("You: ")
        if user_input.lower() in {"exit", "quit"}:
            print("Exiting...")
            break

        messages = [HumanMessage(content=user_input)]
        response = agent.invoke(messages)
        print(response.content)


if __name__ == "__main__":
    main()
