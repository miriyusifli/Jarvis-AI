import yaml
from pathlib import Path
from core.agent import CustomAgent
from core.ollama_client import OllamaClient

def load_config():
    config_path = Path(__file__).parent / "config.yaml"
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def main():
    # Load configuration
    config = load_config()
    
    # Create Ollama client with config
    ollama_client = OllamaClient(
        model=config["llm"]["model"],
        temperature=config["llm"]["temperature"],
        max_tokens=config["llm"]["max_tokens"]
    )
    
    # Create agent with config
    agent = CustomAgent(
        agent_name=config["agent"]["name"],
        response_prefix=config["agent"]["response_prefix"],
        max_response_length=config["agent"]["max_response_length"],
        ollama_client=ollama_client
    )

    while True:
        user_input = input("You: ")
        if user_input.lower() in {"exit", "quit"}:
            print("Exiting...")
            break

        response = agent.invoke(user_input)
        print(response.content)

if __name__ == "__main__":
    main()