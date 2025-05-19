from core.agent import CustomAgent

# TODO fields should be used
agent = CustomAgent(agent_name="Jarvis")

def main():
    while True:
        user_input = input("You: ")
        if user_input.lower() in {"exit", "quit"}:
            print("Exiting...")
            break

        response = agent.invoke(user_input)
        print(response.content)

if __name__ == "__main__":
    main()