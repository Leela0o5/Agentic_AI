import os
from dotenv import load_dotenv
load_dotenv()

from src.agent import DataAnalystAgent

def main():
    print("Initializing Data Analyst Agent...")
    try:
        agent = DataAnalystAgent()
    except Exception as e:
        print(f"Failed to start agent: {e}")
        print("Did you forget to set GEMINI_API_KEY in your .env file?")
        return

    print("\nAgent Ready!")
    print("Put your CSV files in the 'workspace' directory.")
    print("Type 'exit' or 'quit' to stop.\n")

    while True:
        try:
            user_input = input("You: ")
            if user_input.lower() in ['exit', 'quit']:
                break
            if not user_input.strip():
                continue
            
            print("\nThinking...")
            response = agent.ask(user_input)
            print(f"\nAgent: {response}\n")
            
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"\nAn error occurred: {e}\n")

if __name__ == "__main__":
    main()
