import os
import asyncio
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from src.agent import BrowserAgent

async def main():
    agent = BrowserAgent()
    
    # We could take goal from CLI, but let's make it interactive
    print("Initializing Browser Automation Agent...")
    print("Welcome! Goal examples: 'Go to google.com and get page title', 'Search for Playwright on Wikipedia'")
    
    while True:
        try:
            goal = input("\nYou: ")
            if goal.lower() in ['exit', 'quit']:
                break
            if not goal.strip():
                continue
            
            print("\nThinking...")
            # We must use asyncio.run to call the async agent.run
            await agent.run(goal)
            
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"\nAn error occurred: {e}\n")

if __name__ == "__main__":
    asyncio.run(main())
