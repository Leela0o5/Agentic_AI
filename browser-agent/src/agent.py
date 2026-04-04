import os
from google import genai
from google.genai import types
from src.tools import BrowserManager

SYSTEM_PROMPT = """You are an autonomous Browser Automation Agent.
Your ONLY purpose is to navigate the web to solve user tasks. 

MANDATORY FIRST STEP: On Turn 1, you MUST always call a tool (usually `navigate` or `get_links`). NEVER provide a text-only response on your first turn.

If a user asks about anything unrelated to browsing the web, using Playwright, or your current workspace, you must politely decline and remind them that you are a browser specialist.

OPERATING PRINCIPLES:
1.  **Efficiency**: Use `get_links` to find the next page quickly. ONLY use `get_page_content` when you are on the actual page you need to read.
2.  **Robustness**: Prefer clicking elements by their visible text. If that fails, use CSS selectors.
3.  **Autonomous Loop**: If you hit a popup, try to click the 'X' or 'Close' button. If a page fails to load, try a different link or search query.
4.  **Token Management**: Do not request the same page content multiple times if you already have the information.
5.  **Perception**: If you are genuinely stuck and cannot "see" why an action is failing, use `take_screenshot` to visualize the layout.
6.  **Completion**: When you have the final answer, call `task_complete` with a concise summary.

Stay professional and focused on the task. No unnecessary conversation.
"""

class BrowserAgent:
    def __init__(self):
        # Initialize the BrowserManager and GenAI client
        self.browser_manager = BrowserManager()
        
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in environment.")
            
        self.client = genai.Client()
        
        def navigate(url: str):
            """Navigates to a URL and returns the page title."""
            pass

        def get_page_content():
            """Extracts a skeleton markdown representation of the current page's headings and buttons."""
            pass

        def get_links():
            """Returns the top 20 most relevant clickable links on the page."""
            pass

        def click_element(text_or_selector: str):
            """Clicks an element by its visible text or a CSS selector."""
            pass

        def fill_form(selector_or_text: str, value: str):
            """Fills an input field with the given value."""
            pass

        def scroll(direction: str = "down"):
            """Scrolls the page up or down."""
            pass

        def take_screenshot(filename: str = "screenshot.png"):
            """Takes a screenshot and saves it to the workspace."""
            pass

        def task_complete(summary: str):
            """Call this ONLY when the task is finished and you have the final answer."""
            pass

        self.tools = [
            navigate, get_page_content, get_links, click_element, 
            fill_form, scroll, take_screenshot, task_complete
        ]

    async def run(self, goal: str):
        print(f"\n--- Starting Browser Agent Goal: {goal} ---")
        
        # Create the chat session
        chat = self.client.chats.create(
            model="gemini-flash-latest",
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                tools=self.tools,
                temperature=0.0
            )
        )
        
        # First message to the agent
        message = f"User Goal: {goal}"
        
        max_turns = 15
        for turn in range(1, max_turns + 1):
            print(f"\n[Turn {turn}/{max_turns}]")
            
            try:
                response = chat.send_message(message)
            except Exception as e:
                print(f"Error calling Gemini: {e}")
                break

            if response.function_calls:
                tool_responses = []
                finish_early = False
                
                for call in response.function_calls:
                    func_name = call.name
                    args = call.args or {}
                    
                    print(f"  Agent Tool Call: {func_name}({args})")
                    
                    # Execute tool
                    result = ""
                    if func_name == "navigate":
                        result = await self.browser_manager.navigate(**args)
                    elif func_name == "get_page_content":
                        result = await self.browser_manager.get_page_content()
                    elif func_name == "get_links":
                        result = await self.browser_manager.get_links()
                    elif func_name == "click_element":
                        result = await self.browser_manager.click_element(**args)
                    elif func_name == "fill_form":
                        result = await self.browser_manager.fill_form(**args)
                    elif func_name == "scroll":
                        result = await self.browser_manager.scroll(**args)
                    elif func_name == "take_screenshot":
                        result = await self.browser_manager.take_screenshot(**args)
                    elif func_name == "task_complete":
                        summary = args.get("summary", "Goal reached.")
                        result = f"TASK_COMPLETE: {summary}"
                        print(f"\nGoal Reached: {summary}")
                        finish_early = True
                    else:
                        result = f"Error: Tool '{func_name}' not recognized."
                    
                    tool_responses.append(types.Part.from_function_response(
                        name=func_name,
                        response={"result": result}
                    ))
                
                if finish_early:
                    break
                    
                message = tool_responses
                
            elif response.text:
                # Agent provided ONLY a text response (stuck or finished)
                print(f"\nAgent Thinking: {response.text}")
                
                # If the agent is stuck/failing, we send the text back as a reminder to use tools
                message = f"You haven't solved the goal yet. Current status: {response.text}. Please use your tools."
            else:
                print("\n[No response from model - possibly quota or safety filter]")
                break
        
        # Cleanup
        await self.browser_manager.stop()
        print("\n--- Browser Session Finished ---")
