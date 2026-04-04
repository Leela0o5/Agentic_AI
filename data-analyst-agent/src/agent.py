import os
from google import genai
from google.genai import types

from src.tools import inspect_data, execute_python_code, save_chart, write_report

SYSTEM_PROMPT = """You are an autonomous Data Analyst Agent.
Your ONLY purpose is to answer user questions about data by writing and running Python code.

If a user asks about anything unrelated to data analysis, Python, charts, or your workspace (e.g., general knowledge, personal opinions, or trivia), you must politely decline and state that you are only here to analyze data for them.

CRITICAL RULES:
1. ALWAYS use the `inspect_data` tool on a new dataset BEFORE you try to answer any questions about it. You must understand the schema and column names first. If you guess column names, you will fail.
2. You don't "know" what's in the data. You must compute it. Use `execute_python_code` to write Pandas code and `print()` the results so you can read them in the output.
3. Your state is persistent. If you import pandas as pd or create a dataframe `df`, it will be available in the next tool call or the next question.
4. When writing python code, be sure to `print()` the specific values you are looking for. 
5. NEVER just explain what you *would* do. DO IT. Write the code, run it, look at the result, and then give the user the final answer.
6. If the user asks for a chart, use the `save_chart` tool.
7. If the user asks to save a report, use the `write_report` tool.
"""

class DataAnalystAgent:
    def __init__(self):
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable not set.")

        self.client = genai.Client()
        self.tools = [inspect_data, execute_python_code, save_chart, write_report]
        
        config = types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            tools=self.tools,
            temperature=0.1, # Keep it deterministic for code writing
        )
        
        self.chat = self.client.chats.create(
            model="gemini-flash-latest",
            config=config
        )

    def ask(self, user_prompt: str):
        # We loop internally to handle tool calls up to a reasonable limit
        max_tool_loops = 10
        
        try:
            response = self.chat.send_message(user_prompt)
        except Exception as e:
            return f"Error communicating with Gemini: {e}"

        attempts = 0
        while attempts < max_tool_loops:
            attempts += 1
            
            if response.function_calls:
                tool_responses = []
                for call in response.function_calls:
                    func_name = call.name
                    args = call.args or {}
                    
                    print(f"  [Agent is using tool: {func_name}]")
                    
                    # Execute tool
                    result_str = ""
                    if func_name == "inspect_data":
                        result_str = inspect_data(**args)
                    elif func_name == "execute_python_code":
                        result_str = execute_python_code(**args)
                    elif func_name == "save_chart":
                        result_str = save_chart(**args)
                    elif func_name == "write_report":
                        result_str = write_report(**args)
                    else:
                        result_str = f"Error: Unknown tool {func_name}"
                        
                    tool_responses.append(types.Part.from_function_response(
                        name=func_name,
                        response={"result": result_str}
                    ))
                
                # Send tool results back to Gemini
                try:
                    response = self.chat.send_message(tool_responses)
                except Exception as e:
                    return f"Error communicating with Gemini during tool loop: {e}"
            else:
                # Agent provided a text response
                return response.text
                
        return "Error: Agent got stuck in a tool loop."
