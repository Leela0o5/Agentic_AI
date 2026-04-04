import os
import sys
import io
import contextlib
import pandas as pd

# Ensure the workspace exists
WORKSPACE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "workspace"))
os.makedirs(WORKSPACE_DIR, exist_ok=True)

# This dictionary holds the variables between code executions
AGENT_GLOBALS = {
    "pd": pd,
}

def _resolve_path(filepath: str) -> str:
    """Safely resolve paths to ensure they stay inside the workspace."""
    safe_path = os.path.normpath(filepath).lstrip("\\/")
    return os.path.join(WORKSPACE_DIR, safe_path)

def inspect_data(filepath: str) -> str:
    """Read a CSV file to show its schema, column types, and first 5 rows. MUST use this before answering any questions about a dataset."""
    target_path = _resolve_path(filepath)
    if not os.path.exists(target_path):
        return f"Error: File {filepath} does not exist in workspace. Did you put it in the workspace folder?"
    
    try:
        df = pd.read_csv(target_path)
        buffer = io.StringIO()
        df.info(buf=buffer)
        info_str = buffer.getvalue()
        head_str = df.head().to_markdown()
        return f"=== SCHEMA ===\n{info_str}\n\n=== FIRST 5 ROWS ===\n{head_str}"
    except Exception as e:
        return f"Error inspecting data: {e}"

def execute_python_code(code: str) -> str:
    """Executes Python code in a persistent environment. Good for pandas data manipulation. Be sure to print() the results you want to see."""
    output_buffer = io.StringIO()
    
    # Run from the workspace directory so relative paths work natively
    original_cwd = os.getcwd()
    os.chdir(WORKSPACE_DIR)
    
    try:
        with contextlib.redirect_stdout(output_buffer), contextlib.redirect_stderr(output_buffer):
            try:
                exec(code, AGENT_GLOBALS)
            except Exception as e:
                import traceback
                print(traceback.format_exc())
    finally:
        os.chdir(original_cwd)
        
    return output_buffer.getvalue() or "Code executed successfully with no output."

def save_chart(code: str, filename: str) -> str:
    """Executes matplotlib code and saves the current figure to a PNG file in the workspace. Example filename: 'chart.png'"""
    target_path = _resolve_path(filename)
    if not target_path.endswith('.png'):
        target_path += '.png'
    
    full_code = f"import matplotlib.pyplot as plt\n{code}\nplt.savefig(r'{target_path}')\nplt.close()"
    
    output = execute_python_code(full_code)
    
    if os.path.exists(target_path):
        return f"Chart saved successfully at {filename}.\nOutput: {output}"
    else:
        return f"Failed to save chart. Make sure your code plots something. Output: {output}"

def write_report(report_content: str, filename: str) -> str:
    """Writes a markdown report of findings to the workspace."""
    target_path = _resolve_path(filename)
    if not target_path.endswith('.md'):
        target_path += '.md'
        
    try:
        with open(target_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
        return f"Report saved to {filename}"
    except Exception as e:
        return f"Error writing report: {e}"
