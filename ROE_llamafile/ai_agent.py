import os
import requests
import ast
import re
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify, render_template

# Llamafile API endpoint
LLAMAFILE_URL = "http://127.0.0.1:8080"

class AIAgent:
    def __init__(self):
        self.max_retries = 3  # Maximum number of retries for code execution
        self.quality_metrics = {
            'syntax_errors': 0,
            'runtime_errors': 0,
            'success_rate': 0
        }

    def _call_llamafile(self, prompt):
        """
        Calls the local Llamafile API to generate a response.
        """
        payload = {
            "prompt": prompt,
            "temperature": 0.3,        # Reduced for more deterministic output
            "max_tokens": 800,         # Increased for longer code completion
            "top_p": 0.9,             # Focus on high-probability tokens
            "repeat_penalty": 1.15,    # Reduce code repetition
            "stop": ["```"]            # Stop sequence for code blocks
        }
        try:
            print(f"Sending request to Llamafile API with prompt: {prompt}")
            response = requests.post(f"{LLAMAFILE_URL}/completion", json=payload)
            response.raise_for_status()
            response_json = response.json()
            print(f"Received response from Llamafile API: {response_json}")
            
            # Handle empty or invalid response
            if "content" not in response_json:
                print("Warning: No 'content' field in API response.")
                return "Error: No response generated."
            
            return response_json["content"]
        except requests.exceptions.RequestException as e:
            print(f"Error calling Llamafile: {e}")
            raise

    def _validate_syntax(self, code):
        """
        Validates the syntax of the generated Python code.
        """
        try:
            # Remove markdown code blocks
            clean_code = re.sub(r'^```python.*?^```', '', code, flags=re.DOTALL|re.MULTILINE)
            ast.parse(clean_code)
            return True
        except SyntaxError as e:
            print(f"Syntax Error: {e}")
            self.quality_metrics['syntax_errors'] += 1
            return False

    def _adapt_solution_plan(self, solution_plan, error_message):
        """
        Adapts the solution plan based on the error encountered during execution.
        """
        prompt = f"""Fix this Python code based on the error:
        
        Error: {error_message}
        
        Original Code:
        {solution_plan}
        
        Provide the corrected code with:
        1. Detailed comments explaining the fix
        2. Proper error handling
        3. PEP8 compliance
        
        Format as:
        ```python
        # Fixed code
        ```"""
        return self._call_llamafile(prompt)

    def solve_question(self, question, files=None, constraints=None):
        """
        Solves a given question using the AI agent.
        """
        attempts = []  # Store results of each attempt
        try:
            # Step 1: Identify the task type and extract key parameters
            task_description = self._identify_task(question)
            print(f"Identified Task: {task_description}")
    
            # Check if the task is fact-based
            if "fact-based" in task_description.lower():
                # Handle fact-based questions directly
                final_answer = self._call_llamafile(f"Answer the following question: {question}")
                return {
                    "attempts": [{
                        "final_answer": final_answer,
                        "execution_steps": "Direct answer for fact-based question.",
                        "debug_logs": "No errors encountered."
                    }]
                }
    
            # Step 2: Plan the solution
            solution_plan = self._plan_solution(task_description, files, constraints)
            print(f"Solution Plan: {solution_plan}")
    
            # Step 3: Generate and execute code with adaptive retries
            for attempt in range(self.max_retries):
                print(f"Attempt {attempt + 1} of {self.max_retries}")
    
                # Generate code based on the current solution plan
                generated_code = self._generate_code(solution_plan)
                print(f"Generated Code: {generated_code}")
    
                # Validate syntax before execution
                if not self._validate_syntax(generated_code):
                    debug_logs = "Syntax error detected. Regenerating code..."
                    print(debug_logs)
                    attempts.append({
                        "final_answer": "Syntax error detected.",
                        "execution_steps": solution_plan,
                        "debug_logs": debug_logs
                    })
                    continue  # Skip execution and retry
                
                # Execute the code
                execution_result = self._execute_code(generated_code, files)
                print(f"Execution Result: {execution_result}")
    
                # Step 4: Analyze the result and adapt if necessary
                if "error" in execution_result.lower():
                    debug_logs = self._debug_code(generated_code, execution_result)
                    print(f"Debug Logs: {debug_logs}")
                    self.quality_metrics['runtime_errors'] += 1
    
                    # Analyze the error and adapt the solution plan
                    if "syntax error" in execution_result.lower():
                        # Syntax errors are handled by regenerating code
                        attempts.append({
                            "final_answer": "Syntax error detected.",
                            "execution_steps": solution_plan,
                            "debug_logs": debug_logs
                        })
                        continue
                    elif "logical error" in execution_result.lower():
                        # Logical errors require adjusting the solution plan
                        solution_plan = self._adapt_solution_plan(solution_plan, execution_result)
                        print(f"Adapted Solution Plan: {solution_plan}")
                        attempts.append({
                            "final_answer": "Logical error detected.",
                            "execution_steps": solution_plan,
                            "debug_logs": debug_logs
                        })
                    else:
                        # Unknown errors: retry with the same plan
                        attempts.append({
                            "final_answer": "Unknown error detected.",
                            "execution_steps": solution_plan,
                            "debug_logs": debug_logs
                        })
                        continue
                else:
                    # Execution succeeded
                    debug_logs = "No errors encountered."
                    self.quality_metrics['success_rate'] += 1
                    attempts.append({
                        "final_answer": execution_result,
                        "execution_steps": solution_plan,
                        "debug_logs": debug_logs
                    })
                    break  # Exit retry loop if execution is successful
            else:
                # If all retries fail, provide a meaningful error message
                debug_logs = "Max retries reached. Unable to execute code successfully."
                execution_result = "Execution failed after maximum retries. Please review the debug logs."
                attempts.append({
                    "final_answer": execution_result,
                    "execution_steps": solution_plan,
                    "debug_logs": debug_logs
                })
    
            # Step 5: Return the final answer
            return {
                "attempts": attempts,
                "quality_metrics": self.quality_metrics
            }
    
        except Exception as e:
            # Handle unexpected errors
            debug_logs = f"Unexpected error: {str(e)}"
            return {
                "attempts": [{
                    "final_answer": "An error occurred while solving the question.",
                    "execution_steps": "Unable to complete the task due to an error.",
                    "debug_logs": debug_logs
                }],
                "quality_metrics": self.quality_metrics
            }

    def _identify_task(self, question):
        """
        Identifies the task type and extracts key parameters.
        """
        prompt = f"Identify the task type and extract key parameters from the following question: {question}. If the question is fact-based (e.g., 'What is the capital of the United Kingdom?'), respond with 'fact-based' and the key parameters (e.g., 'capital of the United Kingdom'). Otherwise, respond with 'other'."
        response = self._call_llamafile(prompt)
        return response

    def _plan_solution(self, task_description, files, constraints):
        """
        Plans the solution by breaking the problem into logical steps.
        """
        prompt = f"Plan a solution for the following task: {task_description}. Files: {files}. Constraints: {constraints}. The solution should include steps for data extraction, analysis, code generation, or any other necessary actions."
        return self._call_llamafile(prompt)

    def _generate_code(self, solution_plan):
        """
        Generates Python code to perform the required tasks.
        """
        prompt = f"""Generate Python code that follows these requirements:
1. Strict PEP8 compliance
2. Proper indentation (4 spaces)
3. Includes necessary imports
4. Uses pandas for data analysis
5. Includes error handling
6. Returns results in a 'result' variable

Problem: {solution_plan}

Format your response as:
```python
# Your code here
```"""
        return self._call_llamafile(prompt)

    def _execute_code(self, generated_code, files):
        """
        Executes the generated Python code.
        """
        try:
            # Prepend essential imports
            header = "import pandas as pd\nimport numpy as np\nfrom typing import Any\n"
            sanitized_code = header + re.sub(r'^```python.*?^```', '', generated_code, flags=re.DOTALL|re.MULTILINE)
            
            print(f"Executing code:\n{sanitized_code}")
            
            exec_globals = {'pd': pd, 'np': np}
            exec_locals = {"files": files}
            exec(sanitized_code, exec_globals, exec_locals)
            result = exec_locals.get('result', 'Execution completed')
            print(f"Execution result: {result}")
            return result
        except Exception as e:
            error_message = f"Error during execution: {str(e)}"
            print(error_message)
            return error_message

    def _debug_code(self, generated_code, error_message):
        """
        Debugs the generated code if an error occurs.
        """
        prompt = f"Debug the following code: {generated_code}. Error: {error_message}"
        return self._call_llamafile(prompt)

app = Flask(__name__)
agent = AIAgent()

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Get user inputs
        question = request.form.get("question")
        files = request.files.getlist("files")
        constraints = request.form.get("constraints")

        # Save uploaded files
        file_paths = []
        for file in files:
            file_path = os.path.join("uploads", file.filename)
            file.save(file_path)
            file_paths.append(file_path)

        # Solve the question
        result = agent.solve_question(question, file_paths, constraints)
        return render_template("index.html", result=result)

    return render_template("index.html")

if __name__ == "__main__":
    os.makedirs("uploads", exist_ok=True)
    app.run(debug=True)