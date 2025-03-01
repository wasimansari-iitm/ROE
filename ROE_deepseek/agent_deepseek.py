import os
import sqlite3
import requests
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime
import numpy as np
from sklearn.linear_model import LinearRegression
import ast  # For syntax validation
from flask import Flask, request, jsonify, render_template  # Import Flask and related modules

# API endpoints
AI_PROXY_URL = "https://aiproxy.sanand.workers.dev"
EMBEDDING_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"

# Set API key from environment variable
API_KEY = "eyJhbGciOiJIUzI1NiJ9.eyJlbWFpbCI6IjI0ZHMzMDAwMDkwQGRzLnN0dWR5LmlpdG0uYWMuaW4ifQ.Z9TWR3dvVwBfx2BCRG6mrAPA7pyYe8tbB_nnXEJ8-WA"

class AIAgent:
    def __init__(self):
        self.headers = {
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json"
        }
        self.max_retries = 3  # Maximum number of retries for code execution

    def _call_api(self, endpoint, payload):
        """Helper function to call the AI Proxy API."""
        url = f"{AI_PROXY_URL}{endpoint}"
        try:
            response = requests.post(url, headers=self.headers, json=payload)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as e:
            print(f"HTTP Error: {e}")
            print(f"Response: {response.text}")
            raise

    def get_embeddings(self, text):
        """Get embeddings for a given text."""
        payload = {
            "input": text,
            "model": EMBEDDING_MODEL
        }
        return self._call_api("/openai/v1/embeddings", payload)

    def chat_completion(self, messages):
        """Get chat completion using GPT-4."""
        payload = {
            "model": CHAT_MODEL,
            "messages": messages
        }
        return self._call_api("/openai/v1/chat/completions", payload)

    def _validate_syntax(self, code):
        """
        Validates the syntax of the generated Python code.
        """
        try:
            ast.parse(code)
            return True
        except SyntaxError as e:
            print(f"Syntax Error: {e}")
            return False

    def _adapt_solution_plan(self, solution_plan, error_message):
        """
        Adapts the solution plan based on the error encountered during execution.
        :param solution_plan: The current solution plan.
        :param error_message: The error message from the failed execution.
        :return: Updated solution plan.
        """
        messages = [
            {"role": "system", "content": "You are an intelligent AI agent designed to solve complex examination-style questions."},
            {"role": "user", "content": f"Adapt the following solution plan based on the error encountered: {solution_plan}. Error: {error_message}"}
        ]
        response = self.chat_completion(messages)
        return response['choices'][0]['message']['content']

    def solve_question(self, question, files=None, constraints=None):
        """
        Solves a given question using the AI agent.
        :param question: The question to solve (plain text).
        :param files: Optional files (e.g., database, HTML, PDF).
        :param constraints: Additional constraints or instructions.
        :return: Final answer, execution steps, and debug logs.
        """
        try:
            # Initialize variables
            execution_result = None
            debug_logs = "No errors encountered."
            solution_plan = None
            generated_code = None

            # Step 1: Identify the task type and extract key parameters
            task_description = self._identify_task(question)
            print(f"Identified Task: {task_description}")

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
                    continue  # Skip execution and retry

                # Execute the code
                execution_result = self._execute_code(generated_code, files)
                print(f"Execution Result: {execution_result}")

                # Step 4: Analyze the result and adapt if necessary
                if "error" in execution_result.lower():
                    debug_logs = self._debug_code(generated_code, execution_result)
                    print(f"Debug Logs: {debug_logs}")

                    # Analyze the error and adapt the solution plan
                    if "syntax error" in execution_result.lower():
                        # Syntax errors are handled by regenerating code
                        continue
                    elif "logical error" in execution_result.lower():
                        # Logical errors require adjusting the solution plan
                        solution_plan = self._adapt_solution_plan(solution_plan, execution_result)
                        print(f"Adapted Solution Plan: {solution_plan}")
                    else:
                        # Unknown errors: retry with the same plan
                        continue
                else:
                    # Execution succeeded
                    debug_logs = "No errors encountered."
                    break  # Exit retry loop if execution is successful
            else:
                # If all retries fail, provide a meaningful error message
                debug_logs = "Max retries reached. Unable to execute code successfully."
                execution_result = "Execution failed after maximum retries. Please review the debug logs."

            # Step 5: Return the final answer
            final_answer = self._format_final_answer(execution_result, solution_plan, debug_logs)
            return final_answer

        except Exception as e:
            # Handle unexpected errors
            debug_logs = f"Unexpected error: {str(e)}"
            return {
                "Final Answer": "An error occurred while solving the question.",
                "Execution Steps": "Unable to complete the task due to an error.",
                "Debug Logs": debug_logs
            }

    def _identify_task(self, question):
        """
        Identifies the task type and extracts key parameters.
        """
        messages = [
            {"role": "system", "content": "You are an intelligent AI agent designed to solve complex examination-style questions."},
            {"role": "user", "content": f"Identify the task type and extract key parameters from the following question: {question}"}
        ]
        response = self.chat_completion(messages)
        return response['choices'][0]['message']['content']

    def _plan_solution(self, task_description, files, constraints):
        """
        Plans the solution by breaking the problem into logical steps.
        """
        messages = [
            {"role": "system", "content": "You are an intelligent AI agent designed to solve complex examination-style questions."},
            {"role": "user", "content": f"Plan a solution for the following task: {task_description}. Files: {files}. Constraints: {constraints}"}
        ]
        response = self.chat_completion(messages)
        return response['choices'][0]['message']['content']

    def _generate_code(self, solution_plan):
        """
        Generates Python code to perform the required tasks.
        """
        messages = [
            {"role": "system", "content": "You are an intelligent AI agent designed to solve complex examination-style questions."},
            {"role": "user", "content": f"Generate valid Python code to implement the following solution plan: {solution_plan}. Ensure the code is syntactically correct and free from errors."}
        ]
        response = self.chat_completion(messages)
        return response['choices'][0]['message']['content']

    def _execute_code(self, generated_code, files):
        """
        Executes the generated Python code.
        """
        try:
            # Dynamically execute the generated code
            exec_globals = {}
            exec_locals = {"files": files}
            exec(generated_code, exec_globals, exec_locals)
            return exec_locals.get("result", "Code executed successfully.")
        except Exception as e:
            return f"Error during execution: {str(e)}"

    def _debug_code(self, generated_code, error_message):
        """
        Debugs the generated code if an error occurs.
        """
        messages = [
            {"role": "system", "content": "You are an intelligent AI agent designed to solve complex examination-style questions."},
            {"role": "user", "content": f"Debug the following code: {generated_code}. Error: {error_message}"}
        ]
        response = self.chat_completion(messages)
        return response['choices'][0]['message']['content']

    def _format_final_answer(self, execution_result, solution_plan, debug_logs):
        """
        Formats the final answer with execution steps and debug logs.
        """
        return {
            "Final Answer": execution_result,
            "Execution Steps": solution_plan,
            "Debug Logs": debug_logs
        }

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
        return jsonify(result)

    return render_template("index.html")

if __name__ == "__main__":
    os.makedirs("uploads", exist_ok=True)
    app.run(debug=True)