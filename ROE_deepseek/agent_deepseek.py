import os
import sqlite3
import requests
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime
import numpy as np
from sklearn.linear_model import LinearRegression
import ast  # For syntax validation
import asyncio  # For asynchronous execution
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

    async def solve_question(self, question, files=None, constraints=None):
        """
        Solves a given question using the AI agent.
        :param question: The question to solve (plain text).
        :param files: Optional files (e.g., database, HTML, PDF).
        :param constraints: Additional constraints or instructions.
        :return: Final answer, execution steps, and debug logs.
        """
        try:
            # Step 1: Identify the task type and extract key parameters
            task_description = await self._identify_task(question)
            print(f"Identified Task: {task_description}")

            # Step 2: Plan the solution
            solution_plan = await self._plan_solution(task_description, files, constraints)
            print(f"Solution Plan: {solution_plan}")

            # Step 3: Generate and execute code
            for attempt in range(self.max_retries):
                generated_code = await self._generate_code(solution_plan)
                print(f"Generated Code (Attempt {attempt + 1}): {generated_code}")

                # Validate syntax before execution
                if not self._validate_syntax(generated_code):
                    print("Syntax error detected. Regenerating code...")
                    continue

                execution_result = await self._execute_code(generated_code, files)
                print(f"Execution Result: {execution_result}")

                # Step 4: Debug and verify
                if "error" in execution_result.lower():
                    debug_logs = await self._debug_code(generated_code, execution_result)
                    print(f"Debug Logs: {debug_logs}")
                else:
                    debug_logs = "No errors encountered."
                    break  # Exit retry loop if execution is successful
            else:
                debug_logs = "Max retries reached. Unable to execute code successfully."

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

    async def _identify_task(self, question):
        """
        Identifies the task type and extracts key parameters.
        """
        messages = [
            {"role": "system", "content": "You are an intelligent AI agent designed to solve complex examination-style questions."},
            {"role": "user", "content": f"Identify the task type and extract key parameters from the following question: {question}"}
        ]
        response = self.chat_completion(messages)
        return response['choices'][0]['message']['content']

    async def _plan_solution(self, task_description, files, constraints):
        """
        Plans the solution by breaking the problem into logical steps.
        """
        messages = [
            {"role": "system", "content": "You are an intelligent AI agent designed to solve complex examination-style questions."},
            {"role": "user", "content": f"Plan a solution for the following task: {task_description}. Files: {files}. Constraints: {constraints}"}
        ]
        response = self.chat_completion(messages)
        return response['choices'][0]['message']['content']

    async def _generate_code(self, solution_plan):
        """
        Generates Python code to perform the required tasks.
        """
        messages = [
            {"role": "system", "content": "You are an intelligent AI agent designed to solve complex examination-style questions."},
            {"role": "user", "content": f"Generate valid Python code to implement the following solution plan: {solution_plan}. Ensure the code is syntactically correct and free from errors."}
        ]
        response = self.chat_completion(messages)
        return response['choices'][0]['message']['content']

    async def _execute_code(self, generated_code, files):
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

    async def _debug_code(self, generated_code, error_message):
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


from quart import Quart, request, jsonify, render_template

app = Quart(__name__)
agent = AIAgent()

@app.route("/", methods=["GET", "POST"])
async def index():
    if request.method == "POST":
        # Get user inputs
        form = await request.form
        question = form.get("question")
        files = await request.files.getlist("files")
        constraints = form.get("constraints")

        # Save uploaded files
        file_paths = []
        for file in files:
            file_path = os.path.join("uploads", file.filename)
            await file.save(file_path)
            file_paths.append(file_path)

        # Solve the question (asynchronous call)
        result = await agent.solve_question(question, file_paths, constraints)
        return jsonify(result)

    return await render_template("index.html")

if __name__ == "__main__":
    os.makedirs("uploads", exist_ok=True)
    app.run(debug=True)