import os
import openai
import json
import traceback
import sqlite3
import pdfplumber
from bs4 import BeautifulSoup
from flask import Flask, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename

# Load API Key from environment variable
API_KEY = os.getenv("AI_PROXY_API_KEY")
BASE_URL = "https://aiproxy.sanand.workers.dev/openai/v1"

# OpenAI API client
openai.api_key = API_KEY
openai.base_url = BASE_URL

app = Flask(__name__, static_folder="frontend")
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Task Identification
def get_task_type(question):
    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You classify questions into task types and extract key details."},
            {"role": "user", "content": f"Classify this question and extract necessary details: {question}"}
        ]
    )
    return response["choices"][0]["message"]["content"]

# Code Generation
def generate_code(task_details):
    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You generate Python code to solve examination-style questions."},
            {"role": "user", "content": f"Generate Python code to perform this task: {task_details}"}
        ]
    )
    return response["choices"][0]["message"]["content"]

# Code Execution
def execute_code(code):
    try:
        exec_locals = {}
        exec(code, {}, exec_locals)
        return exec_locals.get("result", "No result returned")
    except Exception as e:
        return f"Error executing code: {traceback.format_exc()}"

# Database Querying
def query_database(db_path, sql_query):
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute(sql_query)
        results = cursor.fetchall()
        conn.close()
        return results
    except Exception as e:
        return str(e)

# HTML Parsing
def parse_html(file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            soup = BeautifulSoup(f, "html.parser")
        return soup.prettify()
    except Exception as e:
        return str(e)

# PDF Extraction
def extract_text_from_pdf(pdf_path):
    try:
        text = ""
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                text += page.extract_text() + "\n"
        return text
    except Exception as e:
        return str(e)

# Main Solve Function
def solve_question(question):
    task_details = get_task_type(question)
    code = generate_code(task_details)
    result = execute_code(code)
    return {"question": question, "task": task_details, "code": code, "result": result}

# API Endpoints
@app.route("/")
def serve_index():
    return send_from_directory("frontend", "index.html")

@app.route("/solve", methods=["POST"])
def solve():
    data = request.get_json()
    if not data or "question" not in data:
        return jsonify({"error": "Question is required"}), 400
    
    question = data["question"]
    solution = solve_question(question)  # Ensure this function exists
    return jsonify(solution)

@app.route("/upload", methods=["POST"])
def upload_file():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400
    
    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    
    try:
        file.save(file_path)
        return jsonify({"message": "File uploaded successfully", "file_path": file_path})
    except Exception as e:
        return jsonify({"error": f"File upload failed: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)