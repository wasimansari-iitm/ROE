import os
import re
import json
import zipfile
import logging
import requests
import openai
from flask import Flask, request, render_template, jsonify
from pathlib import Path
from bs4 import BeautifulSoup
from geopy.distance import geodesic

# Configure logging
logging.basicConfig(level=logging.INFO)

app = Flask(__name__, template_folder="templates")
app.config['UPLOAD_FOLDER'] = 'workspace'
ALLOWED_EXTENSIONS = {'zip', 'csv', 'json', 'txt', 'xml', 'pdf', 'md', 'db', 'sqlite', 'sql', 'duckdb', 'parquet', 'html', 'png', 'jpg', 'jpeg'}

# OpenAI API Configuration with AI Proxy
client = openai.OpenAI(
    base_url="https://aiproxy.sanand.workers.dev/openai/v1/",
    api_key=os.getenv("OPENAI_API_KEY")  # Ensure this is set in your environment variables
)

class ExamAgent:
    def __init__(self):
        self.token_usage = 0
        self.remaining_tokens = 1_000_000
        self.workspace = Path(app.config['UPLOAD_FOLDER'])
        self.workspace.mkdir(exist_ok=True)
        
        self.tools = {
            'scrape_web': self.scrape_web,
            'extract_zip': self.extract_zip,
            'calculate_distance': self.calculate_distance
        }

    def track_tokens(self, response):
        usage = response.usage
        self.token_usage += usage.total_tokens
        self.remaining_tokens -= usage.total_tokens

    def scrape_web(self, url):
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')
            return soup.get_text()
        except requests.RequestException as e:
            logging.error(f"Web scraping error: {e}")
            return f"Error: {str(e)}"

    def extract_zip(self, path):
        try:
            with zipfile.ZipFile(path, 'r') as zip_ref:
                zip_ref.extractall(self.workspace)
            return f"Extracted to {self.workspace}"
        except zipfile.BadZipFile as e:
            logging.error(f"ZIP extraction error: {e}")
            return f"Error: {str(e)}"

    def calculate_distance(self, coords1, coords2):
        try:
            return geodesic(coords1, coords2).kilometers
        except Exception as e:
            logging.error(f"Distance calculation error: {e}")
            return f"Error: {str(e)}"

    def generate_code(self, prompt, context):
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": f"You are an exam-solving AI.\nAvailable tools: {list(self.tools.keys())}\nContext: {context}\nReturn ONLY CODE wrapped in triple backticks."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1000,  # Adjust based on your needs
                temperature=0.7,
                top_p=1.0
            )

            if response and response.choices:
                self.track_tokens(response)
                return response.choices[0].message.content
            else:
                logging.error("OpenAI API response is empty or invalid.")
                return "Error: OpenAI API response is invalid."
        except Exception as e:
            logging.error(f"Code generation error: {e}")
            return f"Error: {str(e)}"

    def execute_code(self, code):
        try:
            restricted_globals = {"__builtins__": None}
            local_vars = {}
            exec(code, restricted_globals, local_vars)
            return local_vars.get("result", "No result found")
        except Exception as e:
            logging.error(f"Code execution error: {e}")
            return f"Execution error: {str(e)}"

    def process_task(self, user_input, files=[]):
        data_sources = []
        
        for file in files:
            if '.' in file.filename and file.filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS:
                file_path = self.workspace / file.filename
                file.save(file_path)
                if file_path.suffix == '.zip':
                    self.extract_zip(file_path)
                    data_sources.append(f"Extracted {file.filename}")
            else:
                logging.warning(f"Unsupported file: {file.filename}")
                return "Unsupported file type."

        match = re.search(r'https?://\S+', user_input)
        if match:
            url = match.group()
            data = self.scrape_web(url)
            data_sources.append(f"Scraped {url}")

        code = self.generate_code(user_input, "\n".join(data_sources))
        matches = re.findall(r'```python(.*?)```', code, re.DOTALL)
        code_block = matches[0].strip() if matches else ""
        
        result = self.execute_code(code_block) if code_block else "No valid code generated."

        return result, self.token_usage, self.remaining_tokens

@app.route('/', methods=['GET', 'POST'])
def index():
    agent = ExamAgent()
    
    if request.method == 'POST':
        user_input = request.form.get('task')
        files = request.files.getlist('files')
        
        try:
            result, used, remaining = agent.process_task(user_input, files)
            return render_template('result.html', 
                result=result,
                used=used,
                remaining=remaining
            )
        except Exception as e:
            logging.error(f"Processing error: {e}")
            return render_template('error.html', error=str(e))
    
    return render_template('index.html')

@app.route('/api/tokens', methods=['GET'])
def token_status():
    agent = ExamAgent()
    return jsonify({"used_tokens": agent.token_usage, "remaining_tokens": agent.remaining_tokens})

if __name__ == '__main__':
    app.run(debug=True)
