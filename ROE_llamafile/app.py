from flask import Flask, request, jsonify, render_template
import sys
import os

# Add the project directory to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from ai_agent import AIAgent
except ImportError:
    raise ImportError("The module 'ai_agent' could not be found. Ensure that 'ai_agent.py' exists in the same directory as 'app.py'.")

app = Flask(__name__)
agent = AIAgent()

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Ensure the uploads directory exists
        uploads_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "uploads")
        print(f"Uploads directory: {uploads_dir}")  # Debugging: Print the uploads directory path
        try:
            os.makedirs(uploads_dir, exist_ok=True)
            print(f"Directory created: {uploads_dir}")  # Debugging: Confirm directory creation
        except Exception as e:
            print(f"Error creating directory: {e}")  # Debugging: Print any errors

        # Get user inputs
        question = request.form.get("question")
        files = request.files.getlist("files")
        constraints = request.form.get("constraints")

        # Save uploaded files
        file_paths = []
        for file in files:
            file_path = os.path.join(uploads_dir, file.filename)
            print(f"Saving file to: {file_path}")  # Debugging: Print the file path
            try:
                file.save(file_path)
                file_paths.append(file_path)
            except Exception as e:
                print(f"Error saving file: {e}")  # Debugging: Print any errors

        # Solve the question
        result = agent.solve_question(question, file_paths, constraints)
        return render_template("index.html", result=result)

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)