import os
import pandas as pd
import sqlite3
import duckdb
import zipfile
import tempfile
from bs4 import BeautifulSoup
from tabula import read_pdf
from datetime import datetime
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

dataframes_dict = {}

# Provide the input path here: Either a folder path or a ZIP file path
input_path = "C:\\Users\\user\\Downloads\\mock_roe_4.zip"  # <-- Change this to your folder or ZIP file

def extract_text_from_html_xml(file_path, file_type):
    parser_type = "html.parser" if file_type == "html" else "xml"
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            soup = BeautifulSoup(f, parser_type)
        return pd.DataFrame([{"filename": os.path.basename(file_path), "content": soup.get_text(separator=" ", strip=True)}])
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None

def extract_tables_from_pdf(file_path):
    try:
        tables = read_pdf(file_path, pages="all", multiple_tables=True, silent=True)
        return [table.assign(filename=os.path.basename(file_path), table_number=i + 1) for i, table in enumerate(tables)]
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return []

def read_file(file_path, file_type):
    try:
        if file_type == "csv":
            df = pd.read_csv(file_path)
        elif file_type == "json":
            df = pd.read_json(file_path)
        elif file_type == "parquet":
            df = pd.read_parquet(file_path)
        elif file_type == "md":
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
            df = pd.DataFrame([{"filename": os.path.basename(file_path), "content": content}])
        else:
            return None
        df["filename"] = os.path.basename(file_path)
        return df
    except Exception as e:
        print(f"Error reading {file_type.upper()} {file_path}: {e}")
        return None

def read_sqlite_db(file_path):
    try:
        conn = sqlite3.connect(file_path)
        tables = pd.read_sql("SELECT name FROM sqlite_master WHERE type='table';", conn)["name"].tolist()
        for table in tables:
            dataframes_dict[f"sqlite_{table}"] = pd.read_sql(f"SELECT * FROM {table};", conn)
        conn.close()
    except Exception as e:
        print(f"Error reading SQLite database {file_path}: {e}")

def read_duckdb_file(file_path):
    try:
        conn = duckdb.connect(database=file_path, read_only=True)
        tables = conn.execute("SHOW TABLES").fetchall()
        for table in tables:
            table_name = table[0]
            dataframes_dict[f"duckdb_{table_name}"] = conn.execute(f"SELECT * FROM {table_name}").fetchdf()
        conn.close()
    except Exception as e:
        print(f"Error reading DuckDB database {file_path}: {e}")

def read_excel_file(file_path):
    try:
        sheets = pd.read_excel(file_path, sheet_name=None)
        for sheet_name, df in sheets.items():
            df["filename"] = os.path.basename(file_path)
            dataframes_dict[f"excel_{sheet_name}"] = df
    except Exception as e:
        print(f"Error reading Excel {file_path}: {e}")

def process_files(directory):
    html_data, xml_data, pdf_data, csv_data, json_data, md_data, parquet_data = [], [], [], [], [], [], []
    for file in os.listdir(directory):
        file_path = os.path.join(directory, file)
        if file.endswith(".html"):
            df = extract_text_from_html_xml(file_path, "html")
            if df is not None:
                html_data.append(df)
        elif file.endswith(".xml"):
            df = extract_text_from_html_xml(file_path, "xml")
            if df is not None:
                xml_data.append(df)
        elif file.endswith(".pdf"):
            tables = extract_tables_from_pdf(file_path)
            pdf_data.extend(tables)
        elif file.endswith((".csv", ".json", ".parquet", ".md")):
            file_type = file.split(".")[-1]
            df = read_file(file_path, file_type)
            if df is not None:
                eval(f"{file_type}_data").append(df)
        elif file.endswith(".db") or file.endswith(".sqlite"):
            read_sqlite_db(file_path)
        elif file.endswith(".duckdb"):
            read_duckdb_file(file_path)
        elif file.endswith((".xlsx", ".xls")):
            read_excel_file(file_path)
    for dtype, data in [("html", html_data), ("xml", xml_data), ("pdf", pdf_data), ("csv", csv_data),
                        ("json", json_data), ("parquet", parquet_data), ("markdown", md_data)]:
        if data:
            dataframes_dict[dtype] = pd.concat(data, ignore_index=True)

def process_input(input_path):
    if input_path.endswith(".zip"):
        with zipfile.ZipFile(input_path, "r") as zip_ref:
            temp_dir = tempfile.mkdtemp()
            zip_ref.extractall(temp_dir)
            print(f"Extracted ZIP to temporary folder: {temp_dir}")
            process_files(temp_dir)
    elif os.path.isdir(input_path):
        process_files(input_path)
    else:
        print("Invalid input: Please provide a valid folder path or ZIP file.")

def get_dataframes():
    return dataframes_dict

# Process the provided input
process_input(input_path)