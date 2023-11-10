import argparse
import os
import pandas as pd
from flask import Flask, render_template
from jinja2 import Template
import re

app = Flask(__name__)

@app.route('/')
def index():
    global current_row, current_file, df, html_template

    # Check if all rows are processed for the current file
    if current_row >= len(df):
        current_row = 0
        current_file += 1

        # If all files are processed, stop the server
        if current_file >= len(html_files):
            print("All files processed. Exiting...")
            os._exit(0)

        load_data(html_files[current_file], csv_files[current_file])

    # Perform text substitution based on the current row
    template_content = re.sub(r'\${(.*?)}', lambda match: str(df.iloc[current_row][match.group(1)]), html_template)

    current_row += 1

    return template_content

def load_data(html_file, csv_file):
    global df, html_template

    # Load CSV and HTML files
    df = pd.read_csv(csv_file)
    with open(html_file, 'r') as f:
        html_template = f.read()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Dynamic Text Substitution for HTML Files')
    parser.add_argument('--folder', type=str, default="output_surveys", help='Folder containing HTML and CSV files')
    args = parser.parse_args()

    html_files = [os.path.join(args.folder, file) for file in os.listdir(args.folder) if file.endswith('.html')]
    csv_files = [os.path.join(args.folder, file) for file in os.listdir(args.folder) if file.endswith('.csv')]

    if not html_files or not csv_files or len(html_files) != len(csv_files):
        print("Error: Make sure there are equal numbers of HTML and CSV files in the specified folder.")
        os._exit(1)

    current_file = 0
    current_row = 0
    df = pd.DataFrame()
    html_template = None

    load_data(html_files[current_file], csv_files[current_file])

    # Start the Flask app
    app.run(debug=True)
