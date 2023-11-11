import argparse
import os
import pandas as pd
from flask import Flask
import re

app = Flask(__name__)

# Declare global variables
html_files = []
csv_files = []
current_file = 0
current_row = 0
df = None
html_template = None

def load_data(html_file, csv_file):
    """
    Load CSV and HTML files.

    Args:
        html_file (str): Path to the HTML file.
        csv_file (str): Path to the CSV file.

    Returns:
        DataFrame: Loaded DataFrame from the CSV file.
        str: Loaded HTML template.
    """
    loaded_df = pd.read_csv(csv_file)
    with open(html_file, 'r') as f:
        html_template = f.read()
    return loaded_df, html_template

def process_template(html_files, csv_files, current_file, current_row, df, html_template):
    """
    Process the HTML template with dynamic text substitution.

    Args:
        html_files (list): List of HTML file paths.
        csv_files (list): List of CSV file paths.
        current_file (int): Index of the current HTML file.
        current_row (int): Index of the current row in the CSV file.
        df (DataFrame): DataFrame containing data from the CSV file.
        html_template (str): HTML template as a string.

    Returns:
        str: Processed HTML template content.
        int: Updated current_file index.
        int: Updated current_row index.
        DataFrame: Updated DataFrame.
        str: Updated HTML template.
    """
    if current_row >= len(df):
        current_row = 0
        current_file += 1
        if current_file >= len(html_files):
            print("All files processed. Exiting...")
            os._exit(0)
        loaded_df, html_template = load_data(html_files[current_file], csv_files[current_file])
        df = loaded_df  # Assign the loaded DataFrame to the global variable

    # Perform text substitution based on the current row
    template_content = re.sub(r'\${(.*?)}', lambda match: str(df.iloc[current_row][match.group(1)]), html_template)

    current_row += 1

    return template_content

@app.route('/')
def index():
    """
    Define the Flask route for dynamic text substitution.
    """
    global current_file, current_row, df, html_template
    template_content = process_template(html_files, csv_files, current_file, current_row, df, html_template)
    return template_content

def main():
    """
    Main function to run the dynamic text substitution Flask app.
    """
    global html_files, csv_files, current_file, current_row, df, html_template
    parser = argparse.ArgumentParser(description='Dynamic Text Substitution for HTML Files')
    parser.add_argument('--folder', type=str, default="output_surveys", help='Folder containing HTML and CSV files')
    parser.add_argument('--shuffled', action='store_true', help='Choose shuffled versions if available')
    args = parser.parse_args()

    file_extension = '_shuffled.csv' if args.shuffled else '.csv'

    # Collect HTML and CSV file paths
    html_files = [os.path.join(args.folder, file) for file in os.listdir(args.folder) if file.endswith('.html')]
    csv_files = [os.path.join(args.folder, file) for file in os.listdir(args.folder) if file.endswith(file_extension)]

    if not args.shuffled:
        # Filter CSV files to include only those with matching prefixes to HTML files
        html_prefixes = set(os.path.splitext(os.path.basename(file))[0] for file in html_files)
        csv_files = [file for file in csv_files if os.path.splitext(os.path.basename(file))[0] in html_prefixes]

    if not html_files or not csv_files or len(html_files) != len(csv_files):
        print("Error: Make sure there are equal numbers of HTML and CSV files in the specified folder.")
        return

    # Initialize variables
    df, html_template = load_data(html_files[current_file], csv_files[current_file])

    # Start the Flask app
    app.run(debug=True)

if __name__ == '__main__':
    main()
