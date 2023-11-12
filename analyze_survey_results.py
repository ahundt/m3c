import argparse
import pandas as pd
import json
import os


def extract_and_process_entry(entry, value):
    if value is None or value == "":
        return None
    if entry.endswith("-rating"):
        return int(value)
    elif entry.endswith("-checkbox"):
        return bool(value['1'])
    elif entry.endswith("-short-answer"):
        return value
    else:
        raise ValueError(f"Unexpected format in taskAnswers: {entry}")


def extract_and_process_task_answers(task_answers):
    task_answers_dict = json.loads(task_answers)
    ratings_data = {}

    for entry, value in task_answers_dict[0].items():
        ratings_data[entry] = extract_and_process_entry(entry, value)

    return pd.Series(ratings_data)  # Convert the dictionary to a Series


def statistical_analysis(data):
    # Define your statistical analysis here
    # You can use functions from libraries like NumPy and pandas
    print(data)
    data.to_csv("statistical_output.csv")


def process_survey_results_csv(csv_file, survey_items_file):
    # Load CSV Data Using Pandas
    df = pd.read_csv(csv_file)

    # Apply extract_and_process_task_answers to extract and process ratings data
    ratings_df = df['Answer.taskAnswers'].apply(extract_and_process_task_answers)

    # Concatenate the new DataFrame with the original one
    df = pd.concat([df, ratings_df], axis=1)

    # Extract country name from the survey results "Title" column
    # Assuming the last word in the "Title" column is the country name
    df['Title'] = df['Title'].str.strip()  # Remove leading/trailing spaces
    country_name = df['Title'].str.split().str[-1]

    # Get the input image columns like "Input.img1" to find the number of images per item
    image_columns = [col for col in df.columns if col.startswith("Input.img")]
    num_images_per_item = len(image_columns)

    # Determine the number of items for analysis from human_survey_items.csv
    human_survey_items = pd.read_csv(survey_items_file)
    num_items = len(human_survey_items)

    # You can now perform further analysis based on the extracted data
    print(f"Country: {country_name.iloc[0]}")
    print(f"Number of Items: {num_items}")
    print(f"Number of Images per Item: {num_images_per_item}")
    print(df)

    # Call Statistical Analysis Function
    statistical_analysis(df)


def main():
    # Command Line Parameter Parsing
    parser = argparse.ArgumentParser(description="Survey Data Analysis")
    parser.add_argument("--response_results", type=str, default="Batch_393773_batch_results.csv", help="Path to the file or folder containing CSV files with Amazon Mechanical Turk survey response results.")
    parser.add_argument("--survey_items_file", type=str, default="human_survey_items.csv", help="Path to the human_survey_items.csv file")
    args = parser.parse_args()

    if os.path.isfile(args.response_results):
        csv_files = [args.response_results]
    elif os.path.isdir(args.response_results):
        # List all CSV files in the results folder
        csv_files = [os.path.join(args.results_folder, filename) for filename in os.listdir(args.response_results) if filename.endswith(".csv")]

    # Loop through all CSV files and call process_survey_results_csv on each
    for csv_file in csv_files:
        process_survey_results_csv(csv_file, args.survey_items_file)


if __name__ == "__main__":
    main()
