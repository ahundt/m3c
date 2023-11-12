import argparse
import pandas as pd
import json


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


def process_survey_results_csv(csv_file):
    # Load CSV Data Using Pandas
    df = pd.read_csv(csv_file)

    # Apply extract_and_process_task_answers to extract and process ratings data
    ratings_df = df['Answer.taskAnswers'].apply(extract_and_process_task_answers)

    # Concatenate the new DataFrame with the original one
    df = pd.concat([df, ratings_df], axis=1)

    # Call Statistical Analysis Function
    statistical_analysis(df)


def main():
    # Command Line Parameter Parsing
    parser = argparse.ArgumentParser(description="Survey Data Analysis")
    parser.add_argument("--csv_file", type=str, default="Batch_393773_batch_results.csv", help="Path to the input CSV file")
    args = parser.parse_args()

    process_survey_results_csv(args.csv_file)

if __name__ == "__main__":
    main()
