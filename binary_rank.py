""" This file binary_rank.py contains functions for converting a DataFrame of rank items into a DataFrame of binary comparison tasks.

The binary comparison tasks are used for the crowd-kit library for aggregating results from crowd workers.

Copyright 2023 Andrew Hundt
"""
import pandas as pd
from itertools import combinations
import csv
from tqdm import tqdm

# Define a function to process a single group of images
def process_group(group):

    # Define key columns
    key_columns = ["Item Title Index", "Item Title", "Item Type",
                   "Country", "Source CSV Row Index", "Input.prompt", "Input.seed", "HITId", "WorkerId"]
    # Get the list of images and sort them
    images = sorted(group['Image File Path'].to_list())

    # Get the list of responses and network models
    responses = group['Response'].to_list()
    network_models = group['Neural Network Model'].to_list()

    # Get the list of image shuffle indices
    shuffle_indices = group['Image Shuffle Index'].to_list()

    # Create a list to store the new rows
    new_rows = []

    # Iterate through unique pairs of images for binary comparison
    for i, j in combinations(range(len(images)), 2):
        left_image = images[i]
        right_image = images[j]

        left_response = responses[i]
        right_response = responses[j]

        left_nn = network_models[i]
        right_nn = network_models[j]

        # Handle None values
        binary_response = None if None in (left_response, right_response) else left_response < right_response

        # Construct the new row for the binary rank DataFrame
        row_data = {
            'Left Binary Rank Image': left_image,
            'Right Binary Rank Image': right_image,
            'Left Neural Network Model': left_nn,
            'Right Neural Network Model': right_nn,
            'Left Image Shuffle Index': shuffle_indices[i],
            'Right Image Shuffle Index': shuffle_indices[j],
            'Binary Rank Response Left Image is Greater': binary_response
        }

        # Add key columns to the row
        for col in key_columns:
            row_data[col] = group[col].values[0]

        # Append the new row to the list
        new_rows.append(row_data)

    # Return the list of new rows
    return new_rows


def binary_rank_table(df, network_models):
    """
    Break down rank items into binary image comparison tasks and create a new DataFrame.

    Parameters:
        df (pandas.DataFrame): The DataFrame containing survey responses.
        network_models (list): List of network models used for evaluation.

    Returns:
        pandas.DataFrame: The new DataFrame with binary comparison tasks.
    """

    # Filter the DataFrame for "Rank" items and relevant network models
    rank_df = df[(df['Item Type'] == "Rank") & df['Neural Network Model'].isin(network_models)]

    # Define key columns
    key_columns = ["Item Title Index", "Item Title", "Item Type",
                   "Country", "Source CSV Row Index", "Input.prompt", "Input.seed", "HITId", "WorkerId"]

    # Define binary key columns for left and right sides
    binary_key_columns = ["Neural Network Model", "Image Shuffle Index"]
    left_key_columns = [f'Left {col}' if col in binary_key_columns else col for col in key_columns]
    right_key_columns = [f'Right {col}' if col in binary_key_columns else col for col in key_columns]

    # Initialize the new DataFrame for binary rank tasks
    binary_rank_df = pd.DataFrame(columns=[
        'Left Binary Rank Image', 'Right Binary Rank Image',
        'Left Neural Network Model', 'Right Neural Network Model',
        'Left Image Shuffle Index', 'Right Image Shuffle Index',
        'Binary Rank Response Left Image is Greater'] + key_columns)

    # Use multiprocessing to parallelize the processing of groups
    import multiprocessing as mp
    pool = mp.Pool(mp.cpu_count())

    # Apply the process_group function to each group and get the results
    # Use imap_unordered and tqdm to show the progress
    results = list(tqdm(pool.imap_unordered(process_group, [group for _, group in rank_df.groupby(['Item Title Index', 'HITId'])]), desc="Creating binary rank table", total=len(rank_df.groupby(['Item Title Index', 'HITId']))))

    # Close the pool and join the processes
    pool.close()
    pool.join()

    # Flatten the results into a single list of rows
    rows = [row for result in results for row in result]

    # Append the rows to the binary rank DataFrame
    binary_rank_df = pd.concat([binary_rank_df, pd.DataFrame(rows)], ignore_index=True)

    return binary_rank_df


def binary_rank_table_old(df, network_models):
    """
    Break down rank items into binary image comparison tasks and create a new DataFrame.

    Parameters:
        df (pandas.DataFrame): The DataFrame containing survey responses.
        network_models (list): List of network models used for evaluation.

    Returns:
        pandas.DataFrame: The new DataFrame with binary comparison tasks.
    """

    # Filter the DataFrame for "Rank" items and relevant network models
    rank_df = df[(df['Item Type'] == "Rank") & df['Neural Network Model'].isin(network_models)]

    # Define key columns
    key_columns = ["Item Title Index", "Item Title", "Item Type",
                   "Country", "Source CSV Row Index", "Input.prompt", "Input.seed", "HITId", "WorkerId"]

    # Define binary key columns for left and right sides
    binary_key_columns = ["Neural Network Model", "Image Shuffle Index"]
    left_key_columns = [f'Left {col}' if col in binary_key_columns else col for col in key_columns]
    right_key_columns = [f'Right {col}' if col in binary_key_columns else col for col in key_columns]

    # Initialize the new DataFrame for binary rank tasks
    binary_rank_df = pd.DataFrame(columns=[
        'Left Binary Rank Image', 'Right Binary Rank Image',
        'Left Neural Network Model', 'Right Neural Network Model',
        'Left Image Shuffle Index', 'Right Image Shuffle Index',
        'Binary Rank Response Left Image is Greater'] + key_columns)

    # Iterate through unique pairs of images for binary comparison
    for _, group in tqdm(rank_df.groupby(['Item Title Index', 'HITId']), desc="Creating binary rank table"):
        image_combinations = list(combinations(sorted(group['Image File Path'].to_list()), 2))

        for left_image, right_image in image_combinations:
            left_group = group[group['Image File Path'] == left_image]
            right_group = group[group['Image File Path'] == right_image]

            left_response = left_group['Response'].values[0]
            right_response = right_group['Response'].values[0]

            left_nn = left_group['Neural Network Model'].values[0]
            right_nn = right_group['Neural Network Model'].values[0]

            # TODO(ahundt) check if the less than comparison below is correct, and if it should be flipped. Handle None values in analysis.
            # Handle None values
            binary_response = None if None in (left_response, right_response) else left_response < right_response

            # Construct the new row for the binary rank DataFrame
            row_data = {
                'Left Binary Rank Image': left_image,
                'Right Binary Rank Image': right_image,
                'Left Neural Network Model': left_nn,
                'Right Neural Network Model': right_nn,
                'Left Image Shuffle Index': left_group['Image Shuffle Index'].values[0],
                'Right Image Shuffle Index': right_group['Image Shuffle Index'].values[0],
                'Binary Rank Response Left Image is Greater': binary_response
            }

            # Add key columns to the row
            for col in key_columns:
                row_data[col] = group[col].values[0]

            # Append the new row to the binary rank DataFrame
            binary_rank_df = pd.concat([binary_rank_df, pd.DataFrame([row_data])], ignore_index=True)

    return binary_rank_df


def simplify_binary_rank_table(
        binary_rank_df, 
        task_columns=['Left Binary Rank Image', 'Right Binary Rank Image', 'Left Neural Network Model', 'Right Neural Network Model', 'Item Title Index', 'Item Title', 'Item Type', 'Country', 'Input.prompt', 'Input.seed'],
        worker_column='WorkerId',
        label_column='Binary Rank Response Left Image is Greater',
        separator='|'):
    """ Simplify the binary rank table by grouping by the specified columns and concatenating the entries into a single string.

    The purpose of this function is to convert the binary rank table into a format that can be used by the
    crowd-kit library. The crowd-kit library requires the table to be in a specific format, which is described
    in the documentation.

    For example, https://toloka.ai/docs/crowd-kit/reference/crowdkit.aggregation.classification.m_msr.MMSR.fit_predict_score/

    Here is an example of the table format:
    task	worker	label
    0	0	0
    0	1	1
    0	2	0

    Parameters:

        binary_rank_df (pandas.DataFrame): The DataFrame containing binary rank responses.
        task_columns (list): List of column names to group by, and concatenate into a single string.
        worker_column (str): Name of the column containing the worker ids.
        label_column (str): Name of the column containing the labels.
    
    Returns:

        The returns include the simplified DataFrame, and maps from the task, worker, and label columns to integer ids for restoring the table.
            pandas.DataFrame: The simplified DataFrame with concatenated entries.
            dict: A map from the task column to integer ids.
            dict: A map from the worker column to integer ids.
            dict: A map from the label column to integer ids.
            list: A list of column names.
    """
    # convert every entry to a string
    binary_rank_df = binary_rank_df.astype(str)

    # Group by the specified columns
    grouped = binary_rank_df.groupby(task_columns)

    # Aggregate the columns into a single string
    simplified_table = grouped.agg({
        worker_column: 'first',
        label_column: 'first'
    }).reset_index()

    # double for loop to concatenate the titles and values of the combined columns into a single string per row
    st2 = pd.DataFrame()
    st2['task'] = simplified_table[task_columns[0]]
    for col in task_columns[1:]:
        st2['task'] = st2['task'] + separator + simplified_table[col]
    st2['worker'] = simplified_table[worker_column]
    st2['label'] = simplified_table[label_column]

    # st2.to_csv('simplified_binary_rank_table.csv', index=False, quoting=csv.QUOTE_ALL)

    task_columns = '|'.join(task_columns)
    # make a map from the task column to integer ids
    task_to_id = {task: i for i, task in enumerate(st2['task'].unique())}
    # make a map from the worker column to integer ids
    worker_to_id = {worker: i for i, worker in enumerate(st2['worker'].unique())}
    # make a map from the label column to integer ids
    label_to_id = {label: i for i, label in enumerate(st2['label'].unique())}

    # create st2_int with integer ids for all columns using the maps above
    st2_int = pd.DataFrame()
    st2_int['task'] = st2['task'].map(task_to_id)
    st2_int['worker'] = st2['worker'].map(worker_to_id)
    st2_int['label'] = st2['label'].map(label_to_id)

    # store the column names as a list
    column_titles = [task_columns, worker_column, label_column]

    # return the simplified int table, the maps, and the column names
    return st2_int, task_to_id, worker_to_id, label_to_id, column_titles


def restore_binary_rank_table(st2_int, task_to_id, worker_to_id, label_to_id, column_titles, separator='|'):
    """ Restore the binary rank table from the simplified table.
    """
    # restore the task column from integer ids to strings
    st2 = pd.DataFrame()
    st2['task'] = st2_int['task'].map({v: k for k, v in task_to_id.items()})
    st2['worker'] = st2_int['worker'].map({v: k for k, v in worker_to_id.items()})
    st2['label'] = st2_int['label'].map({v: k for k, v in label_to_id.items()})

    # split the task column into the original columns
    combined_columns = column_titles[0].split(separator)
    for i, col in enumerate(combined_columns):
        st2[col] = st2['task'].apply(lambda x: x.split('|')[i])

    # drop the task column
    st2 = st2.drop(columns=['task'])

    # return the restored table
    return st2


if __name__ == "__main__":
    # Example usage with your provided data
    data = {
        'Item Title Index': [1, 1, 2, 2, 3, 3],
        'Item Title': ['Title1', 'Title1', 'Title2', 'Title2', 'Title3', 'Title3'],
        'Item Type': ['Rank', 'Rank', 'Rank', 'Rank', 'Rank', 'Rank'],
        'Neural Network Model': ['baseline', 'contrastive', 'baseline', 'contrastive', 'baseline', 'contrastive'],
        'Image File Path': ['img1.png', 'img2.png', 'img1.png', 'img2.png', 'img1.png', 'img2.png'],
        'Image Shuffle Index': [1, 2, 1, 2, 1, 2],  # Add this line
        'Response': [2, 1, 3, 4, 2, 3],
        'HITId': ['hit1', 'hit1', 'hit2', 'hit2', 'hit3', 'hit3'],
        'Country': ['Country1', 'Country1', 'Country2', 'Country2', 'Country3', 'Country3'],
        'Source CSV Row Index': [1, 2, 3, 4, 5, 6],
        'Input.prompt': ['Prompt1', 'Prompt1', 'Prompt2', 'Prompt2', 'Prompt3', 'Prompt3'],
        'Input.seed': [123, 123, 456, 456, 789, 789],
        'WorkerId': ['Worker1', 'Worker1', 'Worker2', 'Worker2', 'Worker3', 'Worker3']
    }

    df = pd.DataFrame(data)

    network_models = ["baseline", "contrastive"]

    # Call the old and new functions
    new_df = binary_rank_table(df, network_models)
    old_df = binary_rank_table_old(df, network_models)

    # Compare the outputs
    print(old_df.equals(new_df))

    # Save the binary rank results to a CSV file
    old_df.to_csv('binary_rank_results_old.csv', index=False)
    new_df.to_csv('binary_rank_results_new.csv', index=False)

    #########################################################
    # Example usage with your provided data
    data = {
        'Left Binary Rank Image': ['img1.png', 'img1.png', 'img2.png'],
        'Right Binary Rank Image': ['img2.png', 'img2.png', 'img3.png'],
        'Left Neural Network Model': ['baseline', 'baseline', 'baseline'],
        'Right Neural Network Model': ['contrastive', 'contrastive', 'contrastive'],
        'Item Title Index': [1, 2, 3],
        'Item Title': ['Title1', 'Title2', 'Title3'],
        'Item Type': ['Rank', 'Rank', 'Rank'],
        'Country': ['Country1', 'Country2', 'Country3'],
        'Input.prompt': ['Prompt1', 'Prompt2', 'Prompt3'],
        'Input.seed': [123, 456, 789],
        'WorkerId': ['Worker1', 'Worker2', 'Worker3'],
        'Binary Rank Response Left Image is Greater': [True, False, True]
    }

    binary_rank_df = pd.DataFrame(data)

    # Simplify the binary rank table
    st2_int, task_to_id, worker_to_id, label_to_id, column_titles = simplify_binary_rank_table(binary_rank_df)

    # Save the simplified table to a CSV file with quotes around entries
    st2_int.to_csv('simplified_binary_rank_table_int.csv', index=False, quoting=csv.QUOTE_ALL)
    
    # Restore the simplified table
    st2 = restore_binary_rank_table(st2_int, task_to_id, worker_to_id, label_to_id, column_titles)
    # Save the restored table to a CSV file with quotes around entries
    st2.to_csv('simplified_restored_binary_rank_table.csv', index=False, quoting=csv.QUOTE_ALL)
