""" This file binary_rank.py contains functions for converting a DataFrame of rank items into a DataFrame of binary comparison tasks.

The binary comparison tasks are used for the crowd-kit library for aggregating results from crowd workers.

Copyright 2023 Andrew Hundt
"""
import numpy as np
import pandas as pd
from itertools import combinations
import csv
from tqdm import tqdm

# Define a function to process a single group of images
def process_group(group):

    # Define key columns
    key_columns = ["Item Title Index", "Item Title", "Item Type", "Country", 
                   "Source CSV File", "Source CSV Row Index", "Input.prompt", "Input.seed", "HITId", "WorkerId"]
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
        # check if the HITId and WorkerID is the same for both images, and print a warning if not
        if group['HITId'].values[i] != group['HITId'].values[j]:
            print(f'Warning: HITId is not the same for both images: {left_image} and {right_image}')
        if group['WorkerId'].values[i] != group['WorkerId'].values[j]:
            print(f'Warning: WorkerId is not the same for both images: {left_image} and {right_image}')

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
    key_columns = ["Item Title Index", "Item Title", "Item Type", "Country", 
                   "Source CSV File", "Source CSV Row Index", "Input.prompt", "Input.seed", "HITId", "WorkerId"]

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
    results = list(tqdm(pool.imap_unordered(process_group, [group for _, group in rank_df.groupby(['Source CSV File', 'Source CSV Row Index', 'Item Title Index', 'HITId'])]), desc="Creating binary rank table", total=len(rank_df.groupby(['Item Title Index', 'HITId']))))

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
    key_columns = ["Item Title Index", "Item Title", "Item Type", "Country", 
                   "Source CSV File", "Source CSV Row Index", "Input.prompt", "Input.seed", "HITId", "WorkerId"]

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
    for _, group in tqdm(rank_df.groupby(['Source CSV File', 'Source CSV Row Index', 'Item Title Index', 'HITId']), desc="Creating binary rank table"):
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


def convert_table_to_crowdkit_format(
        binary_rank_df, 
        task_columns=['Left Binary Rank Image', 'Right Binary Rank Image', 'Left Neural Network Model', 'Right Neural Network Model', 'Item Title Index', 'Item Title', 'Item Type', 'Country', 'Input.prompt', 'Input.seed'],
        worker_column='WorkerId',
        label_column='Binary Rank Response Left Image is Greater',
        separator='|'):
    """ Simplify the binary rank table by grouping by the specified columns and concatenating the entries into a single string.

    See restore_from_crowdkit_format() for restoring the table.

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

    # # Group by the specified columns
    # grouped = binary_rank_df.groupby(task_columns)

    # # Aggregate the columns into a single string
    # simplified_table = grouped.agg({
    #     worker_column: 'first',
    #     label_column: 'first'
    # }).reset_index()
    # # save out worker and label simplified table
    # simplified_table.to_csv('simplified_binary_rank_table_worker_and_label.csv', index=False, quoting=csv.QUOTE_ALL)

    column_titles = {
        'task': task_columns,
        'worker': worker_column,
        'label': label_column
    }
    
    # concatenate all the task columns into a single column
    st2 = pd.DataFrame()
    st2['task'] = binary_rank_df[task_columns].apply(lambda row: separator.join(row), axis=1)
    st2['worker'] = binary_rank_df[worker_column]
    st2['label'] = binary_rank_df[label_column]

    # get the reformatting variables
    task_columns = '|'.join(task_columns)
    # make a map from the task column to integer ids
    task_to_id = {task: i for i, task in enumerate(st2['task'].unique())}
    # make a map from the worker column to integer ids
    worker_to_id = {worker: i for i, worker in enumerate(st2['worker'].unique())}
    # make a map from the label column to integer ids
    label_to_id = {label: i for i, label in enumerate(st2['label'].unique())}

    table_restore_metadata = {
        'column_titles': column_titles,
        'tasks_mapping': task_to_id,
        'workers_mapping': worker_to_id,
        'labels_mapping': label_to_id,
        'task_columns': task_columns,
        'separator': separator,
        'n_tasks': len(task_to_id),
        'n_workers': len(worker_to_id),
        'n_labels': len(label_to_id)
    }

    # return the simplified int table, the maps, and the column names
    return st2, table_restore_metadata


def restore_from_crowdkit_format(crowdkit_df, table_restore_metadata):
    """ Restore the binary rank table from the simplified table or a results table.
    """
    # If crowdkit_df is a Series, convert it to a DataFrame
    if isinstance(crowdkit_df, pd.Series):
        crowdkit_df = crowdkit_df.to_frame()
        crowdkit_df['task'] = crowdkit_df.index

    # Check if the 'task' column exists
    print(f"step 1 of restoring from crowdkit format crowdkit_df.columns: {crowdkit_df.columns}")

    # restore the task column from integer ids to strings
    st2 = pd.DataFrame()
    # Check if the columns exist and the first value is an integer, then perform the mapping
    # Typical columns include: ['task', 'worker', 'label'] or ['agg_label']
    for column in crowdkit_df.columns:
        if column in crowdkit_df:
            first_value = pd.to_numeric(crowdkit_df[column].iloc[0], errors='coerce')
            if not np.isnan(first_value) and np.issubdtype(first_value, np.integer) and column in table_restore_metadata['column_titles']:
                st2[column] = crowdkit_df[column].map({v: k for k, v in table_restore_metadata[column].items()})
            else:
                st2[column] = crowdkit_df[column]

    print(f"step 2 of restoring from crowdkit format crowdkit_df.columns: {crowdkit_df.columns}, st2.columns: {st2.columns}")
    # split the task column into the original columns
    for i, col in enumerate(table_restore_metadata['column_titles']['task']):
        st2[col] = st2['task'].apply(lambda x: x.split(table_restore_metadata['separator'])[i])

    # drop the task column
    st2 = st2.drop(columns=['task'])

    # return the restored table
    return st2


def reconstruct_ranking(binary_rank_df, group_by, left_category_column='Left Neural Network Model', right_category_column='Right Neural Network Model', aggregate_response_column='agg_label'):
    """
    Reconstruct the ranking of the categories according to the specified grouping.

    Parameters:
        binary_rank_df (pandas.DataFrame): The DataFrame containing the binary rank tasks and responses.
        group_by (list): The list of columns to group by. The function will calculate the ranking within each group separately.
        left_category_column (str): The name of the column containing the left category. Default is 'Left Neural Network Model'.
        right_category_column (str): The name of the column containing the right category. Default is 'Right Neural Network Model'.

    Returns:
        pandas.DataFrame: The DataFrame containing the reconstructed ranking of the categories for each group.
    """

    # Initialize the new DataFrame for the reconstructed ranking
    rank_df = pd.DataFrame(columns=group_by + ['Category', 'Rank'])

    # Iterate through each group of binary rank tasks
    for group_key, group in binary_rank_df.groupby(group_by):
        # Get the list of categories
        categories = group[left_category_column].unique()

        # Create a dictionary to store the number of wins for each category
        wins = {category: 0 for category in categories}

        # Iterate through each binary rank task in the group
        for _, row in group.iterrows():
            # Get the left and right categories and the binary response
            left_category = row[left_category_column]
            right_category = row[right_category_column]
            binary_response = row[aggregate_response_column]

            # Update the number of wins for the winning category
            if binary_response is True:
                wins[left_category] += 1
            elif binary_response is False:
                wins[right_category] += 1

        # Sort the categories by the number of wins in descending order
        sorted_categories = sorted(categories, key=lambda x: wins[x], reverse=True)

        # Assign the rank to each category based on the sorted order
        ranks = {category: i + 1 for i, category in enumerate(sorted_categories)}

        # Construct the new rows for the rank DataFrame
        new_rows = [{**dict(zip(group_by, group_key)), 'Category': category, 'Rank': ranks[category]} for category in categories]

        # Append the new rows to the rank DataFrame
        rank_df = rank_df.append(new_rows, ignore_index=True)

    return rank_df


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
        'WorkerId': ['Worker1', 'Worker1', 'Worker2', 'Worker2', 'Worker3', 'Worker3'],
        'Source CSV File': ['data.csv', 'data.csv', 'data.csv', 'data.csv', 'data.csv', 'data.csv'],
        'Source CSV Row Index': [1, 1, 2, 2, 3, 3]
    }

    df = pd.DataFrame(data)

    network_models = ["baseline", "contrastive"]

    # Call the old and new functions
    new_df = binary_rank_table(df, network_models)
    old_df = binary_rank_table_old(df, network_models)

    # Compare the outputs
    functions_match = old_df.equals(new_df)
    print(f"Parallel and Single Threaded Function outputs match: {functions_match}")

    # If the results are different, print the differences
    if not functions_match:
        print("Bug Detcted! The Parallel and Single Threaded code differs!")
        # check each column name and print those that differ
        for i, col_name in enumerate(new_df.columns):
            if col_name != old_df.columns[i]:
                print(f"Column {i} name differs: {col_name} != {old_df.columns[i]}")
        # print the old df
        new_s = new_df.to_string()
        old_s = old_df.to_string()
        # check if old string and new string are equal
        print(f"New Dataframe String Equals Old Dataframe String: \n{new_s==old_s}")
        print(f"New DataFrame:\n{new_s}")
        print(f"Old DataFrame:\n{old_s}")
        # Compare the dataframes
        diff_df = old_df.compare(new_df)

        # Iterate over the rows
        for row in diff_df.iterrows():
            index, data = row

            # Iterate over the columns
            for column in data.iteritems():
                column_name, diff = column

                # If the cell is not NaN, print the difference
                if pd.notna(diff[0]) or pd.notna(diff[1]):
                    print(f"Row {index}, Column {column_name} - old value: {diff[0]}, new value: {diff[1]}")



    # Save the binary rank results to a CSV file
    old_df.to_csv('binary_rank_results_example_old.csv', index=False)
    new_df.to_csv('binary_rank_results_example_new.csv', index=False)

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
        'Binary Rank Response Left Image is Greater': [True, False, True],
        'Source CSV File': ['data.csv', 'data.csv', 'data.csv'],
        'Source CSV Row Index': [1, 1, 1]
    }

    binary_rank_df = pd.DataFrame(data)

    # Simplify the binary rank table
    crowdkit_table, table_restore_metadata = convert_table_to_crowdkit_format(binary_rank_df)

    # Save the simplified table to a CSV file with quotes around entries
    crowdkit_table.to_csv('simplified_binary_rank_example_table_int.csv', index=False, quoting=csv.QUOTE_ALL)
    
    # Restore the simplified table
    st2 = restore_from_crowdkit_format(crowdkit_table, table_restore_metadata)
    # Save the restored table to a CSV file with quotes around entries
    st2.to_csv('simplified_restored_binary_rank_example_table.csv', index=False, quoting=csv.QUOTE_ALL)
