""" This file binary_rank.py contains functions for converting a DataFrame of rank items into a DataFrame of binary comparison tasks.

The binary comparison tasks are used for the crowd-kit library for aggregating results from crowd workers.

Copyright 2023 Andrew Hundt
"""
import numpy as np
import pandas as pd
from itertools import combinations
import csv
from tqdm import tqdm


def binary_rank_one_table_df_group(group):
    """
    Break down one group of ranked survey items into binary image comparison tasks and create a new DataFrame.

    If there are four ranks, then there are six unique pairs of images for binary comparison (n chose 2)
    to fully define the ranking. The binary comparison tasks are used for the crowd-kit library for 
    aggregating results from crowd workers.

    This is a helper function for binary_rank_table().

    Parameters:
        group (pandas.DataFrame): The DataFrame containing survey responses for a single group.

    Returns:
        list: The list of new rows for the binary comparison DataFrame.
    """

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


def binary_rank_table(df, network_models, groupby=['Source CSV File', 'Source CSV Row Index', 'Item Title Index', 'HITId']):
    """
    Break down rank items into binary image comparison tasks and create a new DataFrame.

    Parameters:
        df (pandas.DataFrame): The DataFrame containing survey responses.
        network_models (list): List of network models used for evaluation.
        groupby (list): List of columns to group by. The function will calculate the binary comparison tasks within each group separately.

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

    # Apply the binary_rank_one_table_df_group function to each group and get the results
    # Use imap_unordered and tqdm to show the progress
    results = list(tqdm(pool.imap_unordered(binary_rank_one_table_df_group, [group for _, group in rank_df.groupby(groupby)]), desc="Creating binary rank table", total=len(rank_df.groupby(groupby))))

    # Close the pool and join the processes
    pool.close()
    pool.join()

    # Flatten the results into a single list of rows
    rows = [row for result in results for row in result]

    # Append the rows to the binary rank DataFrame
    binary_rank_df = pd.concat([binary_rank_df, pd.DataFrame(rows)], ignore_index=True)

    return binary_rank_df


def binary_rank_table_single_threaded(df, network_models, groupby=['Source CSV File', 'Source CSV Row Index', 'Item Title Index', 'HITId']):
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
    for _, group in tqdm(rank_df.groupby(groupby), desc="Creating binary rank table"):
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


def convert_table_to_crowdkit_classification_format(
        binary_rank_df, 
        task_columns=['Left Binary Rank Image', 'Right Binary Rank Image', 'Left Neural Network Model', 'Right Neural Network Model', 'Item Title Index', 'Item Title', 'Item Type', 'Country', 'Input.prompt', 'Input.seed'],
        worker_column='WorkerId',
        label_column='Binary Rank Response Left Image is Greater',
        separator='|',
        append_columns=[],
        remap_to_integer_ids=False):
    """ Simplify the binary rank table by grouping by the specified columns and concatenating the entries into a single string in the format expected by the crowd-kit library.

    The purpose of this function is to convert the binary rank table into a format that can be used by the
    crowd-kit library. The crowd-kit library requires the table to be in a specific format, which is described
    in the documentation.

    For example, https://toloka.ai/docs/crowd-kit/reference/crowdkit.aggregation.classification.m_msr.MMSR.fit_predict_score/

    Here is an example of the table format:
    task	worker	label
    0	    0	    0
    0	    1	    1
    0	    2	    0

    
    See restore_from_crowdkit_format() for restoring the table, and the original column names. 
    Restoring the table is valuable for saving and visualizing the result of crowd-kit aggregation for human readable visualization analysis.

    Parameters:

        binary_rank_df (pandas.DataFrame): The DataFrame containing binary rank responses.
        task_columns (list): List of column names to group by, and concatenate into a single string.
        worker_column (str): Name of the column containing the worker ids.
        label_column (str): Name of the column containing the labels.
        separator (str): The separator to use for concatenating the task columns. Defaults to '|'.
        append_columns (list): List of columns to append to the task column. Defaults to [].
        remap_to_integer_ids (bool): Whether to remap the task, worker, and label columns to integer ids. Defaults to False.
            If True, the task, worker, and label columns will be remapped to integer ids.
            Otherwise the task, worker, and label columns will be left as strings.
    
    Returns:

        The returns include the simplified DataFrame, and maps from the task, worker, and label columns to integer ids for restoring the table.
            pandas.DataFrame: The simplified DataFrame with the columns 'task', 'worker', and 'label', where task is a single string containing the concatenated task columns.
            table_restore_metadata (dict): A dictionary containing the maps from the task, worker, and label columns to integer ids for restoring the table to the original format.
    """
    # convert every entry to a string
    binary_rank_df = binary_rank_df.astype(str)

    # get the column titles for restoring the table
    column_titles = {
        'task': task_columns,
        'worker': worker_column,
        'label': label_column
    }
    # add the append columns to the column titles
    for col in append_columns:
        column_titles[col] = col
    # concatenate all the task columns into a single column
    st2 = pd.DataFrame()
    st2['task'] = binary_rank_df[task_columns].apply(lambda row: separator.join(row), axis=1)
    st2['worker'] = binary_rank_df[worker_column]
    st2['label'] = binary_rank_df[label_column]
    # append the additional columns
    st2[append_columns] = binary_rank_df[append_columns]

    # get the reformatting variables
    task_columns = separator.join(task_columns)
    # make a map from the task column to integer ids
    task_list = st2['task'].unique()
    task_to_id = {task: i for i, task in enumerate(task_list)}
    # make a map from the worker column to integer ids
    worker_list = st2['worker'].unique()
    worker_to_id = {worker: i for i, worker in enumerate(worker_list)}
    # make a map from the label column to integer ids
    label_list = st2['label'].unique()
    label_to_id = {label: i for i, label in enumerate(label_list)}

    table_restore_metadata = {
        'column_titles': column_titles,
        'tasks_mapping': task_to_id,
        'workers_mapping': worker_to_id,
        'labels_mapping': label_to_id,
        'task_columns': task_columns,
        'separator': separator,
        'n_tasks': len(task_to_id),
        'n_workers': len(worker_to_id),
        'n_labels': len(label_to_id),
        'workers_list': worker_list,
        'labels_list': label_list,
        'tasks_list': task_list
    }
    table_restore_metadata['append_columns'] = append_columns
    # add the append columns to the table restore metadata
    for col in append_columns:
        col_list = st2[col].unique()
        table_restore_metadata[f'{col}_list'] = col_list
        col_to_id = {col: i for i, col in enumerate(col_list)}
        table_restore_metadata[f'{col}_mapping'] = col_to_id
        table_restore_metadata[f'n_{col}'] = len(col_to_id)

    # remap the task, worker, and label columns to integer ids
    if remap_to_integer_ids:
        st2['task'] = st2['task'].map(task_to_id)
        st2['worker'] = st2['worker'].map(worker_to_id)
        st2['label'] = st2['label'].map(label_to_id)
        for col in append_columns:
            st2[col] = st2[col].map(col_to_id)

    # return the simplified int table, the maps, and the column names
    return st2, table_restore_metadata


def restore_from_crowdkit_classification_format(crowdkit_df, table_restore_metadata):
    """ Restore the binary rank table from the simplified table or a results table.

    Restoring the table is valuable for saving and visualizing the result of crowd-kit aggregation for human readable visualization analysis.

    Parameters:
            
            crowdkit_df (pandas.DataFrame): The DataFrame containing binary rank responses.
            table_restore_metadata (dict): A dictionary containing the maps from the task, worker, and label columns to integer ids for restoring the table to the original format.
    
    Returns:
    
                pandas.DataFrame: The restored DataFrame with the original columns.
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


def convert_table_to_crowdkit_pairwise_format(
        binary_rank_df, 
        worker_column='WorkerId',
        label_column='Binary Rank Response Left Image is Greater',
        pairwise_left_columns=['Left Binary Rank Image', 'Left Neural Network Model'],
        pairwise_right_columns=['Right Binary Rank Image', 'Right Neural Network Model'],
        separator='|',
        left_right_separator='>',
        remap_to_integer_ids=False):
    """ Simplify the binary rank table by grouping by the specified columns and concatenating the entries into a single string in the format expected by the crowd-kit library for pairwise.


    Here is an example of the table format for pairwise:
    worker  left	right	label
    0	    0	    1	    0
    1	    0	    1	    1
    2	    0	    1	    0

    Parameters:

        binary_rank_df (pandas.DataFrame): The DataFrame containing binary rank responses.
        worker_column (str): The name of the column containing the worker ids. Defaults to 'WorkerId'.
        label_column (str): The name of the column containing the labels. Defaults to 'Binary Rank Response Left Image is Greater'.
        pairwise_left_columns (list): The list of columns to use as the left task in the pairwise format. Defaults to ['Left Binary Rank Image', 'Left Neural Network Model'].
        pairwise_right_columns (list): The list of columns to use as the right task in the pairwise format. Defaults to ['Right Binary Rank Image', 'Right Neural Network Model'].
        separator (str): The separator to use for concatenating the task columns. Defaults to '|'.
        left_right_separator (str): The separator to use for concatenating the left and right columns. Defaults to '>'.
        remap_to_integer_ids (bool): Whether to remap the task, worker, and label columns to integer ids. Defaults to False.
            If True, the task, worker, and label columns will be remapped to integer ids.
            Otherwise the task, worker, and label columns will be left as strings.
    
    Returns:

        The returns include the simplified DataFrame, and maps from the task, worker, and label columns to integer ids for restoring the table.
            pandas.DataFrame: The simplified DataFrame with the columns 'worker', 'left', 'right', and 'label', where left and right are single strings containing the concatenated task columns.
            table_restore_metadata (dict): A dictionary containing the maps from the task, worker, and label columns to integer ids for restoring the table to the original format.
    """
    # convert every entry to a string
    binary_rank_df = binary_rank_df.astype(str)

    # use the pairwise left and right columns as the task columns
    st2 = pd.DataFrame()
    st2['worker'] = binary_rank_df[worker_column]
    st2['left'] = binary_rank_df[pairwise_left_columns].apply(lambda row: separator.join(row), axis=1)
    st2['right'] = binary_rank_df[pairwise_right_columns].apply(lambda row: separator.join(row), axis=1)
    st2['label'] = binary_rank_df[label_column]

    # use the label column to insert either the left or right value of a given row into the label column
    # if the label is True or 1, then the left value is greater, otherwise the right value is greater
    # The contents of st2['label'] will be the greater of of st2['left'] or st2['right'].
    st2['label'] = st2.apply(lambda row: row['left'] if row['label'] else row['right'], axis=1)

    # get the column titles for restoring the table
    column_titles = {
        'worker': worker_column,
        'left': pairwise_left_columns,
        'right': pairwise_right_columns,
        'label': label_column
    }

    # get the reformatting variables
    tasks = st2[['left', 'right']].apply(lambda row: left_right_separator.join(row), axis=1)
    task_list = tasks.unique()
    task_to_id = {task: i for i, task in enumerate(task_list)}
    left_list = st2['left'].unique()
    left_to_id = {left: i for i, left in enumerate(left_list)}
    right_list = st2['right'].unique()
    right_to_id = {right: i for i, right in enumerate(right_list)}
    worker_list = st2['worker'].unique()
    worker_to_id = {worker: i for i, worker in enumerate(worker_list)}
    label_list = st2['label'].unique()
    label_to_id = {label: i for i, label in enumerate(label_list)}

    table_restore_metadata = {
        'column_titles': column_titles,
        'tasks_mapping': task_to_id,
        'workers_mapping': worker_to_id,
        'labels_mapping': label_to_id,
        'separator': separator,
        'left_right_separator': left_right_separator,
        'n_tasks': len(task_to_id),
        'n_workers': len(worker_to_id),
        'n_labels': len(label_to_id),
        'workers_list': worker_list,
        'labels_list': label_list,
        'tasks_list': task_list,
        'left_list': left_list,
        'right_list': right_list,
        'left_to_id': left_to_id,
        'right_to_id': right_to_id,
        'n_left': len(left_to_id),
        'n_right': len(right_to_id)
    }

    if remap_to_integer_ids:
        st2['left'] = st2['left'].map(task_to_id)
        st2['right'] = st2['right'].map(task_to_id)
        st2['worker'] = st2['worker'].map(worker_to_id)
        st2['label'] = st2['label'].map(label_to_id)

    return st2, table_restore_metadata


def restore_from_crowdkit_pairwise_format(crowdkit_df, table_restore_metadata):
    """ Restore the binary rank table from the simplified table or a results table for the pairwise format.

    Restoring the table is valuable for saving and visualizing the result of crowd-kit aggregation for human readable visualization analysis.

    Parameters:
            
            crowdkit_df (pandas.DataFrame): The DataFrame containing binary rank responses.
            table_restore_metadata (dict): A dictionary containing the maps from the task, worker, and label columns to integer ids for restoring the table to the original format.
    
    Returns:
    
                pandas.DataFrame: The restored DataFrame with the original columns.
    """
    # If crowdkit_df is a Series, convert it to a DataFrame
    if isinstance(crowdkit_df, pd.Series):
        crowdkit_df = crowdkit_df.to_frame()
        crowdkit_df['label'] = crowdkit_df.index

    # Check if the 'left' and 'right' columns exist
    print(f"step 1 of restoring from crowdkit format crowdkit_df.columns: {crowdkit_df.columns}")

    # restore the left and right columns from integer ids to strings
    st2 = pd.DataFrame()
    # Check if the columns exist and the first value is an integer, then perform the mapping
    # Typical columns include: ['worker', 'left', 'right', 'label'] or ['agg_label', 'agg_score']
    for column in crowdkit_df.columns:
        if column in crowdkit_df:
            first_value = pd.to_numeric(crowdkit_df[column].iloc[0], errors='coerce')
            if not np.isnan(first_value) and np.issubdtype(first_value, np.integer) and column in table_restore_metadata['column_titles']:
                st2[column] = crowdkit_df[column].map({v: k for k, v in table_restore_metadata[column].items()})
            else:
                st2[column] = crowdkit_df[column]

    print(f"step 2 of restoring from crowdkit format crowdkit_df.columns: {crowdkit_df.columns}, st2.columns: {st2.columns}")
    # split the left and right columns into the original columns
    if 'left' in st2.columns:
        for i, col in enumerate(table_restore_metadata['column_titles']['left']):
                st2[col] = st2['left'].apply(lambda x: x.split(table_restore_metadata['separator'])[i])
    if 'right' in st2.columns:
        for i, col in enumerate(table_restore_metadata['column_titles']['right']):
                st2[col] = st2['right'].apply(lambda x: x.split(table_restore_metadata['separator'])[i])

    # drop the left and right columns if they exist
    if 'left' in st2.columns:
        st2 = st2.drop(columns=['left'])
    if 'right' in st2.columns:
        st2 = st2.drop(columns=['right'])

    # return the restored table
    return st2


def reconstruct_ranking(df, group_by=['Item Title', 'Country'], category='Neural Network Model', response='agg_label', rank_method='max'):
    """
    Reconstruct the ranking of the categories according to the specified grouping via a simple majority vote.

    Parameters:
        df (pandas.DataFrame): The DataFrame containing the binary rank tasks and responses.
        group_by (list): The list of columns to group by. The function will calculate the ranking within each group separately.
        category (str): The name of the column containing the category.
        response (str): The name of the column containing the response. The response should be either "True", "False", 1, 0, or None, where 1 means that the left category won the comparison, 0 means that the right category won the comparison, and None means that the comparison was not completed or the response was invalid.
        rank_method (str): The method for assigning the rank. Default is 'max'.

    Returns:
        pandas.DataFrame: The DataFrame containing the reconstructed ranking of the categories for each group, sorted from highest to lowest rank within each group.
    """

    # Create a copy of the DataFrame
    df = df.copy()

    # Search for columns containing the category column string
    category_columns = [col for col in df.columns if category in col]

    # Check if there are exactly two category columns
    if len(category_columns) != 2:
        raise ValueError(f'Expected 2 columns containing {category}, found {len(category_columns)}')

    # Assign the left and right category columns
    left_category_column = category_columns[0]
    right_category_column = category_columns[1]

    # Check if the response column contains any values other than "True", "False", 1, 0, or None
    valid_responses = ["True", "False", 1, 0, None]
    if not df[response].isin(valid_responses).all():
        raise ValueError(f'The response column should contain only {valid_responses}, found {df[response].unique()}')

    # Map "True" to 1 and "False" to 0
    df[response] = df[response].map({True: 1, False: 0, "True": 1, "False": 0, 1:1, 0:0, None:None})

    # Count wins for the left and right models, grouped by the specified columns
    left_wins = df[df[response] == 1].groupby(group_by + [left_category_column]).size().reset_index().rename(columns={left_category_column: category, 0: 'wins'})
    right_wins = df[df[response] == 0].groupby(group_by + [right_category_column]).size().reset_index().rename(columns={right_category_column: category, 0: 'wins'})
    print(f'left_wins: \n{left_wins}')
    print(f'right_wins: \n{right_wins}')

    # Count losses for the left and right models, grouped by the specified columns
    left_losses = df[df[response] == 0].groupby(group_by + [left_category_column]).size().reset_index().rename(columns={left_category_column: category, 0: 'losses'})
    right_losses = df[df[response] == 1].groupby(group_by + [right_category_column]).size().reset_index().rename(columns={right_category_column: category, 0: 'losses'})
    print(f'left_losses: \n{left_losses}')
    print(f'right_losses: \n{right_losses}')

    # Combine wins and losses for all models, ensuring all models are accounted for
    wins = pd.concat([left_wins, right_wins], join='outer', ignore_index=True).groupby(group_by + [category]).sum().reset_index()
    losses = pd.concat([left_losses, right_losses], join='outer', ignore_index=True).groupby(group_by + [category]).sum().reset_index()
    print(f'wins: \n{wins}')
    print(f'losses: \n{losses}')

    # Create a set of all the models in the data
    all_models = set(df[left_category_column].unique()).union(set(df[right_category_column].unique()))

    # Create a set of all the models that have wins in the wins dataframe
    winning_models = set(wins[category].unique())

    # Find the difference between the two sets to get the models that have no wins
    no_winning_models = all_models - winning_models

    # Create a DataFrame with all models
    no_wins = pd.DataFrame({category: list(all_models), 'wins': [0]*len(all_models)})
    if group_by:
        for col in group_by:
            no_wins[col] = df[col].unique()[0]
    # in no_wins drop win rows in the winning_models set
    no_wins = no_wins[~no_wins[category].isin(winning_models)]
    # Assign a wins value of zero to the no_wins dataframe
    no_wins['wins'] = 0

    # drop duplicates in the no_wins dataframe
    no_wins = no_wins.drop_duplicates()

    # Append the no_wins dataframe to the wins dataframe
    wins = pd.concat([wins, no_wins], join='outer', ignore_index=True)

    print(f'no_wins: \n{no_wins}')

    # Create a set of all the models that have losses in the losses dataframe
    losing_models = set(losses[category].unique())

    # Find the difference between the two sets to get the models that have no losses
    no_losing_models = all_models - losing_models

    # Create a DataFrame with all models
    no_losses = pd.DataFrame({category: list(all_models), 'losses': [0]*len(all_models)})
    if group_by:
        for col in group_by:
            no_losses[col] = df[col].unique()[0]
    # in no_losses drop loss rows in the losing_models set
    no_losses = no_losses[~no_losses[category].isin(losing_models)]
    # Assign a losses value of zero to the no_losses dataframe
    no_losses['losses'] = 0

    # drop duplicates in the no_losses dataframe
    no_losses = no_losses.drop_duplicates()

    # Append the no_losses dataframe to the losses dataframe
    losses = pd.concat([losses, no_losses], join='outer', ignore_index=True)

    print(f'no_losses: \n{no_losses}')

    # Merge the wins and losses dataframes on the group_by and category columns, fill NaN values with 0
    results = pd.merge(wins, losses, on=group_by + [category], how='outer').fillna(0)

    # Calculate the difference between the wins and losses values
    results['diff'] = results['wins'] - results['losses']

    # Check if the group_by parameter is empty or not
    if group_by:
        # If not empty, calculate the rank within each group based on the diff column
        results['Rank'] = results.groupby(group_by)['diff'].rank(method=rank_method, ascending=False)
    else:
        # If empty, calculate the rank for the whole DataFrame based on the diff column
        results['Rank'] = results['diff'].rank(method=rank_method, ascending=False)
        
    # Sort by the group_by columns and the diff column in descending order
    results = results.sort_values(by=group_by + ['diff'], ascending=False)

    return results



if __name__ == "__main__":
    # Create the DataFrame
    data = {
        'task': ['baseline|contrastive|Rank', 'baseline|genericSD|Rank', 'baseline|positive|Rank', 'contrastive|genericSD|Rank', 'contrastive|positive|Rank', 'genericSD|positive|Rank'],
        'agg_label': [False, True, False, True, True, False],
        'Left Neural Network Model': ['baseline', 'baseline', 'baseline', 'contrastive', 'contrastive', 'genericSD'],
        'Right Neural Network Model': ['contrastive', 'genericSD', 'positive', 'genericSD', 'positive', 'positive'],
        'Item Type': ['Rank', 'Rank', 'Rank', 'Rank', 'Rank', 'Rank']
    }
    df = pd.DataFrame(data)

    # Current Result:
    # left_wins: 
    #   Item Type Neural Network Model  wins
    # 0      Rank             baseline     1
    # 1      Rank          contrastive     2
    # right_wins: 
    #   Item Type Neural Network Model  wins
    # 0      Rank          contrastive     1
    # 1      Rank             positive     2
    # rank_results: 
    #   Item Type Neural Network Model  wins  Rank
    # 1      Rank          contrastive     3   1.0
    # 2      Rank             positive     2   2.0
    # 0      Rank             baseline     1   3.0

    # Expected Result is similar to the Current Result but should have all 4 models.
    # ,Neural Network Model,wins,Rank
    # 1,contrastive,3.0,1.0
    # 2,positive,2.0,2.0
    # 0,baseline,1.0,3.0
    # 3,genericSD,1.0,3.0

    # Call the reconstruct_ranking function
    rank_results = reconstruct_ranking(df, group_by=['Item Type'], category='Neural Network Model', response='agg_label', rank_method='max')

    # Print the results
    print(f'rank_results: \n{rank_results}')

    # Example usage
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
    old_df = binary_rank_table_single_threaded(df, network_models)

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
