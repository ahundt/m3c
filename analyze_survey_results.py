""" Analyze survey results from Amazon Mechanical Turk.

    This script reads and reorganizes one CSV file of survey results into a DataFrame to prepare for statistical analysis.

    Copyright 2023 Andrew Hundt
"""
import argparse
import pandas as pd
import numpy as np
import json
import os
import re
import binary_rank
import crowdkit
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
import plot_ranking
import plot_consensus_alignment_skills


def check_column_group_for_consistent_values_in_another_column(df, columns_to_group=["HITId"], columns_to_match=["WorkerId"]):
    """ Check if a group of columns have consistent values in another column. Useful for dataset and processing validation.

        Parameters:
            df (pandas.DataFrame): The DataFrame containing the columns to check.
            column_to_check (str, optional): The name of the column to check (default is "HITId").
            column_to_match (str, optional): The name of the column to match (default is "WorkerId").

        Returns:
            bool: True if the values in the column are consistent, False otherwise.
    """
    # Vectorized pandas check if the values in the column to check are consistent
    # with the values in the column to match
    group = df.groupby(columns_to_group)[columns_to_match]
    all_ok = group.nunique().eq(1).all()

    # print a warning for the user if the values are not consistent, specifying which values are problematic
    if not all_ok.all():
        print(f'WARNING: {columns_to_group} values are not consistent with {columns_to_match} values!')
        print(f'Problematic values: {group.nunique()[group.nunique() != 1]}')
        # print the file and rows that apply
        print(f'Problematic file: {df["Source CSV File"].unique()}')
        print(f'Problematic rows: {df[group.nunique() != 1].index}')
        # raise an exception if the values are not consistent
        raise ValueError(f'{columns_to_group} rows have {columns_to_match} values that vary!')
    
    return all_ok

def extract_and_process_entry(entry, value):
    """ Extract and process a single entry from the Answer.taskAnswers column.
    """
    if value is None or value == "":
        return None
    if entry.endswith("-rating"):
        return int(value)
    elif entry.endswith("-checkbox"):
        return bool(value['1'])
    elif entry.endswith("-short-answer"):
        return value
    else:
        raise ValueError(f"Unexpected format in Answer.taskAnswers: {entry}")


def extract_and_process_task_answers(task_answers):
    """ Extract and process the Answer.taskAnswers column.

        The column contains a JSON string with the responses to the survey questions.
    """
    task_answers_dict = json.loads(task_answers)
    ratings_data = {}

    for entry, value in task_answers_dict[0].items():
        ratings_data[entry] = extract_and_process_entry(entry, value)

    return pd.Series(ratings_data)  # Convert the dictionary to a Series


def assess_worker_responses(
        binary_rank_df,
        worker_column="WorkerId", 
        label_column="Binary Rank Response Left Image is Greater",
        crowdkit_grouping_columns = ['Left Neural Network Model', 'Right Neural Network Model', 'Item Title Index', 'Item Title', 'Item Type', 'Country'],
        binary_rank_reconstruction_grouping_columns=['Item Title', 'Country'],
        seed=None,  
    ):
    """
    Assess worker responses using the MMSR (Matrix Mean-Subsequence-Reduced) algorithm.

    Parameters:
        df (pandas.DataFrame): The DataFrame containing worker responses.

    Returns:
        pandas.Series: The estimated skill levels of workers.
    """
    # print running crowdkit
    print('Running Crowdkit MMSR')
    # print the columns of the DataFrame
    print(f'binary_rank_df.columns: {binary_rank_df.columns}')
    # print the unique workerIds in the binary_rank_df
    print(f'Unique workers: binary_rank_df[worker_column].unique(): {len(binary_rank_df[worker_column].unique())}')
    # Assuming you have 'WorkerId' and 'Binary Rank Response Left Image is Greater' columns
    task_worker_label_df, table_restore_metadata = binary_rank.convert_table_to_crowdkit_format(
        binary_rank_df, worker_column=worker_column, label_column=label_column,
        task_columns=crowdkit_grouping_columns
    )
    # print table restore metadata
    # join crowdkit_grouping_columns with a dash and replace space with dash
    crowdkit_grouping_columns_str = '-'.join(crowdkit_grouping_columns).replace(' ', '-')
    # join binary_rank_reconstruction_grouping_columns with a dash
    binary_rank_reconstruction_grouping_columns_str = '-'.join(binary_rank_reconstruction_grouping_columns).replace(' ', '-')

    if seed is not None:
        np.random.seed(seed)
    # TODO(ahundt) might need to add a third label for "None" when there is no response, particularly for n_labels=2
    # Create the MMSR model https://toloka.ai/docs/crowd-kit/reference/crowdkit.aggregation.classification.m_msr.MMSR/
    from crowdkit.aggregation import MMSR
    mmsr = MMSR(
        n_iter=10000,
        tol=1e-10,
        n_workers=table_restore_metadata['n_workers'],
        n_tasks=table_restore_metadata['n_tasks'],
        n_labels=table_restore_metadata['n_labels'],  # Assuming binary responses
        workers_mapping=table_restore_metadata['workers_mapping'],
        tasks_mapping=table_restore_metadata['tasks_mapping'],
        labels_mapping=table_restore_metadata['labels_mapping']
    )
    print(f'st2_int.shape: {task_worker_label_df.shape} \ntask_worker_label_df:\n{task_worker_label_df}')
    task_worker_label_df.to_csv("task_worker_label_df.csv")

    print(f'Running CrowdKit Optimization MMSR.fit_predict()')
    # Fit the model and predict worker skills
    results_df = mmsr.fit_predict(task_worker_label_df)
    print(f'Finished CrowdKit Optimization MMSR.fit_predict(), Results:')
    print(results_df)
    # save results to a file
    results_df.to_csv(f"mmsr_results-{crowdkit_grouping_columns_str}.csv")

    #########################################
    # Extract the worker consensus alignment (aka "skills") (this can be visualized with plot_consensus_alignment_skills.py)

    # result = mmsr.fit_predict_score(simplified_table)
    worker_skills = pd.DataFrame(mmsr.skills_)

    # Create a smaller DataFrame with unique 'workerId' and 'Country' pairs
    unique_worker_countries = binary_rank_df[['WorkerId', 'Country']].drop_duplicates()

    # Create a mapping from 'workerId' to 'Country' in the smaller DataFrame
    country_mapping = unique_worker_countries.set_index('WorkerId')['Country']

    # Add the 'Country' column to worker_skills using the mapping
    worker_skills['Country'] = worker_skills.index.map(country_mapping)

    # rename "skill" Consensus Alignment and "worker" Participant
    worker_skills.rename(columns={ 'skill': 'Consensus Alignment', 'worker': 'Participant'}, inplace=True)

    # save worker skills to a file
    worker_skills.to_csv(f"mmsr_worker_skills-{crowdkit_grouping_columns_str}.csv")
    print(worker_skills)

    # plot the consensus alignment of peception (the underlying algorithm referes to this as "skills")
    plot_consensus_alignment_skills.swarm_violin_plot(worker_skills, filename=f"consensus_alignment_plot-{crowdkit_grouping_columns_str}", orient='v', size=(10, 6), palette=None)
    # convert the results_df to a dataframe and make the index the task columns

    #########################################
    # visualize the mmsr observation matrix
    observation_matrix = mmsr._observation_matrix
    # Visualizing the Observation Matrix as a Heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(observation_matrix, cmap='viridis', fmt='.2f')
    # sns.heatmap(observation_matrix, cmap='viridis', annot=True, fmt='.2f')
    plt.title('Observation Matrix Heatmap')
    plt.xlabel('Variables')
    plt.ylabel('Observations')
    # plt.show()
    # save the observation matrix and plot to a file
    np.savetxt(f"mmsr_observation_matrix-{crowdkit_grouping_columns_str}.csv", observation_matrix, delimiter=",")
    plt.savefig(f"mmsr_observation_matrix-{crowdkit_grouping_columns_str}.png")
    plt.savefig(f"mmsr_observation_matrix-{crowdkit_grouping_columns_str}.pdf")

    #########################################
    # visualize the mmsr covariance matrix
    covariance_matrix = mmsr._covariation_matrix
    # Visualizing the Covariance Matrix as a Heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(covariance_matrix, cmap='coolwarm', fmt='.2f')
    # sns.heatmap(covariance_matrix, cmap='coolwarm', annot=True, fmt='.2f')
    plt.title('Covariance Matrix Heatmap')
    plt.xlabel('Variables')
    plt.ylabel('Variables')
    # plt.show()
    # save the covariance matrix and plot to a file
    np.savetxt(f"mmsr_covariance_matrix-{crowdkit_grouping_columns_str}.csv", covariance_matrix, delimiter=",")
    plt.savefig(f"mmsr_covariance_matrix-{crowdkit_grouping_columns_str}.png")
    plt.savefig(f"mmsr_covariance_matrix-{crowdkit_grouping_columns_str}.pdf")
    
    #########################################
    # calculate the mmsr rank results
    results_df = binary_rank.restore_from_crowdkit_format(results_df, table_restore_metadata)
    print(f'Finished CrowdKit Optimization MMSR.fit_predict(), restore_from_crowdkit_format() results_df: {results_df}')
    results_df.to_csv(f"mmsr_results_restored-{binary_rank_reconstruction_grouping_columns_str}.csv")
    # print the columns of results_df
    print(f'results_df.columns: {results_df.columns}')
    # print results_df
    print(f'results_df:\n{results_df}')
    rank_results = binary_rank.reconstruct_ranking(results_df, binary_rank_reconstruction_grouping_columns)
    print(f'Finished CrowdKit Optimization MMSR.fit_predict(), reconstruct_ranking() rank_results: {rank_results}')
    rank_results.to_csv(f"mmsr_rank_results-{binary_rank_reconstruction_grouping_columns_str}.csv")
    # save a plot of the model rank results
    hue = None
    if 'Country' in binary_rank_reconstruction_grouping_columns:
        hue = 'Country'
    if 'Item Title' in binary_rank_reconstruction_grouping_columns:
        hue = 'Item Title'
    plot_ranking.strip_plot_rank(rank_results, x='Neural Network Model', y='Rank', hue=hue, filename=f"mmsr_rank_results-{binary_rank_reconstruction_grouping_columns_str}", show_plot=False)

    return rank_results, results_df, worker_skills


def plot_binary_comparisons(df, models_ordered=['contrastive','positive','baseline','genericSD']):
    """ 
    Create grouped bar charts comparing the rankings amongst methods head to head 

    Parameters:
        df (pandas.DataFrame): Binary Comparison data
    """
    # Get the mean binary rank comparison (proportion of times left was ranked lower than right)
    df_grouped = df.groupby(["Country", "Item Title", "Left Neural Network Model", "Right Neural Network Model"])\
        ['Binary Rank Response Left Image is Greater'].mean().reset_index()
    df_grouped['Binary Rank Response Left Image is Greater'] *= 100
    
    # Make a copy with reciprocal left & right
    df_grouped_rec = df_grouped.copy()
    df_grouped_rec["Left Neural Network Model"] = df_grouped["Right Neural Network Model"]
    df_grouped_rec["Right Neural Network Model"] = df_grouped["Left Neural Network Model"]
    df_grouped_rec['Binary Rank Response Left Image is Greater'] = 100 - df_grouped_rec['Binary Rank Response Left Image is Greater']

    # Double the dataframe so that left&right methods also contain right&left
    df_grouped = pd.concat([df_grouped, df_grouped_rec], ignore_index=True)

    # Remove rows where the two Neural Network Models compared are the same
    df_grouped = df_grouped[df_grouped["Left Neural Network Model"] != df_grouped["Right Neural Network Model"]]


    # Only include the key comparisons, such as contrastive with all and baseline with only genericSD
    for i in range(len(models_ordered)):
        for j in range(i-1, -1, -1):
            df_grouped = df_grouped[~((df_grouped["Left Neural Network Model"] == models_ordered[i]) \
                                    & (df_grouped["Right Neural Network Model"] == models_ordered[j]))]

    countries = df_grouped['Country'].unique()
    item_titles = df_grouped['Item Title'].unique()

    def plot_barchart(df_grouped_subset, ax):
        sns.barplot(
                ax=ax,
                data=df_grouped_subset, 
                x="Left Neural Network Model", 
                y='Binary Rank Response Left Image is Greater', 
                hue="Right Neural Network Model",
                order=models_ordered[:-1],
                hue_order=models_ordered[1:])
        ax.set_ylim(0, 100)
        ax.set_xlabel('Approach')
        ax.set_ylabel('% times approach ranked lower (better)')
        # Plot dashed line at 50%
        ax.axhline(y=50, color='r', linestyle='--')
        ax.legend(title=None)

    fig, axes = plt.subplots(len(countries), len(item_titles), figsize=(len(item_titles)*5, len(countries)*5))
    for i in range(len(countries)):
        for j in range(len(item_titles)):
            country = countries[i]
            item_title = item_titles[j]

            ax = axes[i,j] if len(countries) > 1 else axes[j]

            # Only look at the country and quesstions
            df_grouped_subset = df_grouped[(df_grouped['Country'] == country) & (df_grouped['Item Title'] == item_title)]
            plot_barchart(df_grouped_subset, ax)
            ax.set_title('{}, {}'.format(country, item_title))
            
    plt.savefig('binary_comparison_per_country.pdf')
    plt.close(fig)
    # plt.show()

    fig, axes = plt.subplots(1, len(item_titles), figsize=(len(item_titles)*5, 5))
    for j in range(len(item_titles)):
        country = countries[i]
        item_title = item_titles[j]

        ax = axes[j]

        # Only look at the question
        df_grouped_subset = df_grouped[df_grouped['Item Title'] == item_title]
        plot_barchart(df_grouped_subset, ax)
        ax.set_title('{}'.format(item_title))
            
    plt.savefig('binary_comparison.pdf')
    plt.close(fig)
    # plt.show()


def statistical_analysis(df, network_models, seed=None):
    """ Perform statistical analysis on the DataFrame.

        Parameters:
            df (pandas.DataFrame): The DataFrame containing worker responses.
                The key columns are: 
                    "Item Title Index", "Item Title", "Item Type", "Neural Network Model", 
                    "Image File Path", "Image Shuffle Index", "Response", "Country", 
                    "Source CSV Row Index", "Input.prompt", "Input.seed", "HITId", and "WorkerId".
                Note that the "Response" column is the rating for part of an item, such as an individual image's rank.
            network_models (list): List of network models used for matching image columns.

        Returns:
            pandas.DataFrame: The aggregated DataFrame with results.
    """
    # Assess worker responses
    # TODO Define your statistical analysis here
    # TODO(ahundt) WARNING: df is unfiltered as of Nov 13, 2023! Items with invalid rank combinations and empty values are included. Add appropriate filtering for your analysis method.
    print(df)
    df.to_csv("statistical_analysis_input.csv")

    ####################
    # All data stats
    ####################
    # Group the DataFrame by "Neural Network Model," "Country," and "Item Title"
    grouped = df.groupby(["Neural Network Model", "Item Title"])

    # Define the aggregation functions you want to apply
    aggregation_functions = {
        "Response": ["count", "median", "min", "max", "sem", "mean"],
        "WorkerId": ["nunique"]
    }

    # Perform aggregation and reset the index
    aggregated_df = grouped.agg(aggregation_functions).reset_index()

    # Save the aggregated DataFrame to a CSV file
    aggregated_df.to_csv("aggregated_statistical_output_all_data.csv", index=False)

    ####################
    # Per country stats
    ####################
    # Group the DataFrame by "Neural Network Model," "Country," and "Item Title"
    grouped = df.groupby(["Neural Network Model", "Country", "Item Title"])

    # Define the aggregation functions you want to apply
    aggregation_functions = {
        "Response": ["count", "median", "min", "max", "sem", "mean"],
        "WorkerId": ["nunique"],
        "Country": ["nunique"]
    }

    # Perform aggregation and reset the index
    aggregated_df_per_country = grouped.agg(aggregation_functions).reset_index()

    # Save the aggregated DataFrame to a CSV file
    aggregated_df_per_country.to_csv("aggregated_statistical_output_by_country.csv", index=False)


    ####################
    # Binary Rank Stats
    ####################
    binary_rank_df = binary_rank.binary_rank_table(df, network_models)
    # single threaded version
    # binary_rank_df = binary_rank.binary_rank_table(df, network_models)
    binary_rank_df.to_csv("statistical_output_binary_rank.csv")
    # print the count of unique workers
    print(f'Unique workers in binary rank table: binary_rank_df["WorkerId"].unique(): {len(binary_rank_df["WorkerId"].unique())}')
    # print a warning if the number of workers isn't the same as the number of unique workers in the original table
    if len(binary_rank_df["WorkerId"].unique()) != len(df["WorkerId"].unique()):
        # throw an exception if the number of workers isn't the same as the number of unique workers in the original table
        raise ValueError(f'WARNING: CREATING THE BINARY TABLE CHANGED THE NUMBER OF WORKERS, THERE IS A BUG!'
                         f'binary_rank_df["WorkerId"].unique(): {binary_rank_df["WorkerId"].unique()} != df["WorkerId"].unique(): {df["WorkerId"].unique()}')

    plot_binary_comparisons(binary_rank_df.copy())

    # ranking per country and per question
    rank_results_ci, results_df_ci, worker_skills_ci = assess_worker_responses(binary_rank_df, seed=seed)
    # ranking per country
    rank_results, results_df, worker_skills = assess_worker_responses(binary_rank_df, crowdkit_grouping_columns=['Left Neural Network Model', 'Right Neural Network Model', 'Item Title Index', 'Country'], binary_rank_reconstruction_grouping_columns=['Country'], seed=seed)
    # ranking per question
    rank_results, results_df, worker_skills = assess_worker_responses(binary_rank_df, crowdkit_grouping_columns=['Left Neural Network Model', 'Right Neural Network Model', 'Item Title Index', 'Item Title', 'Item Type'], binary_rank_reconstruction_grouping_columns=['Item Title'], seed=seed)
    # overall ranking (across all countries and questions)
    rank_results, results_df, worker_skills = assess_worker_responses(binary_rank_df, crowdkit_grouping_columns=['Left Neural Network Model', 'Right Neural Network Model','Item Type'], binary_rank_reconstruction_grouping_columns=[], seed=seed)

    # TODO(ahundt) add statistical analysis here, save results to a file, and visualize them

    return aggregated_df


def assign_network_models_to_duplicated_rows(
    dataframe, 
    duplicate_column="Item Type", 
    match_values=["Rank", "Binary Checkbox"], 
    new_column_name="Neural Network Model", 
    network_models=["baseline", "contrastive", "genericSD", "positive"]
):
    """
    Assign specified network models to duplicated rows in a DataFrame based on matching values in a column.

    This function duplicates df rows with "Rank" "Item Type" by the number of network models
    because that is the number of ratings per item, e.g. individual ranks or individual binary checkboxes.

    Parameters:
        dataframe (pandas.DataFrame): The DataFrame to modify.
        duplicate_column (str, optional): The name of the column used to filter rows for duplication (default is "Item Type").
        match_values (list, optional): List of values in 'duplicate_column' to trigger duplication (default is ["Rank", "Binary Checkbox"]).
        new_column_name (str, optional): The name of the new column to add on duplicated rows (default is "Neural Network Model").
        network_models (list, optional): List of network models to assign on duplicated rows (default is ["baseline", "contrastive", "genericSD", "positive"]).

    Returns:
        pandas.DataFrame: A new DataFrame with the specified column added on duplicated rows.
    """
    
    # Filter rows based on 'duplicate_column' and 'match_values'
    filtered_dataframe = dataframe[dataframe[duplicate_column].isin(match_values)]

    # Create a new DataFrame with repeated rows for each value in 'network_models'
    final_dataframe = filtered_dataframe.loc[filtered_dataframe.index.repeat(len(network_models))].reset_index(drop=True)

    # Add the new column and set it to the values in 'network_models'
    final_dataframe[new_column_name] = network_models * len(filtered_dataframe)

    # Merge the final DataFrame with the original DataFrame using an outer join
    result_dataframe = dataframe.merge(final_dataframe, how="left", on=dataframe.columns.tolist())

    # Fill NaN values with None
    result_dataframe.fillna("None", inplace=True)

    return result_dataframe


def add_survey_item_data_to_dataframe(df, human_survey_items, network_models):
    """ The survey is divided into items like "offensiveness" and "image and description alignment",
        so here we add the item titles, the type of item (e.g. Rank, Binary Checkbox or Short Answer),
        and the network models used for each item to the DataFrame. This is done by duplicating the rows
        in the DataFrame.

        New columns added:
        - "Item Title" specifying the descriptive title of the item, e.g. "offensiveness".
        - "Item Title Index" specifying the index of the item title from human_survey_item csv rows, e.g. 1 for "offensiveness".
        - "Item Type" specifying the type of item, e.g. Rank, Binary Checkbox or Short Answer.
        - "Neural Network Model" specifying the network model used for that item response component to be used in the ablation.

        Parameters:
            df (pandas.DataFrame): The DataFrame to modify.
            human_survey_items (pandas.DataFrame): The DataFrame containing survey item data.
            network_models (list): List of network models used for matching image columns.
        
        Returns:
            pandas.DataFrame: A new DataFrame with the specified columns added.
    """
    df_initial_size = len(df)

    # Add "Item Title Index" column to human_survey_items
    human_survey_items['Item Title Index'] = human_survey_items.index + 1
    num_items = len(human_survey_items)

    # Duplicate df by the number of items
    df = df.loc[df.index.repeat(num_items)].reset_index(drop=True)

    # Assign Tiled "Item Title", "Item Title Index", and 'Item Type' values to df
    df['Item Title'] = human_survey_items['Item Title'].tolist() * df_initial_size
    df['Item Title Index'] = human_survey_items['Item Title Index'].tolist() * df_initial_size
    df['Item Type'] = human_survey_items['Item Type'].tolist() * df_initial_size

    # Duplicate df rows with "Rank" "Item Type" by the number of network models
    # because that is the number of ratings per item, e.g. individual ranks or individual binary checkboxes.
    df = assign_network_models_to_duplicated_rows(df, duplicate_column="Item Type", match_values=["Rank", "Binary Checkbox"], new_column_name="Neural Network Model", network_models=network_models)

    return df


def get_response_rows_per_image(df, network_models):
    """
    Modify a DataFrame by adding new columns based on image-related columns.

    Parameters:
        df (pandas.DataFrame): The DataFrame to be modified.
        network_models (list): List of network models used for matching image columns.

    Returns:
        pandas.DataFrame: The modified DataFrame.
    """

    # Get the list of image-related columns
    img_columns = [col for col in df.columns if col.startswith("Input.img")]
    df_columns = df.columns
    print(df_columns)

    # Use regex to read trailing integers from the columns and create a mapping
    col_mapping = {col: int(re.search(r'\d+$', col).group()) for col in img_columns}

    # Initialize new columns with None values
    df['Image File Path'] = None
    df['Image Shuffle Index'] = None
    df['Response'] = None

    # Iterate through rows of the DataFrame
    for idx, row in df.iterrows():
        if row['Neural Network Model'] is not None:  # Step 1
            folder_name = row['Neural Network Model']

            # Search for matching image data in image-related columns using the mapping
            for col, image_shuffle_index in col_mapping.items():
                file_path = row[col]

                # Check if folder_name is in the file_path
                if folder_name in file_path:
                    # Create the column name for "Response"
                    response_column_name = f"promptrow{row['Item Title Index']}-img{image_shuffle_index}-rating"

                    # Assign the values to the new columns
                    df.at[idx, 'Image File Path'] = file_path
                    df.at[idx, 'Image Shuffle Index'] = image_shuffle_index
                    df.at[idx, 'Response'] = row[response_column_name]

    # Drop the original image-related columns
    # df.drop(columns=img_columns, inplace=True)

    return df


def process_survey_results_csv(csv_file, survey_items_file, network_models):
    """ Read and reorganize one CSV file of survey results into a DataFrame to prepare for statistical analysis.

        Load the csv files, add the survey metadata, and put each element of a response on a separate row
        A value in the response column is a single image rank, a single binary checkbox, or a single short answer.
        
        The new columns added are:
             "Item Title Index", "Item Title", "Item Type", "Neural Network Model", 
             "Image File Path", "Image Shuffle Index", "Country", and "Response".

        Parameters:
            csv_file (str): Path to the CSV file.
            survey_items_file (str): Path to the human_survey_items.csv file.
            network_models (list): List of network models used for matching image columns.
        
        Returns:
            pandas.DataFrame: A new DataFrame with the specified columns added.
    """
    # Load CSV Data Using Pandas
    df = pd.read_csv(csv_file)
    # add the csv file name as a column
    df['Source CSV File'] = csv_file
    # Create "Source CSV Row Index" column
    df['Source CSV Row Index'] = df.index
    check_column_group_for_consistent_values_in_another_column(df, columns_to_group=['Source CSV File','Source CSV Row Index'], columns_to_match=["WorkerId"])

    # Apply extract_and_process_task_answers to extract and process ratings data
    ratings_df = df['Answer.taskAnswers'].apply(extract_and_process_task_answers)

    # Concatenate the new DataFrame with the original one
    df = pd.concat([df, ratings_df], axis=1)

    # Extract country name from the survey results "Title" column
    # Assuming the last word in the "Title" column is the country name
    df['Title'] = df['Title'].str.strip()  # Remove leading/trailing spaces
    country_name = df['Title'].str.split().str[-1]
    df['Country'] = country_name

    # Determine the number of items for analysis from human_survey_items.csv
    human_survey_items = pd.read_csv(survey_items_file)
    num_items = len(human_survey_items)

    # Add "Item Title Index", "Item Title", "Item Type", and "Neural Network Model" columns to df,
    # and duplicate rows by the number of items so that these columns are unique and can be used for analysis
    df = add_survey_item_data_to_dataframe(df, human_survey_items, network_models)
    
    # Add "Image File Path", "Image Shuffle Index", and "Response" columns to df
    # by matching image-related columns with the "Neural Network Model" column
    # and the "promptrowX-imgY-rating" columns. 
    # The "Response" column is the rating for part of an item, such as an individual image's ranks.
    df = get_response_rows_per_image(df, network_models)

    check_column_group_for_consistent_values_in_another_column(df, columns_to_group=['Source CSV File','Source CSV Row Index'], columns_to_match=["WorkerId"])
    return df


def test():
    """ Small tests of the functions in this file.
    """
    # Example usage of get_response_rows_per_image():
    # Corrected example usage with data format:
    data = {
        'Input.img1': ['abc/hello1.png', 'def/helo2.png', 'ghi/hello3.png'],
        'Input.img2': ['def/ty.png', 'abc/as.png', 'ghi/io.png'],
        'Item Title Index': [1, 2, 3],
        'Neural Network Model': ['abc', 'def', None]
    }

    # Columns needed for response_column_name
    for idx in range(1, 4):  # Assuming promptrow1, promptrow2, promptrow3 are used
        for img_index in range(1, 3):  # Assuming img1 and img2 are used
            data[f'promptrow{idx}-img{img_index}-rating'] = [1, 2, 3]

    df = pd.DataFrame(data)
    network_models = ['abc', 'def']
    df = get_response_rows_per_image(df, network_models)
    # print(df)


    # Example usage of assign_network_models_to_duplicated_rows():
    data = {
        "Item Type": ["Rank", "Binary Checkbox", "Other"],
        "Other_Column": [1, 2, 3]
    }
    df = pd.DataFrame(data)
    result = assign_network_models_to_duplicated_rows(df)
    # print(result)


def main():
    """ Main function.
    """
    # Command Line Parameter Parsing
    parser = argparse.ArgumentParser(description="Survey Data Analysis")
    parser.add_argument("--response_results", type=str, default="m3c_cvpr_results_11_14", help="Path to the file or folder containing CSV files with Amazon Mechanical Turk survey response results.")
    # parser.add_argument("--response_results", type=str, default="Batch_393773_batch_results.csv", help="Path to the file or folder containing CSV files with Amazon Mechanical Turk survey response results.")
    parser.add_argument("--survey_items_file", type=str, default="human_survey_items.csv", help="Path to the human_survey_items.csv file")
    parser.add_argument("--network_models", type=str, nargs='+', default=["baseline", "contrastive", "genericSD", "positive"], help="List of neural network model names")
    # note that you can specify no random seed by passing: --random_seed=None
    parser.add_argument("--random_seed", type=int, default=8827, nargs='?', help="Random seed for reproducibility, default is 8827, you can specify no random seed with --random_seed=None.")
    args = parser.parse_args()

    test()

    # Get the list of CSV files to process
    if os.path.isfile(args.response_results):
        csv_files = [args.response_results]
    elif os.path.isdir(args.response_results):
        csv_files = [os.path.join(args.response_results, filename) for filename in os.listdir(args.response_results) if filename.endswith(".csv")]

    # Load the csv files, add the survey metadata, and put each element of a response on a separate row
    # An element of a response is a single image rank, a single binary checkbox, or a single short answer.
    #
    # The new columns added are:
    # "Item Title Index", "Item Title", "Item Type", "Neural Network Model", "Image File Path", "Image Shuffle Index", "Response", "Country", "Source CSV Row Index".
    # 
    # The key columns of df, including new columns added are: 
    # "Item Title Index", "Item Title", "Item Type", "Neural Network Model", "Image File Path", "Image Shuffle Index", "Response", "Country", "Source CSV Row Index", "Input.prompt", "Input.seed", "HITId", and "WorkerId".
    dataframes = []
    for csv_file in tqdm(csv_files):
        df = process_survey_results_csv(csv_file, args.survey_items_file, args.network_models)
        dataframes.append(df)
    
    # Concatenate the DataFrames
    combined_df = pd.concat(dataframes, axis=0)
    aggregated_df = statistical_analysis(combined_df, args.network_models, args.random_seed)

if __name__ == "__main__":
    main()
