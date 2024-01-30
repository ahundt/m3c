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
from crowdkit.aggregation import MMSR, NoisyBradleyTerry
from crowdkit.metrics.data import alpha_krippendorff, consistency, uncertainty
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
import plot_ranking
import plot_consensus_alignment_skills
# import inter_rater_reliability

def to_dataframe(dest_df):
    """
    Ensures the input is a Pandas DataFrame.
    
    If the input is not a DataFrame, this function converts it to one.
    If the input is a Series, it resets the index to ensure the index becomes a column in the resulting DataFrame.
    If the Series index has a name, it will be set as the column name in the resulting DataFrame.

    Parameters:
    dest_df (DataFrame or Series or any): The input data that needs to be converted or ensured as a DataFrame.

    Returns:
    DataFrame: The input as a DataFrame.

    Example:
    >>> data = [1, 2, 3, 4, 5]
    >>> index = pd.Index(data, name='MyIndex')
    >>> series = pd.Series([10, 20, 30, 40, 50], index=index)
    >>> result_df = to_dataframe(series)
    >>> print(result_df)
    """
    if not isinstance(dest_df, pd.DataFrame):
        # If dest_df is not a DataFrame, convert it to one
        if isinstance(dest_df, pd.Series):
            index_name = dest_df.index.name
            dest_df = dest_df.reset_index()
            if index_name:
                dest_df.columns = [index_name] + list(dest_df.columns[1:])
        else:
            dest_df = pd.DataFrame(dest_df)
    return dest_df


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


def add_column_from_mapping(dest_df, source_df, column_to_match="WorkerId", column_to_add="Country"):
    """
    Adds a new column to the destination DataFrame based on a mapping from the source DataFrame.

    The mapping is created from two columns in the source DataFrame: one to use as the key (column_to_match) 
    and one to use as the value (column_to_add). Each unique value in the key column is mapped to the corresponding 
    value in the value column. This mapping is then used to add a new column to the destination DataFrame. 
    The new column's values are determined by looking up each value in the destination DataFrame's column_to_match 
    column in the mapping.

    Parameters:
    dest_df (pandas.DataFrame): The DataFrame to which the new column will be added.
    source_df (pandas.DataFrame): The DataFrame from which the mapping will be created.
    column_to_match (str): The name of the column in the source DataFrame to use as the key for the mapping. Default is 'WorkerId'.
    column_to_add (str): The name of the column in the source DataFrame to use as the value for the mapping, and the name of the new column in the destination DataFrame. Default is 'Country'.

    Returns:
    pandas.DataFrame: The destination DataFrame with the new column added.
    """
    # Ensure dest_df is a DataFrame
    dest_df = to_dataframe(dest_df)

    # Create a smaller DataFrame with unique pairs from the source DataFrame
    unique_pairs = source_df[[column_to_match, column_to_add]].drop_duplicates()

    # Create a mapping from the column_to_match to the column_to_add in the smaller DataFrame
    column_mapping = unique_pairs.set_index(column_to_match)[column_to_add]

    # Add the new column to dest_df using the mapping
    dest_df[column_to_add] = dest_df[column_to_match].map(column_mapping)

    return dest_df


def worker_skills_add_country_and_save_to_csv(worker_skills, binary_rank_df, csv_filename="worker_skills.csv", plot_filename="worker_skills", xlim=(0, 1.1)):
    """ Save the worker skills to a CSV file, adding their country as an extra column.
    
    Extract the participant (worker) consensus alignment (aka "skills") (this can be visualized with plot_consensus_alignment_skills.py)
    """
    # make it a dataframe if it isn't already
    worker_skills = to_dataframe(worker_skills)

    # Add the 'Country' column to worker_skills using the mapping
    worker_skills = add_column_from_mapping(worker_skills, binary_rank_df, 'WorkerId', 'Country')

    # rename "skill" Consensus Alignment and "worker" Participant
    worker_skills.rename(columns={ 'skill': 'Consensus Alignment', 'worker': 'Participant'}, inplace=True)

    # save worker skills to a file
    worker_skills.to_csv(csv_filename)
    print(f'worker_skills:\n{worker_skills}')
    # plot the consensus alignment of peception (the underlying algorithm referes to this as "skills")
    plot_consensus_alignment_skills.swarm_violin_plot(worker_skills, filename=plot_filename, orient='v', size=(10, 6), palette=None, xlim=xlim)
    return worker_skills


# def prepare_kendalls_w_data(df,
#         worker_column="WorkerId", 
#         label_column="Binary Rank Response Left Image is Greater",
#         crowdkit_grouping_columns = ['Left Neural Network Model', 'Right Neural Network Model', 'Item Title Index', 'Item Title', 'Item Type', 'Country'],
#         item_grouping_columns=['Item Title', 'Country', 'Source CSV Row Index']):
#     #########################################
#     # Kendall's w
#     grouping_str = '-'.join(crowdkit_grouping_columns).replace(' ', '-')
#     # Assuming you have 'WorkerId' and 'Binary Rank Response Left Image is Greater' columns
#     # debug save task_worker_label_df to csv
#     # task_worker_label_df.to_csv(f"task_worker_label_df-{crowdkit_model}-{crowdkit_grouping_columns_str}.csv")
#     # Pivot the DataFrame to get labels by worker and task
#     annotations = df.pivot_table(index='worker', columns='task', values='label', aggfunc='first').reset_index().astype(float)
#     kendalls_w = inter_rater_reliability.kendalls_w(annotations)
#     print(f'Kendall\'s W: {kendalls_w}')
#     # save to csv
#     annotations.to_csv(f"kendalls_w_annotations-{grouping_str}.csv")

# def assess_kendalls_w(df,
#         worker_column="WorkerId", 
#         label_column="Response",
#         crowdkit_grouping_columns=['Item Title', 'Country', 'Source CSV Row Index'],
#         item_grouping_columns=['Item Title', 'Country', 'Source CSV Row Index']):
    
#     # Print the columns
#     print(f'kendalls_w_df.columns: {df.columns}')
    
#     # Make a copy of the dataframe
#     df = df.copy()
    
#     # Group by 'crowdkit_grouping_columns' and convert 'Response' column to numeric
#     df[label_column] = df.groupby(crowdkit_grouping_columns)[label_column].transform(lambda x: pd.to_numeric(x, errors='coerce'))
    
#     # Pivot the dataframe to have items as rows and raters as columns
#     pivot_df = df.pivot_table(index=item_grouping_columns, columns=worker_column, values=label_column)
    
#     # Calculate Kendall's W
#     kendalls_w_value = inter_rater_reliability.kendalls_w(pivot_df)
#     print(f"Kendall's W: {kendalls_w_value}")
    
#     # Save annotations to CSV file
#     grouping_str = '-'.join(crowdkit_grouping_columns).replace(' ', '-')
#     pivot_df.to_csv(f"kendalls_w_annotations-{grouping_str}.csv")

#     return kendalls_w_value



def plot_rank_strip(results_df, crowdkit_model, binary_rank_reconstruction_grouping_columns, show_plot=False):
    """ Plot the rank results using strip plot.
    """
    binary_rank_reconstruction_grouping_columns_str = '-'.join(binary_rank_reconstruction_grouping_columns).replace(' ', '-')
    # reverse the grouping columns because that's consistent with the other plots
    binary_rank_reconstruction_grouping_columns = binary_rank_reconstruction_grouping_columns[::-1]
    # Aggregate the ranks based on the groupings and 'agg_label' column
    aggregation_functions = {
        'agg_score': ['mean', 'median', 'sem'],
        'Binary Rank Image': ['nunique', 'count'],
        'Neural Network Model': ['nunique', 'count'],
        'Item Title': ['nunique', 'count']
    }
    results_agg_df = results_df.groupby(binary_rank_reconstruction_grouping_columns).agg(aggregation_functions).reset_index()
    results_agg_df.to_csv(f"{crowdkit_model}_rank_results-{binary_rank_reconstruction_grouping_columns_str}_aggregated_statistical_summary.csv")
    # Aggregate the ranks based on the groupings and 'agg_label' column
    aggregation_functions = {
        'agg_score': 'mean'
    }
    results_agg_df = results_df.groupby(binary_rank_reconstruction_grouping_columns).agg(aggregation_functions).reset_index()
    # group, then sort within groups by agg_score
    # results_agg_df.sort_values(by=binary_rank_reconstruction_grouping_columns + ['agg_score'], inplace=True, ascending=True)
    # get the number of models
    # num_models = len(results_agg_df['Neural Network Model'].unique())
    # subtract the aggreesive score from 1 to get the rank
    # results_agg_df['Rank'] = results_agg_df['agg_score'].apply(lambda x: (1 - x) * main_category_count)
    # group again and set the rank to the one indexed index within the group
    if len(binary_rank_reconstruction_grouping_columns) > 1:
        # exclude the last column from the grouping because that's what we want to rank by (Neural Network Model)
        group_cols = binary_rank_reconstruction_grouping_columns[:-1]
        results_agg_df['Rank'] = results_agg_df.groupby(group_cols)['agg_score'].rank(method='first', ascending=False)
        # sort by rank within the groups
        results_agg_df.sort_values(by=binary_rank_reconstruction_grouping_columns[:-1] + ['Rank'], inplace=True, ascending=True)
    else:
        results_agg_df['Rank'] = results_agg_df['agg_score'].rank(method='first', ascending=False)
        # sort by rank
        results_agg_df.sort_values(by=['Rank'], inplace=True, ascending=True)
    # add a rank column by the index within the group
    results_agg_df.to_csv(f"{crowdkit_model}_rank_results-{binary_rank_reconstruction_grouping_columns_str}.csv")
    # print(f'Finished CrowdKit Optimization NoisyBradleyTerry.fit_predict(), Results:')
    # print(results_agg_df)

    # create the rank df by dropping the agg_score column
    rank_df = results_agg_df.drop(columns=['agg_score'])

    # save a plot of the model rank results
    hue = None
    if 'Country' in binary_rank_reconstruction_grouping_columns:
        hue = 'Country'
    if 'Item Title' in binary_rank_reconstruction_grouping_columns:
        hue = 'Item Title'
    if ['Country', 'Item Title'] in binary_rank_reconstruction_grouping_columns:
        hue = ['Country', 'Item Title']
    
    if binary_rank_reconstruction_grouping_columns:
        title = f"Model Rankings by {', '.join(binary_rank_reconstruction_grouping_columns)}"
    else:
        title = f"Model Rankings Across All Responses"

    plot_ranking.strip_plot_rank(rank_df, x='Neural Network Model', y='Rank', hue=hue, filename=f"{crowdkit_model}_rank_results-{binary_rank_reconstruction_grouping_columns_str}", show_plot=show_plot, title=title)


def assess_worker_binary_responses_noisy_bradley_terry(
        binary_rank_df,
        worker_column="WorkerId", 
        label_column="Binary Rank Response Left Image is Greater",
        pairwise_left_columns=['Left Binary Rank Image', 'Left Neural Network Model', 'Item Title', 'Country'],
        pairwise_right_columns=['Right Binary Rank Image', 'Right Neural Network Model', 'Item Title', 'Country'],
        binary_rank_reconstruction_grouping_columns=['Neural Network Model', 'Item Title', 'Country'],
        crowdkit_model='noisy_bradley_terry',
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
    print(f'Running Crowdkit {crowdkit_model}')
    # print the columns of the DataFrame
    print(f'binary_rank_df.columns: {binary_rank_df.columns}')
    # print the unique workerIds in the binary_rank_df
    print(f'Unique workers: binary_rank_df[worker_column].unique(): {len(binary_rank_df[worker_column].unique())}')
    # join crowdkit_grouping_columns with a dash and replace space with dash
    # crowdkit_grouping_columns_str = '-'.join(crowdkit_grouping_columns).replace(' ', '-')
    # join binary_rank_reconstruction_grouping_columns with a dash
    binary_rank_reconstruction_grouping_columns_str = '-'.join(binary_rank_reconstruction_grouping_columns).replace(' ', '-')

    task_worker_label_df, table_restore_metadata = binary_rank.convert_table_to_crowdkit_pairwise_format(
        binary_rank_df, worker_column=worker_column, label_column=label_column, pairwise_left_columns=pairwise_left_columns, pairwise_right_columns=pairwise_right_columns
    )

    #########################################
    # Noisy Bradley Terry
    # pairwise comparison algorithm like Noisy BradleyTerry https://toloka.ai/docs/crowd-kit/reference/crowdkit.aggregation.pairwise.noisy_bt.NoisyBradleyTerry/ https://github.com/Toloka/crowd-kit/blob/v1.2.1/examples/Readability-Pairwise.ipynb
    if seed is not None:
        np.random.seed(seed)
    noisy_bt = NoisyBradleyTerry(
        n_iter=100,
        tol=1e-5,
        random_state=seed
    )

    print(f'st2_int.shape: {task_worker_label_df.shape} \ntask_worker_label_df:\n{task_worker_label_df}')
    task_worker_label_df.to_csv("task_worker_label_df.csv")

    print(f'Running CrowdKit Optimization NoisyBradleyTerry.fit_predict()')
    # Fit the model and predict worker skills
    noisy_bt = noisy_bt.fit(task_worker_label_df)
    # set the skills index name to WorkerId
    noisy_bt.skills_.index.name = 'WorkerId'
    print(f'Finished CrowdKit Optimization NoisyBradleyTerry.fit(), Results:')
    # plot and save worker skills
    worker_skills = worker_skills_add_country_and_save_to_csv(
        noisy_bt.skills_, binary_rank_df,
        csv_filename=f"{crowdkit_model}_consensus_alignment-{binary_rank_reconstruction_grouping_columns_str}.csv", 
        plot_filename=f"{crowdkit_model}_consensus_alignment_plot-{binary_rank_reconstruction_grouping_columns_str}")
    
    noisy_bt.scores_.to_csv(f"{crowdkit_model}_scores-{binary_rank_reconstruction_grouping_columns_str}.csv")
    # restore the results dataframe
    # results_df = binary_rank.restore_from_crowdkit_pairwise_format(noisy_bt.scores_, table_restore_metadata)
    results_df = noisy_bt.scores_.to_frame()
    results_df['label'] = results_df.index
    # after running noisy bradley terry, the label column is renamed to agg_label
    # make a left and right column title list by zipping the pairwise_left_columns and pairwise_right_columns, 
    # stripping the words Left and Right, and making sure they are otherwise equal
    left_column_titles = [col.replace('Left ', '') for col in table_restore_metadata['column_titles']['left']]
    right_column_titles = [col.replace('Right ', '') for col in table_restore_metadata['column_titles']['right']]
    # if the column titles groups are not equal print a warning
    if left_column_titles != right_column_titles:
        print(f'WARNING: assess_worker_binary_responses_noisy_bradley_terry() left_column_titles != right_column_titles: {left_column_titles} != {right_column_titles}'
              f'\nThe pairwise_left_columns and pairwise_left_columns are expected to be equal once all "Left " and "Right " prefixes are stripped from the list elements.')

    # split the 'label' column into multiple columns
    for i, col in enumerate(left_column_titles):
        results_df[col] = results_df['label'].apply(lambda x: x.split(table_restore_metadata['separator'])[i])

    results_df.to_csv(f"{crowdkit_model}_results-{binary_rank_reconstruction_grouping_columns_str}.csv")

    # plot the rank results
    plot_rank_strip(results_df.copy(), crowdkit_model, binary_rank_reconstruction_grouping_columns)
    plot_rank_strip(results_df.copy(), crowdkit_model, binary_rank_reconstruction_grouping_columns[:-1]) if len(binary_rank_reconstruction_grouping_columns) > 1 else None
    plot_rank_strip(results_df.copy(), crowdkit_model, binary_rank_reconstruction_grouping_columns[:-2]) if len(binary_rank_reconstruction_grouping_columns) > 2 else None
    plot_rank_strip(results_df.copy(), crowdkit_model, [binary_rank_reconstruction_grouping_columns[0], binary_rank_reconstruction_grouping_columns[-1]]) if len(binary_rank_reconstruction_grouping_columns) > 1 else None
    # save biases to a file
    # TODO(ahundt) address the fact that the images have been sorted by model name, which means the biases will detect a bias according to the model name, not worker bias
    noisy_bt.biases_.to_csv(f"{crowdkit_model}_biases-{binary_rank_reconstruction_grouping_columns_str}.csv")

    return results_df, worker_skills


def reliability_metrics(binary_rank_df, 
                        worker_column="WorkerId", 
                        label_column="Binary Rank Response Left Image is Greater", 
                        crowdkit_grouping_columns=['Item Title', 'Country', 'HITId', 'Left Binary Rank Image', 'Right Binary Rank Image', 'Left Neural Network Model', 'Right Neural Network Model'], 
                        reliability_grouping_columns=['Country'],
                        filename=True):
    """ Calculate Krippendorf's alpha, consistency, and uncertainty for all data and by country.

    TODO(ahundt) Don't use Krippendorf's alpha until suspicion that the way we sorted questions & labels might cause krippendorf agreement to be underestimated is fixed.
    
    Parameters:
    binary_rank_df (DataFrame): Input DataFrame containing binary rank data.
    worker_column (str): Column name for worker identification.
    label_column (str): Column name for binary rank response.
    crowdkit_grouping_columns (list): Columns for crowdkit grouping.
    reliability_grouping_columns (list): Columns for reliability grouping.

    Returns:
    DataFrame: DataFrame with calculated metrics.
    """
    # Create a copy of the original DataFrame
    task_worker_label_df = binary_rank_df.copy()
    
    # Convert the binary_rank_df to the format required by crowdkit
    task_worker_label_df, table_restore_metadata = binary_rank.convert_table_to_crowdkit_classification_format(
        task_worker_label_df, worker_column=worker_column, label_column=label_column,
        task_columns=crowdkit_grouping_columns, append_columns=crowdkit_grouping_columns
    )
    
    # # Calculate metrics for each group
    groups = task_worker_label_df.groupby(reliability_grouping_columns)
    aggregated_metrics = groups.apply(lambda x: pd.Series({
        'Krippendorf\'s Alpha': alpha_krippendorff(x) if len(x) > 1 else np.nan,
        'Uncertainty': uncertainty(x) if len(x) > 1 else np.nan,
        'Consistency': consistency(x) if len(x) > 1 else np.nan
    })).reset_index()
    # print the number of tasks
    print(f'reliability_metrics() Number of tasks: {len(task_worker_label_df["task"].unique())} rows: {len(task_worker_label_df)}')
    # save the task_worker_label_df to a file
    task_worker_label_df.to_csv(f"reliability_task_worker_label_df.csv")

    # Calculate metrics for 'All Countries'
    alpha_all_countries = alpha_krippendorff(task_worker_label_df)
    uncertainty_all_countries = uncertainty(task_worker_label_df)
    consistency_all_countries = consistency(task_worker_label_df)
    
    # Create 'All Countries' row
    all_countries_values = {'Krippendorf\'s Alpha': alpha_all_countries,
                            'Uncertainty': uncertainty_all_countries,
                            'Consistency': consistency_all_countries}
    for column in reliability_grouping_columns:
        all_countries_values[f'{column}'] = 'All'
    
    # Create DataFrame for 'All Countries'
    reliability_df = pd.DataFrame(all_countries_values, index=[0])

    # Concatenate 'All Countries' row with the aggregated metrics
    reliability_df = pd.concat([reliability_df, aggregated_metrics], ignore_index=True)

    # reorder the columns
    columns_order = reliability_grouping_columns + ['Krippendorf\'s Alpha', 'Uncertainty', 'Consistency']
    reliability_df = reliability_df[columns_order]
    
    # Save the results to a CSV file
    crowdkit_grouping_columns_str = '-'.join(crowdkit_grouping_columns).replace(' ', '-')
    if filename:
        reliability_df.to_csv(f"reliability-{crowdkit_grouping_columns_str}.csv", index=False)  # Avoid writing the index as a column
    
    # Print the table
    print(f'reliability_df:\n{reliability_df}')
    return reliability_df



def assess_worker_responses(
        binary_rank_df,
        worker_column="WorkerId", 
        label_column="Binary Rank Response Left Image is Greater",
        crowdkit_grouping_columns = ['Left Neural Network Model', 'Right Neural Network Model', 'Item Title Index', 'Item Title', 'Item Type', 'Country'],
        binary_rank_reconstruction_grouping_columns=['Item Title', 'Country'],
        crowdkit_model='mmsr',
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
    print(f'Running Crowdkit {crowdkit_model}')
    # print the columns of the DataFrame
    print(f'binary_rank_df.columns: {binary_rank_df.columns}')
    # print the unique workerIds in the binary_rank_df
    print(f'Unique workers: binary_rank_df[worker_column].unique(): {len(binary_rank_df[worker_column].unique())}')
    # join crowdkit_grouping_columns with a dash and replace space with dash
    crowdkit_grouping_columns_str = '-'.join(crowdkit_grouping_columns).replace(' ', '-')
    # join binary_rank_reconstruction_grouping_columns with a dash
    binary_rank_reconstruction_grouping_columns_str = '-'.join(binary_rank_reconstruction_grouping_columns).replace(' ', '-')

    # convert the binary_rank_df to the format required by crowdkit with 'task', 'worker', and 'label' columns
    task_worker_label_df, table_restore_metadata = binary_rank.convert_table_to_crowdkit_classification_format(
        binary_rank_df, worker_column=worker_column, label_column=label_column,
        task_columns=crowdkit_grouping_columns
    )


    #########################################
    # MMSR
    # print table restore metadata
    # Assuming you have 'WorkerId' and 'Binary Rank Response Left Image is Greater' columns
    if seed is not None:
        np.random.seed(seed)

    # Create the MMSR model https://toloka.ai/docs/crowd-kit/reference/crowdkit.aggregation.classification.m_msr.MMSR/
    mmsr = MMSR(
        n_iter=10000,
        tol=1e-10,
        n_workers=table_restore_metadata['n_workers'],
        n_tasks=table_restore_metadata['n_tasks'],
        n_labels=table_restore_metadata['n_labels'],  # Assuming binary responses
        workers_mapping=table_restore_metadata['workers_mapping'],
        tasks_mapping=table_restore_metadata['tasks_mapping'],
        labels_mapping=table_restore_metadata['labels_mapping'],
        random_state=seed
    )
    print(f'st2_int.shape: {task_worker_label_df.shape} \ntask_worker_label_df:\n{task_worker_label_df}')
    task_worker_label_df.to_csv("task_worker_label_df.csv")

    print(f'Running CrowdKit Optimization MMSR.fit_predict()')
    # Fit the model and predict worker skills
    results_df = mmsr.fit_predict(task_worker_label_df)
    # set the skills index name to WorkerId
    mmsr.skills_.index.name = 'WorkerId'
    print(f'Finished CrowdKit Optimization MMSR.fit_predict(), Results:')
    print(results_df)
    # save results to a file
    results_df.to_csv(f"{crowdkit_model}_results-{crowdkit_grouping_columns_str}.csv")

    #########################################
    # Extract the worker consensus alignment (aka "skills") (this can be visualized with plot_consensus_alignment_skills.py)
    worker_skills = worker_skills_add_country_and_save_to_csv(mmsr.skills_, binary_rank_df, 
                                                              csv_filename=f"{crowdkit_model}_participant_consensus_alignment-{crowdkit_grouping_columns_str}.csv", 
                                                              plot_filename=f"{crowdkit_model}_participant_consensus_alignment_plot-{crowdkit_grouping_columns_str}",
                                                              xlim=(-3.5, 3.5))

    #########################################
    # visualize the mmsr observation matrix
    observation_matrix = mmsr._observation_matrix
    # Visualizing the Observation Matrix as a Heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(observation_matrix, cmap='viridis', fmt='.2f')
    # sns.heatmap(observation_matrix, cmap='viridis', fmt='.2f', xticklabels=table_restore_metadata['tasks_list'])
    plt.title('Observation Matrix Heatmap')
    plt.ylabel('Participant ID')
    # Make the Y label include the Model and binary_rank_reconstruction_grouping_columns combinations in human readable form
    label = f"Rank Comparisons Across Combos of {', '.join(['Model'] + binary_rank_reconstruction_grouping_columns)}"
    plt.xlabel(label)

    # plt.show()
    # save the observation matrix and plot to a file
    np.savetxt(f"{crowdkit_model}_observation_matrix-{crowdkit_grouping_columns_str}.csv", observation_matrix, delimiter=",")
    plt.savefig(f"{crowdkit_model}_observation_matrix-{crowdkit_grouping_columns_str}.png")
    plt.savefig(f"{crowdkit_model}_observation_matrix-{crowdkit_grouping_columns_str}.pdf")

    #########################################
    # visualize the mmsr covariance matrix
    covariance_matrix = mmsr._covariation_matrix
    # Visualizing the Covariance Matrix as a Heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(covariance_matrix, cmap='coolwarm', fmt='.2f')
    # sns.heatmap(covariance_matrix, cmap='coolwarm', annot=True, fmt='.2f')
    plt.title('Participant MMSR Covariance Matrix Heatmap')
    plt.xlabel('Participant ID')
    plt.ylabel('Participant ID')
    # plt.show()
    # save the covariance matrix and plot to a file
    np.savetxt(f"{crowdkit_model}_covariance_matrix-{crowdkit_grouping_columns_str}.csv", covariance_matrix, delimiter=",")
    plt.savefig(f"{crowdkit_model}_covariance_matrix-{crowdkit_grouping_columns_str}.png")
    plt.savefig(f"{crowdkit_model}_covariance_matrix-{crowdkit_grouping_columns_str}.pdf")
    
    #########################################
    # calculate the mmsr rank results
    results_df = binary_rank.restore_from_crowdkit_classification_format(results_df, table_restore_metadata)
    print(f'Finished CrowdKit Optimization MMSR.fit_predict(), restore_from_crowdkit_format() results_df:\n{results_df}')
    results_df.to_csv(f"{crowdkit_model}_results_restored-{binary_rank_reconstruction_grouping_columns_str}.csv")
    # print the columns of results_df
    print(f'results_df.columns: {results_df.columns}')
    # print results_df
    print(f'results_df:\n{results_df}')
    rank_results = binary_rank.reconstruct_ranking(results_df, binary_rank_reconstruction_grouping_columns)
    print(f'Finished CrowdKit Optimization MMSR.fit_predict(), reconstruct_ranking() rank_results:\n{rank_results}')
    rank_results.to_csv(f"{crowdkit_model}_rank_results-{binary_rank_reconstruction_grouping_columns_str}.csv")
    # save a plot of the model rank results
    hue = None
    if 'Country' in binary_rank_reconstruction_grouping_columns:
        hue = 'Country'
    if 'Item Title' in binary_rank_reconstruction_grouping_columns:
        hue = 'Item Title'
    
    if binary_rank_reconstruction_grouping_columns:
        title = f"Model Rankings by {', '.join(binary_rank_reconstruction_grouping_columns)}"
    else:
        title = f"Model Rankings Across All Responses"
    plot_ranking.strip_plot_rank(rank_results, x='Neural Network Model', y='Rank', hue=hue, filename=f"{crowdkit_model}_rank_results-{binary_rank_reconstruction_grouping_columns_str}", show_plot=False, title=title)

    return rank_results, results_df, worker_skills


def plot_binary_comparisons(df, network_models):
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
    for i in range(len(network_models)):
        for j in range(i-1, -1, -1):
            df_grouped = df_grouped[~((df_grouped["Left Neural Network Model"] == network_models[i]) \
                                    & (df_grouped["Right Neural Network Model"] == network_models[j]))]

    countries = df_grouped['Country'].unique()
    item_titles = df_grouped['Item Title'].unique()

    def plot_barchart(df_grouped_subset, ax):
        sns.barplot(
                ax=ax,
                data=df_grouped_subset, 
                x="Left Neural Network Model", 
                y='Binary Rank Response Left Image is Greater', 
                hue="Right Neural Network Model",
                order=network_models[:-1],
                hue_order=network_models[1:])
        ax.set_ylim(0, 100)
        ax.set_xlabel(None)
        ax.set_ylabel(None)
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
            
    axes[0,0].set_ylabel('% ranked lower (better)')
    plt.tight_layout()
    plt.savefig('binary_comparison_per_country.pdf')
    plt.savefig('binary_comparison_per_country.png')
    plt.close(fig)
    # plt.show()

    fig, axes = plt.subplots(1, len(item_titles), figsize=(len(item_titles)*4, 3))
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


def plot_violins(df, network_models):
    
    # df = df[df['Item Title'] == 'Offensiveness']
    # from plot_ranking import strip_plot_rank
    # strip_plot_rank(df, y="Response")


    item_titles = df['Item Title'].unique()
    fig, axes = plt.subplots(1, len(item_titles), figsize=(len(item_titles)*4, 4))
    for i in range(len(item_titles)):
        question = item_titles[i]
        df2 = df[df['Item Title'] == question].copy()
        df2['count'] = 1
        df2 = df2.groupby(["Neural Network Model", "Response"]).count().reset_index()

        df_pivot =  df2.pivot(columns="Neural Network Model", index="Response", values="count")

        # print(df_pivot.columns)
        # df_pivot = df_pivot.rename({
        #     "genericSD":"Stable Diffusion",
        #     "contrastive":"Self-Contrastive"}, axis='index', errors="raise")

        df_pivot = df_pivot[network_models]
        for mod in network_models:
            df_pivot[mod] /= df_pivot[mod].sum()
            df_pivot[mod] *= 100

        sns.heatmap(
            annot=True,
            fmt=".0f",
            data=df_pivot,
            cmap=sns.color_palette("rocket_r", as_cmap=True),
            ax=axes[i],
            cbar=False
        )
        axes[i].set_title(question)
        axes[i].set_xlabel(None)
        axes[i].set_ylabel(None)
    axes[0].set_ylabel("Ranking")
    plt.tight_layout()
    plt.savefig('heat_map_by_item.pdf')
    plt.savefig('heat_map_by_item.png')
    plt.close(fig)

    # create figure and axes
    fig, ax = plt.subplots()
    color = {"India":'orange', 'Korea':'blue', 'China':'green', 'Nigeria':'purple', 'Mexico':'red'}

    countries = set(df['Country'].unique())
    for country in countries:
        sns.violinplot(
            x="Neural Network Model", 
            y="Response", 
            hue="Country",
            data=df[df['Country'] == country],
            palette=color,
            # split=True,
            ax=ax,
            density_norm="count",
            common_norm=False,
            # saturation=0.75,
            inner=None,
            # inner='quartile',
            order=network_models,
            fill=False
        )
    df_median = df.groupby(['Neural Network Model'])['Response'].median().reset_index()
    sns.scatterplot(data=df_median, 
                    x='Neural Network Model', 
                    y="Response", 
                    s=100,
                    c='black',
                    markers=['X'],
                    style="Neural Network Model",
                    legend=False,
                    ax=ax)
    df_mean = df.groupby(['Neural Network Model'])['Response'].mean().reset_index()
    sns.scatterplot(data=df_mean, 
                    x='Neural Network Model', 
                    y="Response", 
                    s=100,
                    c='black',
                    markers=['s'],
                    style="Neural Network Model",
                    legend=False,
                    ax=ax)

    # Set transparancy for all violins
    for violin in ax.collections:
        violin.set_alpha(0.75)
    
    plt.ylabel('Rank')
    plt.yticks([1,2,3,4])
    sns.move_legend(ax, "upper center", bbox_to_anchor=(1, 1))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # ax.spines['bottom'].set_visible(False)
    # ax.spines['left'].set_visible(False)
    plt.tight_layout()

    plt.savefig('violin_plot_by_country.pdf')
    plt.savefig('violin_plot_by_country.png')
    plt.close(fig)
    # plt.show()


    fig, ax = plt.subplots()
    sns.violinplot(
        x="Neural Network Model", 
        y="Response", 
        # hue="Country",
        data=df,
        # palette=color,#[country],
        # split=True,
        ax=ax,
        density_norm="count",
        common_norm=False,
        # saturation=0.75,
        inner=None,
        # inner='quartile',
        order=network_models,
        # fill=False
    )
    df_median = df.groupby(['Neural Network Model'])['Response'].median().reset_index()
    sns.scatterplot(data=df_median, 
                    x='Neural Network Model', 
                    y="Response", 
                    s=100,
                    c='black',
                    markers=['X'],
                    style="Neural Network Model",
                    legend=False,
                    ax=ax)
    df_mean = df.groupby(['Neural Network Model'])['Response'].mean().reset_index()
    sns.scatterplot(data=df_mean, 
                    x='Neural Network Model', 
                    y="Response", 
                    s=100,
                    c='black',
                    markers=['s'],
                    style="Neural Network Model",
                    legend=False,
                    ax=ax)

    # Set transparancy for all violins
    for violin in ax.collections:
        # violin.set_alpha(0.25)
        violin.set_alpha(0.75)
    
    plt.ylabel('Rank')
    plt.yticks([1,2,3,4])
    # sns.move_legend(ax, "upper center", bbox_to_anchor=(1, 1))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # ax.spines['bottom'].set_visible(False)
    # ax.spines['left'].set_visible(False)
    plt.tight_layout()

    plt.savefig('violin_plot.pdf')
    plt.savefig('violin_plot.png')
    plt.close(fig)
    

def statistical_analysis(df, network_models, load_existing=False, seed=None):
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
        "WorkerId": ["nunique", "count"]
    }

    # Perform aggregation and reset the index
    aggregated_df = grouped.agg(aggregation_functions).reset_index()

    # Save the aggregated DataFrame to a CSV file
    aggregated_df.to_csv("aggregated_statistical_output_all_data.csv", index=False)

    ####################
    # Per country stats
    ####################
    # Group the DataFrame by "Neural Network Model," "Country," and "Item Title"
    nci_group_names = ["Neural Network Model", "Country", "Item Title"]
    grouped = df.groupby(nci_group_names)

    # Define the aggregation functions you want to apply
    aggregation_functions = {
        "Response": ["count", "median", "min", "max", "sem", "mean"],
        "WorkerId": ["nunique", "count"],
        "Country": ["nunique", "count"]
    }

    # Perform aggregation and reset the index
    aggregated_df_per_country = grouped.agg(aggregation_functions).reset_index()

    # Save the aggregated DataFrame to a CSV file
    aggregated_df_per_country.to_csv("aggregated_statistical_output_by_country.csv", index=False)

    plot_violins(df, network_models)

    #########################################
    # Inter Rater Reliability (IRR)

    # Kendall's W
    # kendalls_w = assess_kendalls_w(df, crowdkit_grouping_columns=['Item Title', 'Country', 'Source CSV Row Index'], item_grouping_columns=['Item Title', 'Country', 'Source CSV Row Index'])

    ####################
    # Binary Rank Stats
    ####################
    if load_existing:
        # load existing binary_rank_df to save processing time, the downside is that it won't be updated if the binary_rank.py code changes
        binary_rank_df = pd.read_csv("statistical_output_binary_rank.csv")
    else:
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

    # Aggregate the ranks based on the groupings and 'agg_label' column
    aggregation_functions = {
        'Left Binary Rank Image': ['nunique', 'count'],
        'Right Binary Rank Image': ['nunique', 'count'],
        'WorkerId': ['nunique', 'count'],
        'Country': ['nunique', 'count'],
        'Binary Rank Response Left Image is Greater': ['nunique', 'count', 'mean', 'median', 'min', 'max', 'sem']
    }
    agg_df = binary_rank_df.agg(aggregation_functions).reset_index()
    agg_df.to_csv(f"aggregated_statistical_output_binary.csv")

    nci_group_names = ["Left Neural Network Model", "Right Neural Network Model", "Country", "Item Title"]
    agg_df = binary_rank_df.groupby(nci_group_names).agg(aggregation_functions).reset_index()
    agg_df.to_csv("aggregated_statistical_output_binary_by_country.csv", index=False)
    nci_group_names += ['Left Binary Rank Image', 'Right Binary Rank Image']
    agg_df = binary_rank_df.groupby(nci_group_names).agg(aggregation_functions).reset_index()
    agg_df.to_csv("aggregated_statistical_output_binary_by_country_and_image.csv", index=False)

    # compute reliability metrics
    reliability_metrics(binary_rank_df.copy())

    # Assess worker responses binary rank with noisy bradley terry
    assess_worker_binary_responses_noisy_bradley_terry(binary_rank_df,seed=seed)

    # ranking per country and per question
    rank_results_ci, results_df_ci, worker_skills_ci = assess_worker_responses(binary_rank_df, seed=seed)
    # ranking per country
    rank_results, results_df, worker_skills = assess_worker_responses(binary_rank_df, crowdkit_grouping_columns=['Left Neural Network Model', 'Right Neural Network Model', 'Item Title Index', 'Country'], binary_rank_reconstruction_grouping_columns=['Country'], seed=seed)
    # ranking per question
    rank_results, results_df, worker_skills = assess_worker_responses(binary_rank_df, crowdkit_grouping_columns=['Left Neural Network Model', 'Right Neural Network Model', 'Item Title Index', 'Item Title', 'Item Type'], binary_rank_reconstruction_grouping_columns=['Item Title'], seed=seed)
    # overall ranking (across all countries and questions)
    rank_results, results_df, worker_skills = assess_worker_responses(binary_rank_df, crowdkit_grouping_columns=['Left Neural Network Model', 'Right Neural Network Model','Item Type'], binary_rank_reconstruction_grouping_columns=[], seed=seed)

    plot_binary_comparisons(binary_rank_df.copy(), network_models)

    # Additional statistical analyses can be added here, save results to a file, and visualize them

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


def get_response_rows_per_image(df):
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
    df = get_response_rows_per_image(df)

    # validate the processing and data organization for expected consistency
    check_column_group_for_consistent_values_in_another_column(df, columns_to_group=['Source CSV File','Source CSV Row Index'], columns_to_match=["WorkerId"])
    
    return df


def rename_throughout_table(df, network_models=None, remap_dict=None, rename_substring=False):
    """
    Renames column titles, row titles, and values in a DataFrame based on a provided mapping dictionary.

    Parameters:
    df (pandas.DataFrame): The DataFrame to be renamed.
    remap_dict (dict): A dictionary mapping the original names to the new names.
    rename_substring (bool): If True, replace substrings that match the keys in remap_dict with the corresponding values.

    Returns:
    pandas.DataFrame: The renamed DataFrame.
    """
    if remap_dict is not None:
        # Remap column titles
        df.rename(columns=remap_dict, inplace=True)

        # Remap row titles
        df.rename(index=remap_dict, inplace=True)

        # Remap values
        if rename_substring:
            df.replace(remap_dict, inplace=True, regex=True)
        else:
            df.replace(remap_dict, inplace=True)

        if network_models is not None:
            # rename the network models list using the remap_dict
            network_models = [remap_dict.get(model, model) for model in network_models]

    return df, network_models


def test(debug=False):
    """ Small tests of the functions in this file.
    """
    # baseline krippendorf test
    val = alpha_krippendorff(pd.DataFrame.from_records([
        {'task': 'X', 'worker': 'A', 'label': 'Yes'},
        {'task': 'X', 'worker': 'B', 'label': 'Yes'},
        {'task': 'Y', 'worker': 'A', 'label': 'No'},
        {'task': 'Y', 'worker': 'B', 'label': 'No'},
        {'task': 'Z', 'worker': 'A', 'label': 'Yes'},
        {'task': 'Z', 'worker': 'B', 'label': 'No'},
    ]))
    # print the alpha 
    if debug:
        print(f'alpha_krippendorf basic test: {val}')
    # print a warning if it isn't about 0.4444444444444444
    if val != 0.4444444444444444:
        raise ValueError(f'alpha_krippendorff() returned {val} instead of 0.4444444444444444')
        # Additional Test Scenario
    baseline_scenario = {
        'WorkerId': [1, 2, 1, 2, 1, 2],
        'Binary Rank Response Left Image is Greater': [True, True, False, False, True, False],
        'Item Title': ['X', 'X', 'Y', 'Y', 'Z', 'Z'],
        'Country': ['C', 'C', 'C', 'C', 'C', 'C']
    }
    binary_rank_df_baseline_scenario = pd.DataFrame(baseline_scenario)
        # Maximum Agreement Scenario
    data_max_agreement = {
        'WorkerId': [1, 2, 3, 4],
        'Binary Rank Response Left Image is Greater': [True, True, True, True],
        'Item Title': ['A', 'A', 'B', 'B'],
        'Country': ['X', 'X', 'Y', 'Y']
    }
    binary_rank_df_max_agreement = pd.DataFrame(data_max_agreement)
    
    # Minimum Agreement Scenario
    data_min_agreement = {
        'WorkerId': [1, 2, 3, 4],
        'Binary Rank Response Left Image is Greater': [True, False, True, False],
        'Item Title': ['A', 'A', 'B', 'B'],
        'Country': ['X', 'X', 'Y', 'Y']
    }
    binary_rank_df_min_agreement = pd.DataFrame(data_min_agreement)
    
    # Intermediate Agreement Scenario
    data_intermediate_agreement = {
        'WorkerId':                                   [1, 2, 3, 1, 2, 3, 1, 2, 3],
        'Binary Rank Response Left Image is Greater': [True, True, True, True, True, False, False, False, False],
        'Item Title':                                 ['A', 'A', 'A', 'B', 'B', 'B', 'C', 'C', 'C'],
        'Country':                                    ['X', 'X', 'X', 'X', 'X', 'X', 'X', 'X', 'X']
    }
    binary_rank_df_intermediate_agreement = pd.DataFrame(data_intermediate_agreement)
    
    # Intermediate Agreement Scenario
    data_intermediate_agreement2 = {
        'WorkerId':                                   [1, 2, 3, 1, 2, 3, 1, 2, 3, 4, 5],
        'Binary Rank Response Left Image is Greater': [False, True, True, True, True, True, True, True, True, True, True],
        'Item Title':                                 ['A', 'A', 'A', 'B', 'B', 'B', 'C', 'C', 'C', 'A', 'A'],
        'Country':                                    ['X', 'X', 'X', 'X', 'X', 'X', 'X', 'X', 'X', 'Y', 'Y']
    }
    binary_rank_df_intermediate_agreement2 = pd.DataFrame(data_intermediate_agreement2)
    
    # Test each scenario
    scenarios = {
        'Approx 0.444444 Agreeement': binary_rank_df_baseline_scenario,
        'Maximum Agreement': binary_rank_df_max_agreement,
        'Minimum Agreement': binary_rank_df_min_agreement,
        'Intermediate Agreement': binary_rank_df_intermediate_agreement,
        'Intermediate Agreement 2': binary_rank_df_intermediate_agreement2
    }
    
    crowdkit_grouping_columns = ['Item Title', 'Country']
    reliability_grouping_columns = ['Country']  # Adjust this to your actual grouping columns
    
    for scenario_name, scenario_df in scenarios.items():
        if debug:
            print(f"\nRunning test scenario: {scenario_name}")
        result = reliability_metrics(scenario_df, crowdkit_grouping_columns=crowdkit_grouping_columns, reliability_grouping_columns=reliability_grouping_columns, filename=False)
        if debug:
            print(f"Result for {scenario_name}:")
            print(result)
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
    df = get_response_rows_per_image(df)
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
    parser.add_argument("--response_results", type=str, default="m3c_cvpr_results_11_16", help="Path to the file or folder containing CSV files with Amazon Mechanical Turk survey response results.")
    # parser.add_argument("--response_results", type=str, default="Batch_393773_batch_results.csv", help="Path to the file or folder containing CSV files with Amazon Mechanical Turk survey response results.")
    parser.add_argument("--survey_items_file", type=str, default="human_survey_items.csv", help="Path to the human_survey_items.csv file")
    parser.add_argument("--network_models", type=str, nargs='+', default=["contrastive", "positive", "baseline", "genericSD"], help="List of neural network model names in the order they should be plotted.")
    # parser.add_argument("--network_models", type=str, nargs='+', default=["baseline", "contrastive", "genericSD", "positive"], help="List of neural network model names")
    # note that you can specify no random seed by passing: --random_seed=None
    parser.add_argument("--remap_model_names", type=json.loads, 
                    default='{"contrastive":"SCoFT+MPC", "positive":"SCoFT+MP", "baseline":"SCoFT+M", "genericSD":"Stable Diffusion"}', 
                    help="Remap model names as specified with a json dictionary, or specify empty curly brackets {{}} for no remapping. Ensures plots use the final names occurring in the paper.")
    parser.add_argument("--random_seed", type=int, default=8827, nargs='?', help="Random seed for reproducibility, default is 8827, you can specify no random seed with --random_seed=None.")
    parser.add_argument("--load_existing", type=bool, default=False, help="Load existing statistical_analysis_input.csv if present, saves processing time, but may not be up to date. Default is True.")
    parser.add_argument("--worker_ids_to_remove", type=str, default=None, help="Path to text file where each line is a AMT worker ID of workers not to include in results.")

    args = parser.parse_args()

    test()

    network_models = args.network_models

    # Get the list of CSV files to process
    if os.path.isfile(args.response_results):
        csv_files = [args.response_results]
    elif os.path.isdir(args.response_results):
        csv_files = [os.path.join(args.response_results, filename) for filename in os.listdir(args.response_results) if filename.endswith(".csv")]

    if args.load_existing and os.path.exists('statistical_analysis_input.csv'):
        # Load the existing csv file, this saves processing time, but may not be up to date.
        combined_df = pd.read_csv('statistical_analysis_input.csv')
    else:
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
            df = process_survey_results_csv(csv_file, args.survey_items_file, network_models)
            dataframes.append(df)
        
        # Concatenate the DataFrames
        combined_df = pd.concat(dataframes, axis=0)
    
    if args.worker_ids_to_remove is not None:
        if os.path.exists(args.worker_ids_to_remove):
            with open(args.worker_ids_to_remove, 'r') as f:
                worker_ids = [line.strip() for line in f]
            # Remove workers one by one from dataframe to analyze how many responses they gave
            n = len(combined_df)
            for worker_id in worker_ids:
                combined_df = combined_df[combined_df["WorkerId"] != worker_id]
                print('Removing Worker ID {} \'s {} responses'.format(worker_id, n-len(combined_df)))
                n = len(combined_df)
        else:
            raise Exception("Could not find file of ids to remove." + args.worker_ids_to_remove)

    if args.remap_model_names:
        # Rename the model names throughout the table, e.g. "contrastive" -> "SCoFT+MPC". 
        # Note substrings will not be updated, such as folder names in the filename paths.
        combined_df, network_models = rename_throughout_table(combined_df, network_models, args.remap_model_names, rename_substring=False)

    aggregated_df = statistical_analysis(combined_df, network_models, args.load_existing, args.random_seed)

if __name__ == "__main__":
    main()
