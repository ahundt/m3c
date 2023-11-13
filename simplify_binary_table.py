import pandas as pd
import csv

def simplify_binary_rank_table(binary_rank_df):
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
    grouped = binary_rank_df.groupby([
        'Left Binary Rank Image', 'Right Binary Rank Image',
        'Left Neural Network Model', 'Right Neural Network Model',
        'Item Title Index', 'Item Title', 'Item Type',
        'Country', 'Input.prompt', 'Input.seed'
    ])

    # Function to aggregate the grouped columns into a single string
    def concatenate_columns(series):
        return ', '.join(series.astype(str))

    # Aggregate the columns into a single string
    simplified_table = grouped.agg({
        'WorkerId': 'first',
        'Binary Rank Response Left is Greater': 'first'
    }).reset_index()

    # Add the combined columns
    combined_columns = [
        'Left Binary Rank Image', 'Right Binary Rank Image',
        'Left Neural Network Model', 'Right Neural Network Model',
        'Item Title Index', 'Item Title', 'Item Type',
        'Country', 'Input.prompt', 'Input.seed'
    ]

    # double for loop to concatenate the titles and values of the combined columns into a single string per row
    st2 = pd.DataFrame()
    st2['task'] = simplified_table[combined_columns[0]]
    for col in combined_columns[1:]:
        st2['task'] = st2['task'] + '|' + simplified_table[col]
    st2['worker'] = simplified_table['WorkerId']
    st2['label'] = simplified_table['Binary Rank Response Left is Greater']

    # st2.to_csv('simplified_binary_rank_table.csv', index=False, quoting=csv.QUOTE_ALL)

    combined_columns = '|'.join(combined_columns)
    worker_column = 'WorkerId'
    label_column = 'Binary Rank Response Left is Greater'
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
    column_titles = [combined_columns, worker_column, label_column]

    # return the simplified int table, the maps, and the column names
    return st2_int, task_to_id, worker_to_id, label_to_id, column_titles


def restore_binary_rank_table(st2_int, task_to_id, worker_to_id, label_to_id, column_titles):
    """ Restore the binary rank table from the simplified table.
    """
    # restore the task column from integer ids to strings
    st2 = pd.DataFrame()
    st2['task'] = st2_int['task'].map({v: k for k, v in task_to_id.items()})
    st2['worker'] = st2_int['worker'].map({v: k for k, v in worker_to_id.items()})
    st2['label'] = st2_int['label'].map({v: k for k, v in label_to_id.items()})

    # split the task column into the original columns
    combined_columns = column_titles[0].split('|')
    for i, col in enumerate(combined_columns):
        st2[col] = st2['task'].apply(lambda x: x.split('|')[i])

    # drop the task column
    st2 = st2.drop(columns=['task'])

    # return the restored table
    return st2

if __name__ == "__main__":
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
        'Binary Rank Response Left is Greater': [True, False, True]
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
