
import argparse
import pandas as pd
import numpy as np
import json
import os
import re
import binary_rank
import crowdkit
from crowdkit.aggregation import MMSR, NoisyBradleyTerry
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
import plot_ranking
import plot_consensus_alignment_skills
import inter_rater_reliability

def assess_kendalls_w(df,
        worker_column="WorkerId", 
        label_column="Response",
        crowdkit_grouping_columns=['Item Title', 'Country', 'Source CSV Row Index'],
        item_grouping_columns=['Item Title', 'Country', 'Source CSV Row Index']):
    
    # Print the columns
    print(f'kendalls_w_df.columns: {df.columns}')
    
    # Make a copy of the dataframe
    df = df.copy()
    
    # Group by 'crowdkit_grouping_columns' and convert 'Response' column to numeric
    df[label_column] = df.groupby(crowdkit_grouping_columns)[label_column].transform(lambda x: pd.to_numeric(x, errors='coerce'))
    
    # Pivot the dataframe to have items as rows and raters as columns
    pivot_df = df.pivot_table(index=item_grouping_columns, columns=worker_column, values=label_column)
    
    # Calculate Kendall's W
    kendalls_w_value = inter_rater_reliability.kendalls_w(pivot_df)
    print(f"Kendall's W: {kendalls_w_value}")
    
    # Save annotations to CSV file
    grouping_str = '-'.join(crowdkit_grouping_columns).replace(' ', '-')
    pivot_df.to_csv(f"kendalls_w_annotations-{grouping_str}.csv")

    return kendalls_w_value

df = pd.read_csv('statistical_analysis_input.csv')
assess_kendalls_w(df)