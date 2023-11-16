import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import argparse

def swarm_violin_plot(data, x="Consensus Alignment", y="Participant", hue="Country", filename="plot", orient='v', size=(6, 10), xlim=(-3.5, 3.5), ylim=None, hue_order=None, palette=None, show=False):
    """
    Creates a combined swarm and violin plot of the given data.

    Parameters:
        data (DataFrame): The data to plot.
        x (str, optional): The column in `data` to use for the x-axis. Defaults to 'Consensus Alignment'.
        y (str, optional): The column in `data` to use for the y-axis. Defaults to 'Participant'.
        hue (str, optional): The column in `data` to use for color encoding. Defaults to 'Country'.
        filename (str, optional): The name of the output file. Defaults to 'plot'.
        orient (str, optional): The orientation of the plot ('v' for vertical, 'h' for horizontal). Defaults to 'v'.
        size (tuple, optional): The size of the plot. Defaults to (6, 10).
        xlim (tuple, optional): The limits for the x-axis. Defaults to (-3.5, 3.5).
        ylim (tuple, optional): The limits for the y-axis. Defaults to None.
        hue_order (list, optional): The order to plot the hue levels. Defaults to None.
        palette (list, optional): A list of colors to use for the hue levels. Defaults to None.
        show (bool, optional): Whether to display the plot. Defaults to False.

    Returns:
        fig (Figure): The created matplotlib Figure object.
        ax (Axes): The created matplotlib Axes object.
    """
    fig, ax = plt.subplots(figsize=size)

    ax.set_xlim(xlim)
    if orient == 'v':
        ax.set_ylim(xlim)
        ax.set_xlim(ylim)
    else:
        if ylim is not None:
            ax.set_ylim(ylim)

    # Define colors and shapes based on country flags for color-blind friendliness
    country_markers = {
        'China': ('#DE2910', 'o'),  # China's flag color and circle marker
        'India': ('#FF9933', 's'),  # India's flag color and square marker
        'Mexico': ('#006847', '^'),  # Mexico's flag color and triangle-up marker
        'Korea': ('#003478', 'D'),  # Korea's flag color and diamond marker
        'Nigeria': ('#008751', 'P')  # Nigeria's flag color and plus marker
    }

    # Create violin plot with distinct colors
    sns.violinplot(data=data, y=x, ax=ax, fill=True, linewidth=1, cut=0, hue_order=hue_order, palette=palette)
    legend_handles = []
    for country, (color, marker) in country_markers.items():
        legend_handles.append(plt.Line2D([0], [0], marker=marker, color='w', markerfacecolor=color, label=country))  # Add markers for legend

    # Create swarm plot with country-based colors and shapes
    swarm = sns.swarmplot(data=data, y=x, ax=ax, size=15, edgecolor="black", hue=hue, hue_order=hue_order, palette=[color for color, _ in country_markers.values()], marker='o')
    handles, labels = swarm.get_legend_handles_labels()
    handles.extend(legend_handles)
    ax.legend(handles=handles, labels=labels, loc="upper right", title=hue)

    ax.set_xlabel(y if orient == 'v' else x)
    ax.set_ylabel(x if orient == 'v' else y)
    ax.set_title("Consensus Alignment")

    fig.savefig(f"{filename}.png", bbox_inches="tight")
    fig.savefig(f"{filename}.pdf", bbox_inches="tight")

    if show:
        plt.show()

    return fig, ax

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type=str, default="mmsr_worker_skills-Left-Neural-Network-Model-Right-Neural-Network-Model-Item-Title-Index-Item-Title-Item-Type.csv", help="The name of the input file")
    return parser.parse_args()

def main():
    args = parse_args()
    data = pd.read_csv(args.input)
    data.columns = data.columns.str.replace('worker', 'Participant')

    # Improving color selection and visual appearance
    custom_palette = sns.color_palette("Set2")  # Use a predefined color palette
    # Improving color selection and visual appearance
    sns.set(style="whitegrid")  # Set seaborn style to improve appearance

    swarm_violin_plot(data, filename="consensus_alignment_plot", orient='v', size=(10, 6), palette=custom_palette)
    print("Consensus alignment plot saved.")

if __name__ == "__main__":
    main()
