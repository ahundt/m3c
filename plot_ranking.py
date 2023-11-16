import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import argparse

def swarm_plot_rank(data, x="Neural Network Model", y="Rank", hue="Country", filename="plot", size=(6, 10), hue_order=None, palette=None):
    fig, ax = plt.subplots(figsize=size)

    # Define colors and shapes based on country flags for color-blind friendliness
    country_markers = {
        'China': ('#DE2910', 'o'),  # China's flag color and circle marker
        'India': ('#FF9933', 's'),  # India's flag color and square marker
        'Mexico': ('#006847', '^'),  # Mexico's flag color and triangle-up marker
        'Korea': ('#003478', 'D'),  # Korea's flag color and diamond marker
        'Nigeria': ('#008751', 'P')  # Nigeria's flag color and plus marker
    }

    # Create swarm plot with country-based colors and shapes
    swarm = sns.swarmplot(data=data, x=x, y=y, hue=hue, hue_order=hue_order, palette=[color for color, _ in country_markers.values()], marker='o')
    legend_handles = []
    for country, (color, marker) in country_markers.items():
        legend_handles.append(plt.Line2D([0], [0], marker=marker, color='w', markerfacecolor=color, label=country))  # Add markers for legend

    handles, labels = swarm.get_legend_handles_labels()
    handles.extend(legend_handles)
    ax.legend(handles=handles, labels=labels, loc="upper right", title=hue)

    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_title("Rank")

    fig.savefig(f"{filename}.png", bbox_inches="tight")
    fig.savefig(f"{filename}.pdf", bbox_inches="tight")

    return fig, ax

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type=str, default="mmsr_rank_results-Item-Title-Country.csv", help="The name of the input file")
    return parser.parse_args()

def main():
    args = parse_args()
    data = pd.read_csv(args.input)

    # Improving color selection and visual appearance
    sns.set(style="whitegrid")  # Set seaborn style to improve appearance

    swarm_plot_rank(data, filename="rank_swarm_plot", hue_order=["China", "India", "Mexico", "Korea", "Nigeria"])
    print("Rank swarm plot saved.")

if __name__ == "__main__":
    main()
