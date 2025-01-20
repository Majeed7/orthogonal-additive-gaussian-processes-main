import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np 
import seaborn as sns 

excel_path_results = Path(f'results/exp_syn/explanation_synthesized.xlsx')

time_table = False 
plot_performance = True

ideal_avg_rank = { 
    'Sine Log': 1.5,
    'Sine Cosine': 1.5,
    'Poly Sine': 1.5,
    'Squared Exponentials': 2,
    'Tanh Sine': 2,
    'Trigonometric Exponential': 2.5,
    'Exponential Hyperbolic': 2.5,
    'XOR': 3
    }

# Set general font size for readability
plt.rcParams.update({
    'font.size': 12,     # Set default font size
    'font.weight': 'bold',  # Set default font weight to bold
    'axes.labelweight': 'bold',  # Ensure the axis labels are bold
    'axes.titleweight': 'bold',  # Ensure the titles are bold
    'figure.titleweight': 'bold',  # Bold for suptitle if you use fig.suptitle()
    'xtick.labelsize': 9,  # Font size for X-tick labels
    'ytick.labelsize': 8,  # Font size for Y-tick labels
    'xtick.major.size': 5,  # Length of major ticks
    'ytick.major.size': 5,  # Length of major ticks
    'xtick.minor.size': 3,  # Length of minor ticks
    'ytick.minor.size': 3   # Length of minor ticks
})

xls = pd.ExcelFile(excel_path_results)
sheet_names = xls.sheet_names

num_sheets = len(xls.sheet_names)
# Adjust figure size for more room, and change layout to 2 rows, 4 columns for better visibility
fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(16, 10))  # Adjust as needed

# Flatten axes for easier iteration
axes = axes.flatten()

# Iterate through each sheet in the Excel file
palette = sns.color_palette("Set2", n_colors=len(sheet_names))
for i , (ax, sheet_name, color) in enumerate(zip(axes, xls.sheet_names, palette)):
    # Read sheet into DataFrame
    df = xls.parse(sheet_name)
    
    # Assuming each row is a method and the rest are results
    df_transposed = df.set_index(df.columns[0]).T
    mean_value = np.mean(df.values[:, 1:].astype(float))
    
    # Plotting boxplot using seaborn on a subplot axis
    sns.boxplot(width=.8, data=df_transposed, ax=ax, palette=[color]).set(xlabel=' ')

    # Add a horizontal line for mean value
    ax.axhline(y=ideal_avg_rank[sheet_name], color='red', linestyle=':', linewidth=2, label='Mean')
    
    ax.set_title(f'Synthesized dataset {sheet_names.index(sheet_name) + 1}')
        # Add Y-axis label only for the first column (most left plot)
    if i % 4 == 0:  # Adjust for the number of columns in your grid layout
        ax.set_ylabel('Average Rank of Influential Features')
    # Optional: Add Y-axis label for clarity
    ax.tick_params(axis='x', rotation=45)  # Rotates the method names for better visibility
    ax.set_ylim(bottom=0)


fig.subplots_adjust(top=0.95, bottom=0.15, wspace=0.3, hspace=0.4)  # Adjust space

fig.savefig("results/exp_syn/exp_syn.png", dpi=500, format='png', bbox_inches='tight')

print("done!")
