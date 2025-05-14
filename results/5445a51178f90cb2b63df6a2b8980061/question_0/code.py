import pandas as pd
from insightbench.tools import plot_countplot, save_json, fix_fnames

# Load the dataset
data_path = '/Users/avramesh/Documents/ServiceNow/BIBench/insight-bench-ext/data/IBExt/correlation_analysis_injected.csv'
df = pd.read_csv(data_path)

# Count occurrences of each short description
short_description_counts = df['Short description'].value_counts().head(50)

# Prepare data for the plot
x_axis_data = short_description_counts.index.tolist()  # X-axis data: Short descriptions
y_axis_data = short_description_counts.values.tolist()  # Y-axis data: Counts

# Generate the count plot
plot_countplot(df, 'Short description', 'Most Common Short Descriptions of Incidents')

# Prepare stats for JSON
stats_data = {
    'name': "Short Description Counts",
    'description': "Counts of the most common short descriptions of incidents.",
    'value': short_description_counts.to_dict()
}

# Save stats JSON
save_json(stats_data, "stat")

# Prepare and save x_axis JSON
x_axis_json = {
    'name': "Short Descriptions",
    'description': "The most common short descriptions of incidents.",
    'value': x_axis_data
}
save_json(x_axis_json, "x_axis")

# Prepare and save y_axis JSON
y_axis_json = {
    'name': "Counts",
    'description': "The counts of the most common short descriptions.",
    'value': y_axis_data
}
save_json(y_axis_json, "y_axis")

# Fix filenames for saved plots and stats
fix_fnames()