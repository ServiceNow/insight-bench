import pandas as pd
from insightbench.tools import save_json, fix_fnames, plot_countplot

# Load the dataset
data_path = '/Users/avramesh/Documents/ServiceNow/BIBench/insight-bench-ext/data/notebooks/csvs/flag-1.csv'
df = pd.read_csv(data_path)

# Filter for unresolved issues
# This variable holds the DataFrame of unresolved issues
unresolved_issues = df[df['state'] == 'Closed']

# Group by location and category, counting unresolved issues
# This variable holds the count of unresolved issues by location and category
location_category_counts = unresolved_issues.groupby(['location', 'category']).size().reset_index(name='count')

# Find the location with the highest number of unresolved issues
# This variable holds the location with the maximum unresolved issues
max_unresolved_location = location_category_counts.loc[location_category_counts['count'].idxmax()]

# Prepare data for plotting
# This variable holds the plot data for the count of unresolved issues by category for the location
plot_data = location_category_counts[location_category_counts['location'] == max_unresolved_location['location']]

# Plot the count of unresolved issues by category
# This variable holds the title for the plot
plot_title = f"Unresolved Issues Count by Category in {max_unresolved_location['location']}"

# Save the plot
plot_countplot(plot_data, 'category', plot_title)

# Prepare stats for JSON
# This variable holds the stats data for the plot
stats_data = {
    'name': "Unresolved Issues Count by Category",
    'description': f"Counts of unresolved issues by category in {max_unresolved_location['location']}.",
    'value': plot_data.to_dict(orient='records')
}

# Save stats JSON
save_json(stats_data, "stat")

# Prepare x_axis data for JSON
# This variable holds the x-axis data for the plot
x_axis_data = {
    'name': "Categories",
    'description': "Categories of unresolved issues.",
    'value': plot_data['category'].tolist()[:50]  # Limit to 50 items
}

# Save x_axis JSON
save_json(x_axis_data, "x_axis")

# Prepare y_axis data for JSON
# This variable holds the y-axis data for the plot
y_axis_data = {
    'name': "Count of Unresolved Issues",
    'description': "Count of unresolved issues for each category.",
    'value': plot_data['count'].tolist()[:50]  # Limit to 50 items
}

# Save y_axis JSON
save_json(y_axis_data, "y_axis")

# Fix filenames for saved plots and stats
fix_fnames()