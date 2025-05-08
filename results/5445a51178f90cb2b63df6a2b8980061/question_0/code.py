import pandas as pd
import numpy as np
from insightbench.tools import generate_wordcloud, plot_countplot, plot_lines, save_json, fix_fnames

# Load the dataset
# Path to the CSV file
data_path = '/Users/avramesh/Documents/ServiceNow/BIBench/insight-bench-ext/data/notebooks/csvs/flag-1.csv'

# Read the CSV file into a DataFrame
df = pd.read_csv(data_path)

# Convert 'opened_at' and 'closed_at' to datetime
# Columns for opening and closing timestamps
df['opened_at'] = pd.to_datetime(df['opened_at'])
df['closed_at'] = pd.to_datetime(df['closed_at'])

# Calculate the time taken to resolve issues
# New column for time taken in hours
df['time_taken'] = (df['closed_at'] - df['opened_at']).dt.total_seconds() / 3600

# Group by category and calculate the average time taken
# Grouping by 'category' and calculating mean of 'time_taken'
avg_time_per_category = df.groupby('category')['time_taken'].mean().reset_index()

# Prepare data for plotting
# X-axis data (categories)
x_axis_data = avg_time_per_category['category'].tolist()
# Y-axis data (average time taken)
y_axis_data = avg_time_per_category['time_taken'].tolist()

# Plot the average time taken to resolve issues across different categories
# Plot title
plot_title = 'Average Time Taken to Resolve Issues by Category'
# Create a line plot
plot_lines(avg_time_per_category, 'category', ['time_taken'], plot_title)

# Prepare stats for JSON
# Stats data dictionary
stats_data = {
    'name': "Average Time Taken to Resolve Issues",
    'description': "Average time taken to resolve issues across different categories.",
    'value': avg_time_per_category.to_dict(orient='records')
}

# Save stats JSON
save_json(stats_data, "stat")

# Prepare x_axis JSON
# X-axis data dictionary
x_axis_json = {
    'name': "Categories",
    'description': "Different categories of issues.",
    'value': x_axis_data[:50]  # Limit to 50 entries
}
# Save x_axis JSON
save_json(x_axis_json, "x_axis")

# Prepare y_axis JSON
# Y-axis data dictionary
y_axis_json = {
    'name': "Average Time Taken (hours)",
    'description': "Average time taken to resolve issues in hours.",
    'value': y_axis_data[:50]  # Limit to 50 entries
}
# Save y_axis JSON
save_json(y_axis_json, "y_axis")

# Fix filenames for saved plots and stats
fix_fnames()