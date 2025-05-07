import pandas as pd
import numpy as np
from insightbench.tools import generate_wordcloud, plot_countplot, plot_lines, save_json, fix_fnames

# Load the dataset
# Path to the CSV file
data_path = '/Users/avramesh/Documents/ServiceNow/BIBench/insight-bench-ext/data/notebooks/csvs/flag-1.csv'

# Read the CSV file into a DataFrame
df = pd.read_csv(data_path)

# Convert 'opened_at' and 'closed_at' to datetime
# Convert the 'opened_at' and 'closed_at' columns to datetime format for time calculations
df['opened_at'] = pd.to_datetime(df['opened_at'])
df['closed_at'] = pd.to_datetime(df['closed_at'])

# Calculate the time taken to resolve issues
# Create a new column 'resolution_time' that calculates the difference between 'closed_at' and 'opened_at'
df['resolution_time'] = (df['closed_at'] - df['opened_at']).dt.total_seconds() / 3600  # Convert to hours

# Group by 'category' and calculate the average resolution time
# Group the DataFrame by 'category' and calculate the mean of 'resolution_time'
avg_resolution_time = df.groupby('category')['resolution_time'].mean().reset_index()

# Prepare data for plotting
# Prepare the x and y axis data for the plot
x_axis_data = avg_resolution_time['category'].tolist()
y_axis_data = avg_resolution_time['resolution_time'].tolist()

# Plot the average resolution time across categories
# Create a line plot of average resolution time by category
plot_title = 'Average Resolution Time by Category'
plot_lines(avg_resolution_time, 'category', ['resolution_time'], plot_title)

# Save stats data for the plot
# Create a dictionary to store the stats data
stats_data = {
    'name': "Average Resolution Time by Category",
    'description': "This shows the average time taken to resolve issues across different categories.",
    'value': avg_resolution_time.to_dict(orient='records')
}
save_json(stats_data, "stat")

# Save x-axis data
# Create a dictionary for x-axis data
x_axis_json = {
    'name': "Categories",
    'description': "Different categories of issues.",
    'value': x_axis_data[:50]  # Limit to 50 entries
}
save_json(x_axis_json, "x_axis")

# Save y-axis data
# Create a dictionary for y-axis data
y_axis_json = {
    'name': "Average Resolution Time (hours)",
    'description': "Average time taken to resolve issues in hours.",
    'value': y_axis_data[:50]  # Limit to 50 entries
}
save_json(y_axis_json, "y_axis")

# Fix filenames for saved plots and stats
fix_fnames()