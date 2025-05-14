import pandas as pd
import insightbench.tools as tools

# Load the dataset
data_path = '/Users/avramesh/Documents/ServiceNow/BIBench/insight-bench-ext/data/IBExt/correlation_analysis_injected.csv'
df = pd.read_csv(data_path)

# Group by 'Assignment group' and calculate the mean business duration
# Grouping variable
assignment_group = 'Assignment group'
# Target variable for analysis
business_duration = 'Business duration'

# Calculate mean business duration for each assignment group
mean_duration = df.groupby(assignment_group)[business_duration].mean().reset_index()

# Prepare data for JSON files
# X-axis data for JSON
x_axis_data = {
    'name': "Assignment Groups",
    'description': "Different assignment groups with their corresponding mean business duration.",
    'value': mean_duration[assignment_group].tolist()
}

# Y-axis data for JSON
y_axis_data = {
    'name': "Mean Business Duration",
    'description': "Mean business duration for each assignment group.",
    'value': mean_duration[business_duration].tolist()
}

# Save the stats JSON file
stats_data = {
    'name': "Business Duration by Assignment Group",
    'description': "This plot shows the variation of business duration across different assignment groups.",
    'value': mean_duration.to_dict(orient='records')
}

# Plotting the count plot
tools.plot_countplot(mean_duration, business_duration, "Business Duration Variation Across Assignment Groups")

# Save JSON files
tools.save_json(x_axis_data, "x_axis")
tools.save_json(y_axis_data, "y_axis")
tools.save_json(stats_data, "stat")

# Fix filenames for saved plots
tools.fix_fnames()