import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# Generate random data
np.random.seed(0)
num_points = 100
x = np.random.rand(num_points)
y = np.random.rand(num_points)

# Generate random sampling frequencies between 1 and 10
sample_counts = np.random.randint(1, 11, num_points)

# Create a discrete color map
unique_counts = np.unique(sample_counts)
num_colors = len(unique_counts)
colors = plt.cm.rainbow(np.linspace(0, 1, num_colors))
color_map = ListedColormap(colors)

# Create a dictionary to map sample counts to color indices
count_to_index = {count: index for index, count in enumerate(unique_counts)}

# Map sample counts to color indices
color_indices = np.array([count_to_index[count] for count in sample_counts])

# Create a scatter plot
plt.figure(figsize=(12, 8))
scatter = plt.scatter(x, y, c=color_indices, cmap=color_map, s=50)

# Create a custom legend
legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                              markerfacecolor=color_map(count_to_index[count]), 
                              markersize=10, label=f'{count} samples')
                   for count in unique_counts]
plt.legend(handles=legend_elements, title="Sample Counts", loc="center left", bbox_to_anchor=(1, 0.5))

# Set labels and title
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Scatter Plot with Discrete Color Mapping for Sampling Frequencies')

# Show the plot
plt.tight_layout()
plt.show()

# Print some information
print(f"Number of datapoints: {num_points}")
print(f"Unique sampling frequencies: {unique_counts}")
print(f"Color mapping: {count_to_index}")