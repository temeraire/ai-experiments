import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# Data for three elements
elements = ['Element A', 'Element B', 'Element C']
capabilities = [1, 2, 3]  # Low (1), Medium (2), High (3)
colors = ['red', 'yellow', 'green']

# Create a figure and a 3D axis
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Set up the positions along the x-axis
x_pos = np.arange(len(elements))

# Set y values to be constant (since we are visualizing in 2.5D on an XY plane)
y_pos = np.zeros_like(x_pos)

# Set z values to represent capabilities
z_values = capabilities

# Width and depth of bars
width = 0.4
depth = 0.4

# Plot the bars with different colors
ax.bar3d(x_pos, y_pos, np.zeros_like(z_values), width, depth, z_values, color=colors)

# Labels for axes
ax.set_xlabel('Elements')
ax.set_ylabel('')
ax.set_zlabel('Capability Level')

# Set x-tick labels to element names
ax.set_xticks(x_pos)
ax.set_xticklabels(elements)

plt.title('3D Visualization of Element Capabilities')
plt.show()