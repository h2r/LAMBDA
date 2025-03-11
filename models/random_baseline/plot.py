import matplotlib.pyplot as plt

# Sample data: list of 5 numbers
data = [10.12, 9.18, 10.41, 7.88, 9.55]

# Setting up the plot
fig, ax = plt.subplots()
ax.bar(range(len(data)), data)

# Labeling the axes
ax.set_ylabel('Avg Accuracy (%)')
ax.set_xlabel('Trajectories')
ax.set_title('Avg Accuracy per Trajectory')

# Setting the x-ticks to correspond to your data
ax.set_xticks(range(len(data)))

# Saving the plot to a file instead of showing it interactively
plt.savefig('bar_plot.png')
