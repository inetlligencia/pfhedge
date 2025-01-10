import matplotlib.pyplot as plt

plt.style.use("college_ruled")  # If the file is named college_ruled.mplstyle

import matplotlib.pyplot as plt

plt.style.use("college_ruled")  # Load the custom style

# Now create your plot via pandas
df.plot()

# Then tweak the y-axis in code
ax = plt.gca()                # get current axes
ax.spines["left"].set_color("red")
ax.tick_params(axis="y", colors="red")

# Show
plt.show()


# Or read from direcory
plt.style.use("./styles/college_ruled.mplstyle")