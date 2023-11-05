import matplotlib.pyplot as plt
import matplotlib.animation as anim
import numpy as np

class DummyAnimation:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.z = y

        fig = plt.figure()
        self.movie_ax = fig.add_subplot(111, projection='3d')
        self.movie_ax.scatter(x,y,y)
        self.highlight_scatter, = self.movie_ax.plot([x[0]], [y[0]], [y[0]], 'ro', markersize=10)  # Initialize the highlight scatterplot

        animation = anim.FuncAnimation(fig, self._update, frames=len(self.x), repeat=True, interval=500)
        plt.show()

    def _update(self, frame):
        x_val = self.x[frame]
        y_val = self.y[frame]
        z_val = self.z[frame]

        self.highlight_scatter.set_data(x_val, y_val)
        self.highlight_scatter.set_3d_properties(z_val)

    # You can add other updates or modifications to the plot using self.movie_ax as needed

        # To clear the highlight, you can set the offsets to NaN
        # self.highlight_scatter.set_offsets(np.column_stack((np.nan, np.nan)))

        # Rest of your animation logic using self.movie_ax

# Example usage
x_data = np.random.rand(10)
y_data = np.random.rand(10)

dummy_anim = DummyAnimation(x_data, y_data)
plt.show()
