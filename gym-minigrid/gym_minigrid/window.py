import sys
import numpy as np

# Only ask users to install matplotlib if they actually need it
try:
    import matplotlib.pyplot as plt
except:
    print('To display the environment in a window, please install matplotlib, eg:')
    print('pip3 install --user matplotlib')
    sys.exit(-1)

class Window:
    """
    Window to draw a gridworld instance using Matplotlib
    """

    def __init__(self, title):
        self.fig = None

        self.imshow_obj = None

        # Create the figure and axes
        self.fig = plt.figure(figsize=(12,8))
        grid = plt.GridSpec(3, 2, wspace=0.4, hspace=0.5, width_ratios=[1, 2])
        
        self.env_ax = self.fig.add_subplot(grid[:, 0])

        self.relevance_ax = self.fig.add_subplot(grid[0, 1])
        self.novelty_ax = self.fig.add_subplot(grid[1, 1])
        self.accountability_ax = self.fig.add_subplot(grid[2, 1])

        self.relevance_ax.set_xlabel('Appraisal 1 (Motivational Relevance): L1 Dist.')
        self.novelty_ax.set_xlabel('Appraisal 2 (Novelty/Unexpectedness): KL Div.')
        self.accountability_ax.set_xlabel('Appraisal 3 (Accountability)')

        # Show the env name in the window title
        self.fig.canvas.set_window_title(title)

        # Turn off x/y axis numbering/ticks
        self.env_ax.xaxis.set_ticks_position('none')
        self.env_ax.yaxis.set_ticks_position('none')
        _ = self.env_ax.set_yticklabels([])
        for ax in self.fig.get_axes():
            _ = ax.set_xticklabels([])

        plt.tight_layout()
        # Flag indicating the window was closed
        self.closed = False

        def close_handler(evt):
            self.closed = True

        self.fig.canvas.mpl_connect('close_event', close_handler)

    def clear(self):
        self.relevance_ax.lines = []
        self.novelty_ax.lines = []
        self.accountability_ax.lines = []

    def plot(self, data, xmin, xmax):
        self.clear()
        self.relevance_ax.plot(data[0], 'red')
        self.novelty_ax.plot(data[1], 'orange')
        self.accountability_ax.plot(data[2], 'purple')

    def show_img(self, img):
        """
        Show an image or update the image being shown
        """

        # Show the first image of the environment
        if self.imshow_obj is None:
            self.imshow_obj = self.env_ax.imshow(img, interpolation='bilinear')

        self.imshow_obj.set_data(img)
        self.fig.canvas.draw()

        # Let matplotlib process UI events
        # This is needed for interactive mode to work properly
        plt.pause(0.001)

    def set_caption(self, text):
        """
        Set/update the caption text below the image
        """

        self.env_ax.set_xlabel(f'Objective: {text}')

    def reg_key_handler(self, key_handler):
        """
        Register a keyboard event handler
        """

        # Keyboard handler
        self.fig.canvas.mpl_connect('key_press_event', key_handler)

    def show(self, block=True):
        """
        Show the window, and start an event loop
        """

        # If not blocking, trigger interactive mode
        if not block:
            plt.ion()

        # Show the plot
        # In non-interative mode, this enters the matplotlib event loop
        # In interactive mode, this call does not block
        plt.show()

    def close(self):
        """
        Close the window
        """

        plt.close()
