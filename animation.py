import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as mpatches
import numpy as np


class DynamicsAnimation(object):
    """An animated plot using matplotlib.animations.FuncAnimation."""
    def __init__(self, func, links, initial, kappa, T, optionals=None):
        self.func = func
        self.links = links
        self.initial = initial
        self.kappa = kappa
        self.T = T
        self.optionals = optionals
        self.N = len(self.initial)
        self.t_vals = np.arange(self.T)

        self.stream = self.data_stream()
        self.counter = 0

        self.fig, (self.ax1, self.ax2) = plt.subplots(nrows=2, sharex=True, gridspec_kw={"height_ratios": [1, 2.4]})
        plt.subplots_adjust(hspace=-0.2)
        self.ani = animation.FuncAnimation(self.fig, self.update, interval=5,
                                          init_func=self.setup_plot, blit=True)

        self.ax1.set_ylim(0.0, 1.0)
        self.ax1.set_title(f"SIS-dynamics", size=18)
        self.ax1.set_ylabel("Density", size=14)
        self.ax2.set_ylabel("Node", size=14)
        self.ax2.set_xlabel("Time", size=14)

    def setup_plot(self):
        """Initial drawing of the plot."""
        data = next(self.stream)
        linedata = [self.t_vals, np.mean(data, axis=1)]
        matdata = data.T
        self.img = [self.ax1.plot(linedata[0], linedata[1])[0], self.ax2.imshow(matdata)]
        values = [0, 1]
        colors = [self.img[-1].cmap(self.img[-1].norm(value)) for value in values]
        label_dict = {
            0: "Inactive",
            1: "Active",
        }
        patches = [mpatches.Patch(color=colors[i], label=label_dict[i].format(l=values[i]) ) for i in range(len(values))]
        # plt.legend(handles=patches, bbox_to_anchor=(1.01, 1), loc=2, borderaxespad=0., fontsize=12)
        return self.img

    def data_stream(self):
        """Animate SIS-dynamics"""
        data = np.zeros((self.T, self.N), dtype=np.uint8)
        data[0] = self.initial
        if self.optionals != None:
            while True:
                self.counter += 1
                if self.counter < self.T:
                    next_col = self.func(self.links, data[self.counter-1], 1, self.kappa, *self.optionals)[1]
                    data[self.counter] = next_col
                else:
                    next_col = self.func(self.links, data[-1], 1, self.kappa, *self.optionals)[1]
                    data = np.roll(data, shift=-1, axis=0)
                    data[-1] = next_col
                # plt.pause(1)
                yield data
        else:
            while True:
                self.counter += 1
                if self.counter < self.T:
                    next_col = self.func(self.links, data[self.counter-1], 1, self.kappa)[1]
                    data[self.counter] = next_col
                else:
                    next_col = self.func(self.links, data[-1], 1, self.kappa)[1]
                    data = np.roll(data, shift=-1, axis=0)
                    data[-1] = next_col
                # plt.pause(1)
                yield data

    def update(self, i):
        """Update the matrix plot."""
        data = next(self.stream)

        # for i in range(len(self.img)-1):
        self.img[0].set_data(self.t_vals, np.mean(data, axis=1))
        self.img[1].set_data(data.T)

        return self.img


if __name__ == "__main__":
    from paper import *
    M0 = 4
    levels = 8
    alpha = 2.5

    links = get_HMN(M0, levels, alpha)
    # # np.save("data/links", links)
    # links = np.load("data/links.npy")
    # # plot_HMN(links)
    # num_rep = 1000
    # initials = np.random.randint(0, 2, size=(num_rep, len(links)), dtype=np.uint8)
    # a_vals = np.ones(num_rep, dtype=np.uint64) * 100
    # b_vals = np.ones(num_rep, dtype=np.uint64) * 500
    # intervals = np.stack([a_vals, b_vals], axis=1)
    # delta = 4# Animate
    # initial = np.ones(len(links), dtype=np.uint8)
    initial = np.random.randint(0, 2, len(links), dtype=np.uint8)
    # initial = np.random.binomial(1, 0.9, len(links)).astype(np.uint8)
    # initial = np.append(np.ones(int(len(links) / 2)), np.zeros(int(len(links) / 2))).astype(np.uint8)
    for kappa in np.arange(0.1, 0.16, 0.01):
    # kappa = 0.1#5#25
    # T = int(len(links) * 3)
        T = 3 * len(links)
        # optionals = [0.1, 1]  # dt, mu for continuous, None for discrete
        # anim = DynamicsAnimation(run_dynamics_cont, links, initial, kappa, T, optionals)
        print(f"Kappa: {kappa}")
        anim = DynamicsAnimation(run_dynamics, links, initial, kappa, T)
        plt.show()
