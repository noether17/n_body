import argparse
import matplotlib.animation as anim
import matplotlib.pyplot as plt
import numpy as np

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('filename')
    args = parser.parse_args()
    filename = args.filename

    states = np.loadtxt(filename, delimiter=',', dtype=float)
    times = np.unique(states[:, 0])

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    ani = anim.FuncAnimation(fig, plot_frame, times, fargs=(states, ax))

    #writer = anim.PillowWriter(fps=15,
    #                           metadata=dict(artist='Me'),
    #                           bitrate=1800)
    writer = anim.PillowWriter()
    ani.save('animation.gif', writer=writer)

def plot_frame(t, states, ax):
    row_indices, = np.where(states[:, 0] == t)
    current_frame = states[row_indices[0]:row_indices[-1], 1:4]
    ax.clear()
    for point in current_frame:
        ax.scatter(point[0], point[1], point[2], marker=',', color='k', alpha=0.2)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_zlim(0, 10)
    return ax

if __name__ == "__main__":
    main()