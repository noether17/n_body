import argparse
import math
import matplotlib.animation as anim
import matplotlib.pyplot as plt
import numpy as np
import os

# For updating progress
total_frames = 0
plotted_frames = 0

animation_time = 10.0 # total seconds
max_fps = 30 # maximum frames per second

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('filename')
    args = parser.parse_args()
    filename = args.filename

    system_size = int(os.path.basename(filename).split('_')[1])

    states = np.loadtxt(filename, delimiter=',', dtype=float)
    times = np.unique(states[:, 0])
    stride = int(math.ceil(times.size / (animation_time * max_fps))) # skip frames to reduce fps
    print(f"Using stride of {stride} frames to limit fps to {max_fps} (from {times.size} frames)")
    times = times[::stride]

    global total_frames
    total_frames = times.size

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    ani = anim.FuncAnimation(fig, plot_frame, times, fargs=(states, ax))

    #writer = anim.PillowWriter(fps=15,
    #                           metadata=dict(artist='Me'),
    #                           bitrate=1800)
    frames_per_second = total_frames / animation_time
    print(f"Saving animation at {frames_per_second:.2f} fps")
    writer = anim.PillowWriter(fps=frames_per_second)
    ani.save(f"animation_{system_size}_particles.gif", writer=writer)
    print() # newline after progress

def plot_frame(t, states, ax):
    global total_frames
    global plotted_frames
    progress_percent = (plotted_frames / total_frames) * 100.0
    print(f"\rProgress: {progress_percent:.2f} %", end='')
    plotted_frames += 1 # update progress count

    row_indices, = np.where(states[:, 0] == t)
    current_frame = states[row_indices[0]:row_indices[-1]+1, 2:5]
    ax.clear()
    for point in current_frame:
        ax.scatter(point[0], point[1], point[2], marker=',', color='k', alpha=0.2)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_zlim(0, 1)
    ax.set_title(f"Number of Particles: {current_frame.shape[0]}\nt = {t:.2f} s")
    return ax

if __name__ == "__main__":
    main()
