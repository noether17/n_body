import argparse
import matplotlib.pyplot as plt
import numpy as np
from numba import jit, prange, set_num_threads
import os

set_num_threads(4)

G = 6.67e-11 # gravitational constant

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('filename')
    args = parser.parse_args()
    filename = args.filename

    system_size = int(os.path.basename(filename).split('_')[1])

    raw_states = np.loadtxt(filename, delimiter=',', dtype=float)
    times = np.unique(raw_states[:, 0])

    frames = []
    for t in times:
        row_indices, = np.where(raw_states[:, 0] == t)
        current_frame = raw_states[row_indices[0]:row_indices[-1], 2:]
        frames.append(current_frame)
    frames = np.array(frames)

    energies = compute_energy_parallel(frames)

    plt.plot(times, energies)
    plt.xlabel("Time (s)")
    plt.ylabel("Energy (J)")
    plt.title(f"System Energy over Time ({system_size} particles)")
    plt.savefig(f"energy_{system_size}.png")

@jit(nopython=True, parallel=True)
def compute_energy_parallel(frames):
    energies = np.empty(frames.shape[0])
    for i in prange(frames.shape[0]):
        energies[i] = compute_energy(frames[i])
    return energies

@jit(nopython=True)
def compute_energy(frame):
    pos = frame[:, 0:3]
    vel = frame[:, 3:6]

    potential_energy = compute_potential_energy(pos)
    kinetic_energy = compute_kinetic_energy(vel)
    return potential_energy + kinetic_energy

@jit(nopython=True)
def compute_potential_energy(pos):
    n_particles = pos.shape[0]
    potential_energy = 0.0
    for i in np.arange(n_particles):
        for j in np.arange(i+1, n_particles):
            r = np.linalg.norm(pos[i] - pos[j])
            potential_energy -= 1.0 / r
    return G*potential_energy

@jit(nopython=True)
def compute_kinetic_energy(vel):
    return 0.5 * np.sum(vel*vel)

if __name__ == "__main__":
    main()