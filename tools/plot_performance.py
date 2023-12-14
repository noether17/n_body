import argparse
import matplotlib.pyplot as plt
import numpy as numpy

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('filename')
    args = parser.parse_args()
    filename = args.filename

    data = []
    with open(filename, 'r') as input_file:
        for line in input_file:
            if line[0] == '#':
                keys = line[1:].split()
            else:
                data.append(line.split())
    data_dict = {}
    for entry in data:
        key = entry[0] + '_'
        if entry[2] == '-':
            key += entry[3]
        else:
            key += entry[2]
        if key not in data_dict:
            data_dict[key] = []
        data_dict[key].append([entry[1], entry[4]])
    plt.figure(figsize=(8, 6))
    for key in data_dict:
        if key.startswith('cpu'): color = 'b'
        elif key.startswith('gpu-w-copy'): color = 'y'
        else: color = 'g'
        key_number = int(key.split('_')[-1])
        if key_number == 1: marker = 'v'
        elif key_number == 2 or key_number == 32: marker = '+'
        elif key_number == 4 or key_number == 64: marker = 'o'
        elif key_number == 8 or key_number == 128: marker = 'x'
        else: marker = '^'
        linestyle = color + marker + '--'
        plt.loglog([float(x[0]) for x in data_dict[key]], [float(x[1]) for x in data_dict[key]], linestyle, label=key)
    plt.title('Performance Comparion Between CPU and GPU')
    plt.legend()
    plt.xlabel(keys[1])
    plt.ylabel(keys[4])
    plt.show()

if __name__ == '__main__':
    main()