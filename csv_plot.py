import matplotlib.pyplot as plt
import matplotlib as mpl
import csv

plt.ion()
mpl.style.use('seaborn')


def plot(path):
    with open(path, 'r') as f:
        rows = csv.reader(f)
        for row in rows:
            print(row)


if __name__ == '__main__':
    path = 'results/test_16-13-04 22-12-2021.csv'

    plot(path)
