import csv
import matplotlib.pyplot as plt
import matplotlib as mpl


mpl.style.use('seaborn')


def read(path):

    episodes = []
    rewards = []

    with open(path, 'r') as f:
        reader = csv.DictReader(f, delimiter=',')

        for row in reader:
            episodes.append(int(row['episode']))
            rewards.append(float(row['average_reward']))

    rewards.insert(0, 0)
    episodes.insert(0, 0)

    return episodes, rewards


if __name__ == '__main__':
    plt.figure(figsize=(20, 10))

    x, y = read('results/flatten_256_16-07-53 19-05-2022.csv')
    x1, y1 = read('results/deep_mind_vanilla_18-00-41 19-05-2022.csv')
    x2, y2 = read('results/new_double_linear_256_128_14-50-30 19-05-2022.csv')

    plt.title('Training')
    plt.xlabel('Episode')
    plt.ylabel('Reward')

    plt.ylim(0, 400)

    plt.plot(x, y, label='Модель №1')
    plt.plot(x1, y1, label='Модель №2')
    plt.plot(x2, y2, label='Двуслойный персептрон 256 128')

    plt.legend()

    plt.show()
