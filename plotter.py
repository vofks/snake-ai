import csv
import matplotlib.pyplot as plt
from statistics import stdev, mean
from plot_helper import plot
import matplotlib.pyplot as plt

FILE_NAME = 'inference/linear2_512_256_inference_12-41-55 27-12-2021.csv'


def plot_eval(scores, mean_scores):
    plt.figure(figsize=(14, 10))
    plt.title('Evaluation')
    plt.xlabel('Episode')
    plt.ylabel('Score')

    if scores is not None:
        plt.plot(scores, color='#00a4f0')
    if mean_scores is not None:
        plt.plot(mean_scores, color='#FF9505')

    plt.legend(['Счёт агента за эпизод',
                'Среднее значение счёта'])

    plt.show()


def main():
    rewards = []
    scores = []
    frames = []
    bests = []

    with open(FILE_NAME, 'r') as f:
        datareader = csv.reader(f, delimiter=',', quotechar='|')
        for row in datareader:
            if len(rewards) < 100:
                rewards.append(int(row[1]))
                scores.append(int(row[2]))
                frames.append(int(row[3]))
                bests.append(int(row[4]))

    means = [0] * len(scores)
    for i in range(100, len(scores)):
        means[i] = mean(scores[i - 100:i])

    min_reward = min(rewards)
    max_reward = max(rewards)
    mean_reward = sum(rewards) / len(rewards)

    min_score = min(scores)
    max_score = max(scores)
    mean_score = sum(scores) / len(scores)
    stdev_score = stdev(scores)

    min_frames = min(frames)
    max_frames = max(frames)
    mean_frames = sum(frames) / len(frames)
    stdev_frames = stdev(frames)

    best_score = max(bests)

    print(f'Min reward: {min_reward}')
    print(f'Max reward: {max_reward}')
    print(f'Mean reward: {mean_reward}')

    print(f'Min score: {min_score}')
    print(f'Max score: {max_score}')
    print(f'Mean score: {mean_score}')
    print(f'Standard deviation: {stdev_score}')

    print(f'Min frames: {min_frames}')
    print(f'Max frames: {max_frames}')
    print(f'Mean frames: {mean_frames}')
    print(f'Standard deviation: {stdev_frames}')

    print(f'Best score: {best_score}')

    ''' plot(scores, means, legend=['Счёт агента за эпизод',
         'Среднее значение счёта за последние 100 эпизодов']) '''

    mean_vals = []
    total = 0
    for i in range(len(scores)):
        total += scores[i]
        mean_vals.append(total/(i + 1))

    plot_eval(scores, mean_vals)


if __name__ == '__main__':
    main()
    input("Press [enter] to continue.")
