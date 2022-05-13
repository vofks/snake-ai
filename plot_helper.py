import matplotlib.pyplot as plt
import matplotlib as mpl

plt.ion()
mpl.style.use('seaborn')


def plot(scores=None, mean_scores=None, legend=[]):
    plt.figure(figsize=(14, 10))
    plt.title('Training')
    plt.xlabel('Episode')
    plt.ylabel('Score')

    if scores is not None:
        plt.plot(scores, color='#00a4f0')
    if mean_scores is not None:
        plt.plot(mean_scores, color='#003f5c')

    plt.legend(legend)

    plt.show()


def show_image(img):
    plt.figure()
    plt.imshow(img)
    plt.show()
