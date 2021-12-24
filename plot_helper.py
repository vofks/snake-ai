import matplotlib.pyplot as plt
import matplotlib as mpl

plt.ion()
mpl.style.use('seaborn')


def plot(scores, mean_scores):
    plt.title('Training')
    plt.xlabel('Episode')
    plt.ylabel('Score')

    plt.plot(scores, color='#00a4f0')
    plt.plot(mean_scores, color='#003f5c')


def show_image(img):
    plt.figure()
    plt.imshow(img)
    plt.show()
