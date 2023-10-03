from matplotlib import pyplot
from pathlib import Path

def plot_D_probabilities(epochs, probs_test, probs_train):
    print("Plot probabilitiy outputs of discriminator")
    fig, ax = pyplot.subplots()
    ax.plot(epochs, probs_test, 'r', label='test samples')
    ax.plot(epochs, probs_train, 'b', label='train samples')
    ax.legend()
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Probability output of D')
    filename = 'plots/probs.png'
    Path("plots").mkdir(parents=False, exist_ok=True)
    pyplot.savefig(filename)
    pyplot.close()

def plot_accuracy(epochs, accuracy, random):
    print("Plot accuracy")
    fig, ax = pyplot.subplots()
    ax.plot(epochs, accuracy, 'r', label='GAN')
    ax.plot(epochs, random, 'b', label='zufällig')
    ax.legend()
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Genauigkeit')
    filename = 'plots/gan_accuracy.png'
    Path("plots").mkdir(parents=False, exist_ok=True)
    pyplot.savefig(filename)
    pyplot.close()