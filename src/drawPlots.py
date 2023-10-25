from matplotlib import pyplot
from pathlib import Path
import numpy as np

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

def plot_accuracy(num_epochs, accuracy, random, function, path, gan_type):
    print("Plot accuracy")

    epochs = np.arange(0, num_epochs, 5)

    fig, ax = pyplot.subplots()
    ax.plot(epochs, accuracy, 'r', label=gan_type)
    ax.plot(epochs, random, 'b', label='zufällig')
    ax.plot(epochs, function(epochs), 'g', label="Trend")
    ax.legend()
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Genauigkeit')
    Path("plots").mkdir(parents=False, exist_ok=True)
    pyplot.savefig(path)
    pyplot.close()