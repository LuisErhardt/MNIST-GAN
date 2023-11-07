from matplotlib import pyplot
from pathlib import Path
import numpy as np

def plot_accuracy(num_epochs, accuracy, random, function, path, gan_type):
    print("Plot accuracy")

    epochs = np.arange(0, num_epochs, 5)

    fig, ax = pyplot.subplots()
    ax.set_ylim([0, 0.8])
    ax.plot(epochs, accuracy, 'r', label=gan_type)
    ax.plot(epochs, random, 'b', label='zufällig')
    if function:
        ax.plot(epochs, function(epochs), 'g', label="Trend")
    ax.legend()
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Genauigkeit')
    Path("plots").mkdir(parents=False, exist_ok=True)
    pyplot.savefig(path)
    pyplot.close()