import json
import torch
from matplotlib import pyplot
from pathlib import Path


def plot_D_probabilities(num_epochs, probs_test, probs_train):
    print("Plot probabilitiy outputs of discriminator")
    fig, ax = pyplot.subplots()
    ax.plot(range(num_epochs), probs_test, 'r', label='test samples')
    ax.plot(range(num_epochs), probs_train, 'b', label='train samples')
    ax.legend()
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Probability output of D')
    filename = 'plots/probs.png'
    Path("plots").mkdir(parents=False, exist_ok=True)
    pyplot.savefig(filename)
    pyplot.close()

probs = json.load(open("probs.json" ))
plot_D_probabilities(probs["epochs"], probs["test_probs"], probs["train_probs"])