# MNIST-GAN

This repository contains code written to run experiments for a bachelor thesis.
It can be used to train GANs and MD-GANs on MNIST or Fashion-MNIST datasets and run membership inference attacks on the trained models.

During training of the models, model parameters of each discriminator are saved every five rounds. Also, output of the generator is saved every 5 rounds.

The attack creates a graph for each discriminators showing the accuracy of the attack on it.

## Installation

Clone this repository and run `pip install requirements.txt`.

## Train GAN

Example:
```
python3 src/gan.py -epochs 500 -dataset MNIST 
```

## Train MD-GAN

Examples:
```
python3 src/md-gan.py -epochs 500 -swap -dataset Fashion -clients 10
```
```
python3 src/md-gan.py -epochs 600 -dataset MNIST -clients 5
```

## Run Membership Inference Attack

It is important that the parameters (e.g. dataset, numEpochs, clients, ..) are same as those that were used for the training of the attacked model.

Example for MIA on GAN:
```
python3 src/inference.py -pathToModel <pathToDirectoryWhereTrainedModelsAreSaved> -numEpochs 500 -dataset MNIST 
```

Example for MIA on MD-GAN with swap:
```
python3 src/inference.py -pathToModel <pathToDirectoryWhereTrainedModelsAreSaved> -numEpochs 500 -dataset Fashion -swap -numClients 10
```
