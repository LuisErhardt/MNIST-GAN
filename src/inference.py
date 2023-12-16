import torch
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST, FashionMNIST
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision.transforms as transforms
from models import Discriminator
from drawPlots import plot_accuracy
import os
import json
import argparse
import statistics
import numpy as np
import json


class Inference():
    #------------------------------------------------
    # If the model was trained on a gpu, the inference
    # must also be run on a system with a gpu!
    #------------------------------------------------

    def __init__(self, dataset, subClass, num_epochs):

        self.num_epochs = num_epochs
        self.B_train = 100
        self.B_test = 1000

        print(dataset)

        if dataset == "MNIST":
            # download MNIST data
            transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
            self.train_set = MNIST(root=".", train=True, download=True, transform=transform)
            self.test_set = MNIST(root=".", train=False, download=True, transform=transform)

        if dataset == "Fashion":
            # download Fashion MNIST data
            transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
            self.train_set = FashionMNIST(root=".", train=True, download=True, transform=transform)
            self.test_set = FashionMNIST(root=".", train=False, download=True, transform=transform)

        if subClass != None:
            # select only data from specific class
            idx = self.train_set.targets == subClass
            self.train_set.targets = self.train_set.targets[idx]
            self.train_set.data = self.train_set.data[idx]

            idx = self.test_set.targets == subClass
            self.test_set.targets = self.test_set.targets[idx]
            self.test_set.data = self.test_set.data[idx]

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

    def membership_inference_attack_md_gan(self, dir_with_models, clientID, swap):
        accuracies = []
        B_total = self.B_train + self.B_test
        guessing_accuracy = self.B_train / B_total

        dirs = sorted(os.listdir(os.path.join(dir_with_models, 'savedModels')))

        for dir in dirs:
            discriminator_path = os.path.join(os.path.join(dir_with_models, 'savedModels'), dir, "Discriminator{}_state_dict_model.pt".format(clientID))
            if swap:
                accuracy = self.mia_on_D(self, discriminator_path, os.path.join(dir_with_models, 'indices.json'), clientID)
            else:
                accuracy = self.mia_on_D(self, discriminator_path, os.path.join(dir_with_models))
            accuracies.append(accuracy)

        random_guessing = [guessing_accuracy for i in range(self.num_epochs) if (i+1) % 5 == 0]

        if swap:
            filename = os.path.join(dir_with_models, 'md-gan_swap_accuracy_D{}.pdf'.format(clientID))
        else:
            filename = os.path.join(dir_with_models, 'md-gan_no_swap_accuracy_D{}.pdf'.format(clientID))

        plot_accuracy(self.num_epochs, accuracies, random_guessing, filename, 'MD-GAN')

        
    def membership_inference_attack_gan(self, dir_with_models):
        accuracies = []
        B_total = self.B_train + self.B_test
        guessing_accuracy = self.B_train / B_total

        # for each saved model, do mia
        dirs = sorted(os.listdir(os.path.join(dir_with_models, 'savedModels')))
        for dir in dirs:
            tmp_path = os.path.join(dir_with_models, 'savedModels', dir)
            path = os.path.join(tmp_path, os.listdir(tmp_path)[0])
            accuracy = self.mia_on_D(self, path)
            accuracies.append(accuracy)

        # plot accuracy values
        random_guessing = [guessing_accuracy for i in range(self.num_epochs) if (i+1) % 5 == 0]
        file_path = os.path.join(dir_with_models, 'gan_accuracy.pdf')
        plot_accuracy(self.num_epochs, accuracies, random_guessing, file_path, 'GAN')


    def mia_on_D(self, discriminator_path, indices_path = None, clientID = None):
        print(discriminator_path)

        # load model from file
        discriminator = Discriminator()
        discriminator.load_state_dict(torch.load(discriminator_path))
        discriminator.eval()

        # load samples from train set
        if indices_path and clientID:
            print("Indices")
            # test the D only on the partial dataset that the client used for training
            with open(indices_path) as file:
                indice_lists = json.load(file)

            indices = indice_lists[clientID]
            sampler = SubsetRandomSampler(indices)
            train_loader = DataLoader(self.train_set, batch_size=self.B_train, sampler=sampler) 

        else:
            print("No Indices")
            # test the D on the whole training set
            train_loader = DataLoader(self.train_set, batch_size=self.B_train, shuffle=True)
          
        train_samples, _ = next(iter(train_loader))

        train_output = discriminator(train_samples)

        # convert Tensor to list and sort the elements
        train_output_list = [elem[0] for elem in sorted(train_output.tolist(), reverse=True)]

        # add Label 1 to training data outputs
        train_outputs_with_labels = [(num, 1) for num in train_output_list]

        train_mean = statistics.mean(train_output_list)

        print("Train output:")
        print(train_mean)

        # load samples from test set
        test_samples = []
        test_loader = DataLoader(self.test_set, batch_size=self.B_test, shuffle=True)
        test_samples, _ = next(iter(test_loader))

        test_output = discriminator(test_samples)

        # convert Tensor to list and sort the elements
        test_output_list = [elem[0] for elem in sorted(test_output.tolist(), reverse=True)]

        # add Label 0 to test data outputs
        test_outputs_with_labels = [(num, 0) for num in test_output_list]

        test_mean = statistics.mean(test_output_list)

        print("Test output:")
        print(test_mean)

        # Concatenate the two lists and sort the elements based on the first element of each tuple
        outputs_with_labels = sorted(train_outputs_with_labels + test_outputs_with_labels, key=lambda x: x[0], reverse=True)

        # count how many elements from train and test output appear in the first B_train elements of the list
        num_of_ones = 0
        num_of_zeros = 0
        for elem in outputs_with_labels[:self.B_train]:
            if elem[1] == 1:
                num_of_ones += 1
            else:
                num_of_zeros += 1

        accuracy = num_of_ones / self.B_train  

        print("Number of 1s (from train set):", num_of_ones)
        print("Number of 0s (from test set):", num_of_zeros)
        print("Accuracy:", accuracy)

        return accuracy

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-pathToModel", type=str, default=""
    )
    parser.add_argument(
        "-numEpochs", type=int, default=500
    )
    parser.add_argument(
        "-numClients", type=int, default=1
    )
    parser.add_argument(
        "-swap", action='store_true'
    )
    parser.add_argument(
        "-dataset", type=str, default="MNIST"
    )
    parser.add_argument(
        "-subClass", type=int
    )
    args = parser.parse_args()

    inference = Inference(args.dataset, args.subClass, args.numEpochs)

    if args.numClients == 1:
        inference.membership_inference_attack_gan(args.pathToModel)
    else:
        for clientID in range(args.numClients):
            inference.membership_inference_attack_md_gan(args.pathToModel, clientID, args.swap)    
