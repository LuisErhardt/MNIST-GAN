import torch
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision.transforms as transforms
from models import Discriminator
from drawPlots import plot_D_probabilities, plot_accuracy
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

    def __init__(self):
        # download MNIST data
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
        self.train_set = MNIST(root=".", train=True, download=True, transform=transform)
        self.test_set = MNIST(root=".", train=False, download=True, transform=transform)

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

    def membership_inference_attack_md_gan(self, dir_with_models, num_epochs, clientID, swap):
        def mia_on_model(self, path):
            print(path)

            # load model from file
            discriminator = Discriminator()
            discriminator.load_state_dict(torch.load(path))
            discriminator.eval()

            # load samples from train set
            if swap:
                print("Swap")
                # test the D on the whole training set
                train_loader = DataLoader(self.train_set, batch_size=n_train, shuffle=True)
            else:
                print("No swap")
                # test the D only on the partial dataset that the client used for training
                with open('indices.json') as file:
                    indice_lists = json.load(file)

                indices = indice_lists[clientID]
                sampler = SubsetRandomSampler(indices)
                train_loader = DataLoader(self.train_set, batch_size=n_train, sampler=sampler)
            
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
            test_loader = DataLoader(self.test_set, batch_size=n_test, shuffle=True)
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

            # count how many elements from train and test output appear in the first n_train elements of the list
            num_of_ones = 0
            num_of_zeros = 0
            for elem in outputs_with_labels[:n_train]:
                if elem[1] == 1:
                    num_of_ones += 1
                else:
                    num_of_zeros += 1

            accuracy = num_of_ones / n_train  
            acc_train.append(accuracy)

            print("Number of 1s (from train set):", num_of_ones)
            print("Number of 0s (from test set):", num_of_zeros)
            print("Accuracy:", accuracy)
            print("Random guessing accuracy:", guessing_accuracy)


        acc_train = []
        n_train = 100
        n_test = 1000
        n_total = n_train + n_test
        guessing_accuracy = n_train / n_total

        dirs = os.listdir(dir_with_models)

        for dir in dirs:
            path = os.path.join(dir_with_models, dir, "Discriminator{}_state_dict_model.pt".format(clientID))
            mia_on_model(self, path)

        # compute polyfit (for trend)
        x = np.arange(0, num_epochs, 5)
        y = np.array(acc_train)
        z = np.polyfit(x, y, 1)
        p = np.poly1d(z)
        print(p)

        acc = {}
        acc["accuracy"] = acc_train
        acc["random"] = [guessing_accuracy for i in range(num_epochs) if (i+1) % 5 == 0]
        acc["epochs"] = np.arange(0, num_epochs, 5)

        if swap:
            filename = 'plots/md-gan_swap_accuracy_D{}.png'.format(clientID)
        else:
            filename = 'plots/md-gan_swap_accuracy_D{}.png'.format(clientID)

        plot_accuracy(acc["epochs"], acc["accuracy"], acc["random"], p, filename, 'MD-GAN')

        
    def membership_inference_attack_gan(self, dir_with_models, num_epochs):
        def mia_on_model(self, path):
            print(path)

            # load model from file
            discriminator = Discriminator()
            discriminator.load_state_dict(torch.load(path))
            discriminator.eval()

            # load samples from train set
            train_loader = DataLoader(self.train_set, batch_size=n_train, shuffle=True)
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
            test_loader = DataLoader(self.test_set, batch_size=n_test, shuffle=True)
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

            # count how many elements from train and test output appear in the first n_train elements of the list
            num_of_ones = 0
            num_of_zeros = 0
            for elem in outputs_with_labels[:n_train]:
                if elem[1] == 1:
                    num_of_ones += 1
                else:
                    num_of_zeros += 1

            accuracy = num_of_ones / n_train  
            acc_train.append(accuracy)

            print("Number of 1s (from train set):", num_of_ones)
            print("Number of 0s (from test set):", num_of_zeros)
            print("Accuracy:", accuracy)
            print("Random guessing accuracy:", guessing_accuracy)


        acc_train = []
        n_train = 100
        n_test = 1000
        n_total = n_train + n_test
        guessing_accuracy = n_train / n_total

        dirs = os.listdir(dir_with_models)

        for dir in dirs:
            path = os.path.join(dir_with_models, dir, "Discriminator_state_dict_model.pt")
            mia_on_model(self, path)

        # compute polyfit (for trend)
        x = np.arange(0, num_epochs, 5)
        y = np.array(acc_train)
        z = np.polyfit(x, y, 1)
        p = np.poly1d(z)
        print(p)

        acc = {}
        acc["accuracy"] = acc_train
        acc["random"] = [guessing_accuracy for i in range(num_epochs) if (i+1) % 5 == 0]
        acc["epochs"] = np.arange(0, num_epochs, 5)

        plot_accuracy(acc["epochs"], acc["accuracy"], acc["random"], p, 'plots/gan_accuracy.png', 'GAN')


    def save_graph_with_probabilities(self, dir_with_models, num_epochs):

        probs_test = []
        probs_train = []

        dirs = os.listdir(dir_with_models)

        for dir in dirs:
            files = os.listdir(os.path.join(dir_with_models, dir))
            for file in files:
                path = os.path.join(dir_with_models, dir, file)

                # load model from file
                discriminator = Discriminator().to(device=self.device)
                discriminator.load_state_dict(torch.load(path))
                discriminator.eval()

                # load test samples
                test_loader = DataLoader(self.test_set, batch_size=32, shuffle=True)
                test_samples, _ = next(iter(test_loader))
                test_samples = test_samples.to(device=self.device)

                # load train samples
                train_loader = DataLoader(self.train_set, batch_size=10, shuffle=True)
                train_samples, _ = next(iter(train_loader))
                train_samples = train_samples.to(device=self.device)

                # inference with test samples
                test_output = discriminator(test_samples)
                test_mean_probability = torch.mean(test_output)
                probs_test.append(test_mean_probability.item())

                # inference with train samples
                train_output = discriminator(train_samples)
                train_mean_probability = torch.mean(train_output)
                probs_train.append(train_mean_probability.item())
        

        # save probabilities to file
        probs = {}
        probs["test_probs"] = probs_test
        probs["train_probs"] = probs_train
        probs["epochs"] = [i for i in range(num_epochs) if (i+1) % 5 == 0]
        json.dump(probs, open("probs.json", 'w' ))

        plot_D_probabilities(probs["epochs"], probs["test_probs"], probs["train_probs"])


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-do", type=str, default="mia"
    )
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
        "-dirWithModels", type=str, default="savedModels/gan"
    )
    args = parser.parse_args()

    inference = Inference()

    if args.do == "mia" and args.pathToModel != "":
        if args.numClients == 1:
            inference.membership_inference_attack_gan(args.pathToModel, args.numEpochs)
        else:
            for clientID in range(args.numClients):
                inference.membership_inference_attack_md_gan(args.pathToModel, args.numEpochs, clientID, args.swap)    
    
    elif args.do == "saveGraph" and args.dirWithModels != "":
        inference.save_graph_with_probabilities(args.dirWithModels, args.numEpochs)
    
    else: 
        raise Exception("Parameters wrong")