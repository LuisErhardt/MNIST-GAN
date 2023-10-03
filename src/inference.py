import torch
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
from models import Discriminator
from drawPlots import plot_D_probabilities, plot_accuracy
import os
import json
import argparse
import statistics

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
        
    def membership_inference_attack(self, dir_with_models, num_epochs):
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
            files = os.listdir(os.path.join(dir_with_models, dir))
            for file in files:
                path = os.path.join(dir_with_models, dir, file)
                mia_on_model(self, path)

        acc = {}
        acc["accuracy"] = acc_train
        acc["random"] = [guessing_accuracy for i in range(num_epochs) if (i+1) % 5 == 0]
        acc["epochs"] = [i for i in range(num_epochs) if (i+1) % 5 == 0]

        plot_accuracy(acc["epochs"], acc["accuracy"], acc["random"])


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
        "-numEpochs", type=int, default=300
    )

    parser.add_argument(
        "-dirWithModels", type=str, default="savedModels/gan"
    )
    args = parser.parse_args()

    inference = Inference()

    if args.do == "mia" and args.pathToModel != "":
        inference.membership_inference_attack(args.pathToModel, args.numEpochs)
    
    elif args.do == "saveGraph" and args.dirWithModels != "":
        inference.save_graph_with_probabilities(args.dirWithModels, args.numEpochs)
    
    else: 
        raise Exception("Parameters wrong")