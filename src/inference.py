import torch
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
from models import Discriminator
from drawPlots import plot_D_probabilities
import os
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
        
    def membership_inference_attack(self, path):

        # load model from file
        discriminator = Discriminator()
        discriminator.load_state_dict(torch.load(path))
        discriminator.eval()

        # load 10 samples from train set
        train_loader = DataLoader(self.train_set, batch_size=10, shuffle=True)
        train_samples, _ = next(iter(train_loader))
        print(train_samples)

        # load 20 samples from test set
        test_samples = []
        test_loader = DataLoader(self.test_set, batch_size=20, shuffle=True)
        test_samples, _ = next(iter(test_loader))
        print(test_samples)

        train_output = discriminator(train_samples)
        print("Train output:")
        print(train_output)

        test_output = discriminator(test_samples)
        print("Test output:")
        print(test_output)

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
    path = ""

    inference = Inference()
    # inference.membership_inference_attack(path)
    inference.save_graph_with_probabilities('savedModels/gan', 200)