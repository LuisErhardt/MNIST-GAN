import torch
from torch.utils.data import DataLoader
from torch import nn
from pathlib import Path
import argparse
from matplotlib import pyplot
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
from models import Discriminator, Generator
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler
import random
import json

def save_plot(samples, epoch):
    for i in range(16):
        ax = pyplot.subplot(4, 4, i + 1)
        pyplot.imshow(samples[i].cpu().detach().reshape(28, 28), cmap="gray_r")
        pyplot.xticks([])
        pyplot.yticks([])
    filename = 'plots/md-gan/epoch%03d.png' % (epoch)
    Path("plots/md-gan").mkdir(parents=True, exist_ok=True)
    pyplot.savefig(filename)
    pyplot.close()

class Client():
    def __init__(self, batch_size, num_epochs, device, lr, loss_function, indices, **kwargs):
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.loss_function = loss_function
        self.device = device
        self.lr = lr
        self.indices = indices

        self.createDataLoader()

        self.discriminator = Discriminator().to(device=self.device)
        self.initialize_optimizer()

    def createDataLoader(self):
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
        train_set = MNIST(root=".", train=True, download=True, transform=transform)

        sampler = SubsetRandomSampler(self.indices)
        dataLoader = DataLoader(train_set, self.batch_size, sampler=sampler)

        self.dataLoader = dataLoader

    def initialize_optimizer(self):
        self.optimizer_discriminator = torch.optim.Adam(self.discriminator.parameters(), lr=self.lr)

    def train(self, server, client):
        for n, (real_samples, _) in enumerate(self.dataLoader):
            sample_size = len(real_samples)
            
            # Data for training the discriminator
            real_samples = real_samples.to(device=self.device)
            real_samples_labels = torch.ones((sample_size, 1)).to(device=self.device)

            latent_space_samples = torch.randn((sample_size, 100)).to(device=self.device)

            generated_samples = server.generator(latent_space_samples)
            generated_samples_labels = torch.zeros((sample_size, 1)).to(device=self.device)

            all_samples = torch.cat((real_samples, generated_samples))
            all_samples_labels = torch.cat((real_samples_labels, generated_samples_labels))

            # Training the discriminator
            self.discriminator.zero_grad()
            output_discriminator = self.discriminator(all_samples)
            loss_discriminator = self.loss_function(output_discriminator, all_samples_labels)
            loss_discriminator.backward()
            self.optimizer_discriminator.step()

            # Train the generator
            loss_generator_avg = server.train_Generator()

            # Show loss
            if n == 300:
                print(f"Loss D {client}: {loss_discriminator}")
                print(f"Loss G avg: {loss_generator_avg}")
            
        
class Server():
    def __init__(self, batch_size, num_epochs, num_clients, swap, **kwargs):
        print("init Server")
        if swap:
            print("Swap")
        else: 
            print("No swap")

        torch.manual_seed(111)

        self.batch_size = batch_size
        self.num_epochs = num_epochs
        lr = 0.0001
        self.loss_function = nn.BCELoss()
        self.num_clients = num_clients
        self.enableSwap = swap

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        self.generator = Generator().to(device=self.device)
        self.optimizer_generator = torch.optim.Adam(self.generator.parameters(), lr=lr)

        indice_lists = self.split_dataset()

        self.client_list= []
        for n in range(num_clients):
            client = Client(self.batch_size, self.num_epochs, self.device, lr, self.loss_function, indice_lists[n])
            self.client_list.append(client)
            print("Added Client", n)

    def save_Discriminator_models(self, epoch):
        dir = 'savedModels/md-gan/epoch{}'.format(epoch)
        Path(dir).mkdir(parents=True, exist_ok=True)
        for id, client in enumerate(self.client_list):
            PATH = dir + '/Discriminator{}_state_dict_model.pt'.format(id)
            print("Save", PATH)
            torch.save(client.discriminator.state_dict(), PATH)

    def split_dataset(self):
        print("Split dataset")

        # download MNIST data
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
        train_set = MNIST(root=".", train=True, download=True, transform=transform)

        # split the indices in equal parts according to number of clients
        partition_value = 1 / self.num_clients
        dataset_size = len(train_set)
        indices = list(range(dataset_size))
        split = int(np.floor(partition_value * dataset_size))

        # shuffle the dataset
        random_seed= 42
        np.random.seed(random_seed)
        np.random.shuffle(indices)

        # create lists of the split indices
        indice_lists = [indices[i:i + split] for i in range(0, dataset_size, split)]

        # save indices to file
        with open("indices.json", 'w') as output_file:
            json.dump(indice_lists, output_file, indent=2)

        return indice_lists
        

    def train_Generator(self):
        # Data for training the generator
        real_samples_labels = torch.ones((self.batch_size, 1)).to(device=self.device)
        latent_space_samples = torch.randn((self.batch_size, 100)).to(device=self.device)

        # Training the generator
        self.generator.zero_grad()
        losses = 0
        # for each client, generate samples, send them to the discriminator and collect the loss
        for n, client in enumerate(self.client_list):
            generated_samples = self.generator(latent_space_samples)
            output_discriminator_generated = client.discriminator(generated_samples)
            loss_generator = self.loss_function(output_discriminator_generated, real_samples_labels)
            losses += loss_generator
        #  calculate the average of the losses
        loss_generator_avg = losses / (len(self.client_list))
        loss_generator_avg.backward()
        self.optimizer_generator.step()

        return loss_generator_avg
        
    def swap(self):
        if self.num_clients < 2:
            print("No swap, less than 2 clients")
            pass

        # save indice lists temporarily
        indice_lists = []
        for client in self.client_list:
             indices = client.indices
             indice_lists.append(indices)

        # for each client, select another client randomly where it should send its weights
        clients = list(range(self.num_clients))
        random.shuffle(clients)
        pairs = []
        for i in range(0, self.num_clients):
            if i + 1 < self.num_clients:
                pair = [clients[i], clients[i + 1]]
                pairs.append(pair)
            else:
                pair = [clients[i], clients[0]]
                pairs.append(pair)

        # swap datasets
        for pair in pairs:
            sender = pair[0]
            receiver = pair[1]

            print(f"Send D{sender} to D{receiver}")
            self.client_list[receiver].indices = indice_lists[sender]
            self.client_list[receiver].createDataLoader()


    def run(self):
        print("run main")

        for epoch in range(self.num_epochs):
            print("Epoch", epoch)

            # swap D datasets every 10 rounds
            if self.enableSwap and (epoch+1) % 10 == 0:
                self.swap()

            # Train all discriminators
            for i, client in enumerate(self.client_list):
                client.train(self, i)

            # save discriminator models every 5 rounds
            if (epoch+1) % 5 == 0:
                self.save_Discriminator_models(epoch)

            # save sample images of Generator every five rounds
            if (epoch+1) % 5 == 0:
                latent_space_samples = torch.randn((self.batch_size, 100)).to(device=self.device)
                generated_samples = self.generator(latent_space_samples)
                save_plot(generated_samples, epoch)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-batch_size", type=int, default=32
    )
    parser.add_argument(
        "-epochs", type=int, default=50
    )
    parser.add_argument(
        "-clients", type=int, default=5
    )
    parser.add_argument(
        "-swap", action='store_true'
    )
    args = parser.parse_args()

    Server = Server(args.batch_size, args.epochs, args.clients, args.swap)

    Server.run()