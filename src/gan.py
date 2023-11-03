import torch
from torch.utils.data import DataLoader
from torch import nn
from pathlib import Path
import argparse
from matplotlib import pyplot
from torchvision.datasets import MNIST, FashionMNIST
import torchvision.transforms as transforms
from models import Discriminator, Generator

def save_plot(samples, epoch):
    for i in range(16):
        ax = pyplot.subplot(4, 4, i + 1)
        pyplot.imshow(samples[i].cpu().detach().reshape(28, 28), cmap="gray_r")
        pyplot.xticks([])
        pyplot.yticks([])
    filename = 'plots/gan/epoch%03d.png' % (epoch)
    Path("plots/gan").mkdir(parents=True, exist_ok=True)
    pyplot.savefig(filename)
    pyplot.close()

def save_Discriminator_models(epoch, discriminator):
        dir = f'savedModels/gan/epoch{epoch:03}'
        Path(dir).mkdir(parents=True, exist_ok=True)
        PATH = dir + '/Discriminator_state_dict_model.pt'.format(id)
        torch.save(discriminator.state_dict(), PATH)

def main(batch_size, num_epochs, dataset, subClass):
    torch.manual_seed(111)

    device = ""
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print(dataset)

    if dataset == "MNIST":

        # download MNIST data
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
        train_set = MNIST(root=".", train=True, download=True, transform=transform)

    if dataset == "Fashion":
        # download Fashion MNIST data
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
        train_set = FashionMNIST(root=".", train=True, download=True, transform=transform)

    if subClass != None:
        # select only data from specific class
        idx = train_set.targets == subClass
        train_set.targets = train_set.targets[idx]
        train_set.data = train_set.data[idx]

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)

    # initialize models
    discriminator = Discriminator().to(device=device)
    generator = Generator().to(device=device)

    lr = 0.0001
    loss_function = nn.BCELoss()

    optimizer_discriminator = torch.optim.Adam(discriminator.parameters(), lr=lr)
    optimizer_generator = torch.optim.Adam(generator.parameters(), lr=lr)

    print("Start Training")

    for epoch in range(num_epochs):
        for n, (real_samples, _) in enumerate(train_loader):
            sample_size = len(real_samples)

            # Data for training the discriminator
            real_samples = real_samples.to(device=device)
            real_samples_labels = torch.ones((sample_size, 1)).to(device=device)

            latent_space_samples = torch.randn((sample_size, 100)).to(device=device)

            generated_samples = generator(latent_space_samples)
            generated_samples_labels = torch.zeros((sample_size, 1)).to(device=device)

            all_samples = torch.cat((real_samples, generated_samples))
            all_samples_labels = torch.cat((real_samples_labels, generated_samples_labels))

            # Training the discriminator
            discriminator.zero_grad()
            output_discriminator = discriminator(all_samples)
            loss_discriminator = loss_function(output_discriminator, all_samples_labels)
            loss_discriminator.backward()
            optimizer_discriminator.step()

            # Data for training the generator
            latent_space_samples = torch.randn((batch_size, 100)).to(device=device)

            # Training the generator
            generator.zero_grad()
            generated_samples = generator(latent_space_samples)
            output_discriminator_generated = discriminator(generated_samples)
            real_samples_labels = torch.ones((batch_size, 1)).to(device=device)
            loss_generator = loss_function(output_discriminator_generated, real_samples_labels)
            loss_generator.backward()
            optimizer_generator.step()

            # Show loss
            if n == batch_size - 1:
                print(f"Epoch: {epoch}, Step: {n}, Loss D: {loss_discriminator}")
                print(f"Epoch: {epoch}, Step: {n}, Loss G: {loss_generator}")

            # save sample images of Generator every five rounds
            if (epoch+1) % 5 == 0 and n == 0:
                latent_space_samples = torch.randn((batch_size, 100)).to(device=device)
                generated_samples = generator(latent_space_samples)
                save_plot(generated_samples, epoch)

            # save discriminator models every 5 rounds
            if (epoch+1) % 5 == 0:
                save_Discriminator_models(epoch, discriminator)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-batch_size", type=int, default=32
    )
    parser.add_argument(
        "-epochs", type=int, default=50
    )
    parser.add_argument(
        "-dataset", type=str, default="MNIST"
    )
    parser.add_argument(
        "-subClass", type=int
    )
    args = parser.parse_args()

    main(args.batch_size, args.epochs, args.dataset, args.subClass)