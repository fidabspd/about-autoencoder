import os
from tqdm import tqdm
import torch
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
from network.aae import AdversarialAutoEncoder, Discriminator
from network.loss import LogLikelihood, DiscriminatorLoss, GeneratorLoss


def train_model(aae, discriminator, train_dl, optim_aae, optim_disc,
                optim_gen, log_likelihood, disc_loss, gen_loss,
                n_epochs, device, aae_file_path, discriminator_file_path):

    def train_one_epoch(aae, discriminator, dl, optim_aae, optim_disc,
                        optim_gen, log_likelihood, disc_loss, gen_loss, device):

        n_data = len(dl.dataset)
        train_nll = 0
        train_disc_loss = 0
        train_gen_loss = 0
        n_processed_data = 0

        aae.train()
        discriminator.train()
        pbar = tqdm(dl)
        for batch in pbar:
            x, _ = batch
            x = x.to(device)
            now_batch_len = len(x)
            n_processed_data += now_batch_len

            # aae
            optim_aae.zero_grad()
            x_hat = aae(x)
            _log_likelihood = log_likelihood(x, x_hat)
            _negative_log_likelihood = -_log_likelihood
            _negative_log_likelihood.backward()
            optim_aae.step()
            train_nll += _negative_log_likelihood.item()/n_data
            train_nll_tmp = train_nll*n_data/n_processed_data

            # discriminator
            optim_disc.zero_grad()
            z_fake = aae.encoder(x)
            z_real = torch.randn_like(z_fake).to(device)
            real_disc_probs = discriminator(z_real)
            fake_disc_probs = discriminator(z_fake)
            _disc_loss = disc_loss(real_disc_probs, fake_disc_probs)
            _disc_loss.backward()
            optim_disc.step()
            train_disc_loss += _disc_loss.item()/n_data
            train_disc_loss_tmp = train_disc_loss*n_data/n_processed_data

            # generator (aae.encoder)
            for _ in range(2):
                optim_gen.zero_grad()
                z_fake = aae.encoder(x)
                fake_disc_probs = discriminator(z_fake)
                _gen_loss = gen_loss(fake_disc_probs)
                _gen_loss.backward()
                optim_gen.step()
            train_gen_loss += _gen_loss.item()/n_data
            train_gen_loss_tmp = train_gen_loss*n_data/n_processed_data

            pbar.set_description(
                'NLL Loss: {:9.6f} | Disc Loss: {:9.6f} | Gen Loss: {:9.6f} | {}/{} '.\
                    format(train_nll_tmp, train_disc_loss_tmp,
                           train_gen_loss_tmp, n_processed_data, n_data))

        return train_nll, train_disc_loss, train_gen_loss

    aae = aae.to(device)
    discriminator = discriminator.to(device)

    for epoch in range(n_epochs):

        print(f'\nEpoch: {epoch+1}/{n_epochs}')
        train_nll, train_disc_loss, train_gen_loss = train_one_epoch(
            aae, discriminator, train_dl, optim_aae, optim_disc,
            optim_gen, log_likelihood, disc_loss, gen_loss, device
        )
        torch.save(aae, aae_file_path)
        torch.save(discriminator, discriminator_file_path)


def main():

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    BATCH_SIZE = 64
    LEARNING_RATE = 0.001
    N_EPOCHS = 20

    IN_DIM = 1*28*28
    HIDDEN_DIM = 64
    LATENT_DIM = 16
    IMG_SIZE = 28
    LOSS_SCALE = 100
    DISC_HIDDEN_DIM = 64
    DISC_OUT_DIM = 128
    DROPOUT_RATIO = 0.1

    MNIST_DIR = "./MNIST_DATASET"
    AAE_FILE_PATH = "./model/aae.pt"
    DISCRIMINATOR_FILE_PATH = "./model/aae_discriminator.pt"


    if not os.path.exists(os.path.dirname(AAE_FILE_PATH)):
        os.mkdir(os.path.dirname(AAE_FILE_PATH))
    if not os.path.exists(os.path.dirname(DISCRIMINATOR_FILE_PATH)):
        os.mkdir(os.path.dirname(DISCRIMINATOR_FILE_PATH))

    mnist_transform = transforms.Compose([
        transforms.ToTensor()
    ])
    dataset = MNIST(MNIST_DIR, transform=mnist_transform, train=True, download=True)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

    # Model
    aae = AdversarialAutoEncoder(IN_DIM, LATENT_DIM, HIDDEN_DIM, IMG_SIZE, DROPOUT_RATIO)
    discriminator = Discriminator(LATENT_DIM, DISC_HIDDEN_DIM, DISC_OUT_DIM, DROPOUT_RATIO)

    # Loss
    log_likelihood = LogLikelihood(LOSS_SCALE)
    discriminator_loss = DiscriminatorLoss(LOSS_SCALE)
    generator_loss = GeneratorLoss(LOSS_SCALE)

    # Optimizer
    optimizer_aae = torch.optim.Adam(aae.parameters(), lr=LEARNING_RATE)
    optimizer_disc = torch.optim.Adam(discriminator.parameters(), lr=LEARNING_RATE/10)
    optimizer_gen = torch.optim.Adam(aae.encoder.parameters(), lr=LEARNING_RATE)

    # Train
    train_model(aae, discriminator, dataloader, optimizer_aae, optimizer_disc,
                optimizer_gen, log_likelihood, discriminator_loss, generator_loss,
                N_EPOCHS, DEVICE, AAE_FILE_PATH, DISCRIMINATOR_FILE_PATH)


if __name__ == "__main__":

    main()
