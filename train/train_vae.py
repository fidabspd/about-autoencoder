from tqdm import tqdm
import torch
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
from network.vae import VariationalAutoEncoder
from network.loss import ELBO


def train_model(model, train_dl, optimizer, criterion, n_epochs, device, model_file_path):

    def train_one_epoch(model, dl, optimizer, criterion, device):

        n_data = len(dl.dataset)
        train_loss = 0
        n_processed_data = 0

        model.train()
        pbar = tqdm(dl)
        for batch in pbar:
            x, _ = batch
            x = x.to(device)
            now_batch_len = len(x)
            n_processed_data += now_batch_len

            optimizer.zero_grad()
            x_hat, mu, sigma = model(x)
            loss, _, _ = criterion(x, x_hat, mu, sigma)
            loss = -loss
            train_loss += loss.item()/n_data

            loss.backward()
            optimizer.step()

            train_loss_tmp = train_loss*n_data/n_processed_data
            pbar.set_description(
                f'Train Loss: {train_loss_tmp:9.6f} | {n_processed_data}/{n_data} ')

        return train_loss

    model = model.to(device)

    for epoch in range(n_epochs):

        print(f'\nEpoch: {epoch+1}/{n_epochs}')
        train_loss = train_one_epoch(model, train_dl, optimizer, criterion, device)

    torch.save(model, model_file_path)


def main():

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    BATCH_SIZE = 64
    LEARNING_RATE = 0.001
    N_EPOCHS = 5

    IN_CH = 1
    HIDDEN_CH = 16
    KERNEL_SIZE = 3
    LATENT_DIM = 32
    IMG_SIZE = 28
    LOSS_SCALE = 100

    MNIST_DIR = "../MNIST_DATASET"
    MODEL_FILE_PATH = "../model/vae.pt"

    mnist_transform = transforms.Compose([
        transforms.ToTensor()
    ])
    dataset = MNIST(MNIST_DIR, transform=mnist_transform, train=True, download=True)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Model
    model = VariationalAutoEncoder(
        IN_CH, LATENT_DIM, HIDDEN_CH, KERNEL_SIZE, IMG_SIZE)

    # Loss
    criterion = ELBO(LOSS_SCALE)
    
    # Train
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    train_model(model, dataloader, optimizer, criterion, N_EPOCHS, DEVICE, MODEL_FILE_PATH)


if __name__ == '__main__':

    main()