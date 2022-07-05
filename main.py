from tqdm import tqdm
import torch
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
from ae import AutoEncoder
from vae import VariationalAutoEncoder
from cvae import ConditionalVariationalAutoEncoder
from loss import LogLikelihood, ELBO


def train_model(model, mode, train_dl, optimizer, criterion, n_epochs, device, model_file_path):

    valid_mode = ["ae", "vae", "cvae"]
    if mode not in valid_mode:
        raise ValueError(f"'mode' must be one of {valid_mode}")

    def train_one_epoch(model, dl, optimizer, criterion, device):

        nonlocal mode

        n_data = len(dl.dataset)
        train_loss = 0
        n_processed_data = 0

        model.train()
        pbar = tqdm(dl)
        for batch in pbar:
            x, condition = batch
            x, condition = x.to(device), condition.unsqueeze(1).to(device)
            now_batch_len = len(x)
            n_processed_data += now_batch_len

            if mode == "ae":
                x_hat = model(x)
                loss = criterion(x, x_hat)
            elif mode == "vae":
                x_hat, mu, sigma = model(x)
                loss, _, _ = criterion(x, x_hat, mu, sigma)
            elif mode == "cvae":
                x_hat, mu, sigma = model(x, condition)
                loss, _, _ = criterion(x, x_hat, mu, sigma)
            loss = -loss
            train_loss += loss.item()/n_data

            optimizer.zero_grad()
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

    COND_EMB_DIM = 16
    IN_CH = 1
    HIDDEN_CH = 16
    KERNEL_SIZE = 3
    LATENT_DIM = 32
    N_CONDITION_LABELS = 10
    IMG_SIZE = 28
    LOSS_SCALE = 100

    MODE = 'vae'
    MODEL_FILE_PATH = f'./{MODE}.pt'

    mnist_transform = transforms.Compose([
        transforms.ToTensor()
    ])

    download_root = './MNIST_DATASET'
    dataset = MNIST(download_root, transform=mnist_transform, train=True, download=True)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Model & Loss
    if MODE == 'ae':
        model = AutoEncoder(
            IN_CH, LATENT_DIM, HIDDEN_CH, KERNEL_SIZE, IMG_SIZE)
        criterion = LogLikelihood(LOSS_SCALE)
    elif MODE == 'vae':
        model = VariationalAutoEncoder(
            IN_CH, LATENT_DIM, HIDDEN_CH, KERNEL_SIZE, IMG_SIZE)
        criterion = ELBO(LOSS_SCALE)
    elif MODE == 'cvae':
        model = ConditionalVariationalAutoEncoder(
            COND_EMB_DIM, IN_CH, LATENT_DIM, HIDDEN_CH, KERNEL_SIZE, N_CONDITION_LABELS, IMG_SIZE)
        criterion = ELBO(LOSS_SCALE)

    # Train
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    train_model(model, MODE, dataloader, optimizer, criterion, N_EPOCHS, DEVICE, MODEL_FILE_PATH)


if __name__ == '__main__':

    main()
