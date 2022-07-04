from tqdm import tqdm
import torch
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
from vae import VariationalAutoEncoder, NegativeELBO


def train_model(model, train_dl, optimizer, criterion, n_epochs, device, model_file_path):

    def train_one_epoch(model, dl, optimizer, criterion, device):

        nonlocal global_step

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

            x_hat, mu, sigma = model(x, condition)
            loss, _, _ = criterion(x, x_hat, mu, sigma)
            train_loss += loss.item()/n_data

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss_tmp = train_loss*n_data/n_processed_data
            pbar.set_description(
                f'Train Loss: {train_loss_tmp:9.6f} | {n_processed_data}/{n_data} ')

            batch_loss = loss.item()/now_batch_len
            
            global_step += 1

        return train_loss


    model = model.to(device)
    global_step = 0

    for epoch in range(n_epochs):

        print(f'\nEpoch: {epoch+1}/{n_epochs}')
        train_loss = train_one_epoch(model, train_dl, optimizer, criterion, device)

    torch.save(model, model_file_path)


def main():

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    BATCH_SIZE = 64
    LEARNING_RATE = 0.0005
    N_EPOCHS = 3

    COND_EMB_DIM = 16
    IN_CH = 1
    HIDDEN_CH = 16
    KERNEL_SIZE = 3
    LATENT_DIM = 32
    N_CONDITION_LABELS = 10
    IMG_SIZE = 28

    MODEL_FILE_PATH = './vae.pt'

    mnist_transform = transforms.Compose([
        transforms.ToTensor()
    ])

    download_root = './MNIST_DATASET'
    dataset = MNIST(download_root, transform=mnist_transform, train=True, download=True)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Model
    vae = VariationalAutoEncoder(
        COND_EMB_DIM, IN_CH, LATENT_DIM, HIDDEN_CH, KERNEL_SIZE, N_CONDITION_LABELS, IMG_SIZE)

    # Loss
    negative_elbo = NegativeELBO()

    # Train
    optimizer = torch.optim.Adam(vae.parameters(), lr=LEARNING_RATE)
    train_model(vae, dataloader, optimizer, negative_elbo, N_EPOCHS, device, MODEL_FILE_PATH)


if __name__ == '__main__':

    main()
