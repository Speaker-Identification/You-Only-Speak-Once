import time

import numpy as np
import torch
import tqdm
from torch import optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

from fbanks.dataset import SiameseAPNDataset
from fbanks.model import FbanksVoiceNet


def train(model, device, train_loader, optimizer, epoch, log_interval):
    model.train()
    losses = []
    for batch_idx, ((ax, ay), (px, py), (nx, ny)) in enumerate(tqdm.tqdm(train_loader)):
        ax, px, nx = ax.to(device), px.to(device), nx.to(device)
        optimizer.zero_grad()
        a_out, p_out, n_out = model(ax, px, nx)
        loss = model.loss(a_out, p_out, n_out)
        losses.append(loss.item())
        loss.backward()
        optimizer.step()

        if batch_idx % log_interval == 0:
            print('{} Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                time.ctime(time.time()),
                epoch, batch_idx * len(ax), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))

    return np.mean(losses)


def test(model, device, test_loader, log_interval=None):
    model.eval()
    test_loss = 0
    positive_accuracy = 0
    negative_accuracy = 0

    with torch.no_grad():
        for batch_idx, ((ax, ay), (px, py), (nx, ny)) in enumerate(tqdm.tqdm(test_loader)):
            ax, px, nx = ax.to(device), px.to(device), nx.to(device)
            a_out, p_out, n_out = model(ax, px, nx)
            test_loss_on = model.loss(a_out, p_out, n_out, reduction='sum').item()
            test_loss += test_loss_on

            if 1 - F.cosine_similarity(a_out, p_out) < model.loss_layer.margin:
                positive_accuracy += 1

            if 1 - F.cosine_similarity(a_out, n_out) > model.loss_layer.margin:
                negative_accuracy += 1

            if log_interval is not None and batch_idx % log_interval == 0:
                print('{} Test: [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    time.ctime(time.time()),
                    batch_idx * len(ax), len(test_loader.dataset),
                    100. * batch_idx / len(test_loader), test_loss_on))

    test_loss /= len(test_loader.dataset)
    positive_accuracy_mean = 100. * positive_accuracy / len(test_loader.dataset)
    negative_accuracy_mean = 100. * negative_accuracy / len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Positive Accuracy: {}/{} ({:.0f}%),  Negative Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, positive_accuracy, len(test_loader.dataset), positive_accuracy_mean, negative_accuracy, len(test_loader.dataset), negative_accuracy_mean))
    return test_loss, positive_accuracy_mean, negative_accuracy_mean


def main():
    use_cuda = True
    device = torch.device("cuda" if use_cuda else "cpu")
    print('using device', device)

    import multiprocessing
    print('num cpus:', multiprocessing.cpu_count())

    kwargs = {'num_workers': multiprocessing.cpu_count(),
              'pin_memory': True} if use_cuda else {}

    test_dataset = SiameseAPNDataset('C:/Users/vivek/dev/DATA599G1/voice-recognition/fbanks_test')
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)
    model = FbanksVoiceNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(10):
        train_loss = train(model, device, test_loader, optimizer, epoch, 100)
        test_loss, positive_accuracy_mean, negative_accuracy_mean = test(model, device, test_loader)
        print('After epoch: {}, test loss is: {}, positive accuracy: {}, and negative accuracy: {} '.format(epoch, test_loss, positive_accuracy_mean, negative_accuracy_mean))


if __name__ == '__main__':
    main()
