import time

import numpy as np
import torch
import tqdm
from torch import optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ..pt_util import restore_model, restore_objects, save_model, save_objects
from .triplet_loss_dataset import FBanksTripletDataset
from .triplet_loss_model import FBankTripletLossNet


def _get_cosine_distance(a, b):
    return 1 - F.cosine_similarity(a, b)


def train(model, device, train_loader, optimizer, epoch, log_interval):
    model.train()
    losses = []
    positive_accuracy = 0
    negative_accuracy = 0

    postitive_distances = []
    negative_distances = []

    for batch_idx, ((ax, ay), (px, py), (nx, ny)) in enumerate(tqdm.tqdm(train_loader)):
        ax, px, nx = ax.to(device), px.to(device), nx.to(device)
        optimizer.zero_grad()
        a_out, p_out, n_out = model(ax, px, nx)
        loss = model.loss(a_out, p_out, n_out)
        losses.append(loss.item())

        with torch.no_grad():
            p_distance = _get_cosine_distance(a_out, p_out)
            postitive_distances.append(torch.mean(p_distance).item())

            n_distance = _get_cosine_distance(a_out, n_out)
            negative_distances.append(torch.mean(n_distance).item())

            positive_distance_mean = np.mean(postitive_distances)
            negative_distance_mean = np.mean(negative_distances)

            positive_std = np.std(postitive_distances)
            threshold = positive_distance_mean + 3 * positive_std

            positive_results = p_distance < threshold
            positive_accuracy += torch.sum(positive_results).item()

            negative_results = n_distance >= threshold
            negative_accuracy += torch.sum(negative_results).item()

        loss.backward()
        optimizer.step()

        if batch_idx % log_interval == 0:
            print('{} Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                time.ctime(time.time()),
                epoch, batch_idx * len(ax), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))

            positive_distance_mean = np.mean(postitive_distances)
            negative_distance_mean = np.mean(negative_distances)
            print('Train Set: positive_distance_mean: {}, negative_distance_mean: {}, std: {}, threshold: {}'.format(
                positive_distance_mean, negative_distance_mean, positive_std, threshold))

    positive_accuracy_mean = 100. * positive_accuracy / len(train_loader.dataset)
    negative_accuracy_mean = 100. * negative_accuracy / len(train_loader.dataset)
    return np.mean(losses), positive_accuracy_mean, negative_accuracy_mean


def test(model, device, test_loader, log_interval=None):
    model.eval()
    losses = []
    positive_accuracy = 0
    negative_accuracy = 0

    postitive_distances = []
    negative_distances = []

    with torch.no_grad():
        for batch_idx, ((ax, ay), (px, py), (nx, ny)) in enumerate(tqdm.tqdm(test_loader)):
            ax, px, nx = ax.to(device), px.to(device), nx.to(device)
            a_out, p_out, n_out = model(ax, px, nx)
            test_loss_on = model.loss(a_out, p_out, n_out, reduction='mean').item()
            losses.append(test_loss_on)

            p_distance = _get_cosine_distance(a_out, p_out)
            postitive_distances.append(torch.mean(p_distance).item())

            n_distance = _get_cosine_distance(a_out, n_out)
            negative_distances.append(torch.mean(n_distance).item())

            positive_distance_mean = np.mean(postitive_distances)
            negative_distance_mean = np.mean(negative_distances)

            positive_std = np.std(postitive_distances)
            threshold = positive_distance_mean + 3 * positive_std

            # experiment with this threshold distance to play with accuracy numbers
            positive_results = p_distance < threshold
            positive_accuracy += torch.sum(positive_results).item()

            negative_results = n_distance >= threshold
            negative_accuracy += torch.sum(negative_results).item()

            if log_interval is not None and batch_idx % log_interval == 0:
                print('{} Test: [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    time.ctime(time.time()),
                    batch_idx * len(ax), len(test_loader.dataset),
                    100. * batch_idx / len(test_loader), test_loss_on))

    test_loss = np.mean(losses)
    positive_accuracy_mean = 100. * positive_accuracy / len(test_loader.dataset)
    negative_accuracy_mean = 100. * negative_accuracy / len(test_loader.dataset)

    positive_distance_mean = np.mean(postitive_distances)
    negative_distance_mean = np.mean(negative_distances)
    print('Test Set: positive_distance_mean: {}, negative_distance_mean: {}, std: {}, threshold: {}'.format(
        positive_distance_mean, negative_distance_mean, positive_std, threshold))

    print(
        '\nTest set: Average loss: {:.4f}, Positive Accuracy: {}/{} ({:.0f}%),  Negative Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, positive_accuracy, len(test_loader.dataset), positive_accuracy_mean, negative_accuracy,
            len(test_loader.dataset), negative_accuracy_mean))
    return test_loss, positive_accuracy_mean, negative_accuracy_mean


def main():
    model_path = 'siamese_fbanks_saved/'
    use_cuda = True
    device = torch.device("cuda" if use_cuda else "cpu")
    print('using device', device)

    import multiprocessing
    print('num cpus:', multiprocessing.cpu_count())

    kwargs = {'num_workers': multiprocessing.cpu_count(),
              'pin_memory': True} if use_cuda else {}

    train_dataset = FBanksTripletDataset('fbanks_train')
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, **kwargs)

    test_dataset = FBanksTripletDataset('fbanks_test')
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True, **kwargs)

    model = FBankTripletLossNet(margin=0.2).to(device)
    model = restore_model(model, model_path)
    last_epoch, max_accuracy, train_losses, test_losses, train_positive_accuracies, train_negative_accuracies, \
    test_positive_accuracies, test_negative_accuracies = restore_objects(model_path, (0, 0, [], [], [], [], [], []))

    start = last_epoch + 1 if max_accuracy > 0 else 0

    optimizer = optim.Adam(model.parameters(), lr=0.0005)

    for epoch in range(start, start + 20):
        train_loss, train_positive_accuracy, train_negative_accuracy = train(model, device, train_loader, optimizer,
                                                                             epoch, 500)
        test_loss, test_positive_accuracy, test_negative_accuracy = test(model, device, test_loader)
        print('After epoch: {}, train loss is : {}, test loss is: {}, '
              'train positive accuracy: {}, train negative accuracy: {}'
              'tes positive accuracy: {}, and test negative accuracy: {} '
              .format(epoch, train_loss, test_loss, train_positive_accuracy, train_negative_accuracy,
                      test_positive_accuracy, test_negative_accuracy))

        train_losses.append(train_loss)
        test_losses.append(test_loss)
        train_positive_accuracies.append(train_positive_accuracy)
        test_positive_accuracies.append(test_positive_accuracy)

        train_negative_accuracies.append(train_negative_accuracy)
        test_negative_accuracies.append(test_negative_accuracy)

        test_accuracy = (test_positive_accuracy + test_negative_accuracy) / 2

        if test_accuracy > max_accuracy:
            max_accuracy = test_accuracy
            save_model(model, epoch, model_path)
            save_objects((epoch, max_accuracy, train_losses, test_losses, train_positive_accuracies,
                          train_negative_accuracies, test_positive_accuracies, test_negative_accuracies),
                         epoch, model_path)
            print('saved epoch: {} as checkpoint'.format(epoch))


if __name__ == '__main__':
    main()
