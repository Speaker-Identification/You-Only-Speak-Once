
import torchvision
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader,Dataset
import matplotlib.pyplot as plt
import torchvision.utils
import numpy as np
import random
from PIL import Image
import torch
from torch.autograd import Variable
import PIL.ImageOps    
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import tqdm
import time
import os

class SiameseNetworkDataset(Dataset):
    
    def __init__(self,imageFolderDataset,transform=None,should_invert=True):
        self.imageFolderDataset = imageFolderDataset    
        self.transform = transform
        self.should_invert = should_invert
        
    def __getitem__(self,index):
        img0_tuple = random.choice(self.imageFolderDataset.imgs)
        # For the INFERENCE case, we just need to make sure every pair of samples is
        # coming from different folders, even if they're from the same person
        while True:
            img1_tuple = random.choice(self.imageFolderDataset.imgs) 
            if img0_tuple[1] !=img1_tuple[1]:
                break

        img0 = Image.open(img0_tuple[0])
        img1 = Image.open(img1_tuple[0])

        ## Conversion to grayscale
        img0 = img0.convert("L")
        img1 = img1.convert("L")
        
        if self.should_invert:
            img0 = PIL.ImageOps.invert(img0)
            img1 = PIL.ImageOps.invert(img1)

        if self.transform is not None:
            img0 = self.transform(img0)
            img1 = self.transform(img1)
        labels = torch.squeeze(torch.from_numpy(np.array([int(img1_tuple[1]==img0_tuple[1])],dtype=np.float32)).long())
        # Label = 1 when the images are from the same class
        return img0, img1 , labels
    
    def __len__(self):
        return len(self.imageFolderDataset.imgs)


def restore(net, save_file):
    """Restores the weights from a saved file. This is courtesy our Deep Learning HW assignments
    """
    net_state_dict = net.state_dict()
    restore_state_dict = torch.load(save_file, map_location=lambda storage, loc: storage)
    #restore_state_dict = torch.load(save_file)

    restored_var_names = set()

    print('Restoring:')
    for var_name in restore_state_dict.keys():
        if var_name in net_state_dict:
            var_size = net_state_dict[var_name].size()
            restore_size = restore_state_dict[var_name].size()
            if var_size != restore_size:
                print('Shape mismatch for var', var_name, 'expected', var_size, 'got', restore_size)
            else:
                if isinstance(net_state_dict[var_name], torch.nn.Parameter):
                    # backwards compatibility for serialized parameters
                    net_state_dict[var_name] = restore_state_dict[var_name].data
                try:
                    net_state_dict[var_name].copy_(restore_state_dict[var_name])
                    print(str(var_name) + ' -> \t' + str(var_size) + ' = ' + str(int(np.prod(var_size) * 4 / 10**6)) + 'MB')
                    restored_var_names.add(var_name)
                except Exception as ex:
                    print('While copying the parameter named {}, whose dimensions in the model are'
                          ' {} and whose dimensions in the checkpoint are {}, ...'.format(
                              var_name, var_size, restore_size))
                    raise ex

    ignored_var_names = sorted(list(set(restore_state_dict.keys()) - restored_var_names))
    unset_var_names = sorted(list(set(net_state_dict.keys()) - restored_var_names))
    print('')
    if len(ignored_var_names) == 0:
        print('Restored all variables')
    else:
        print('Did not restore:\n\t' + '\n\t'.join(ignored_var_names))
    if len(unset_var_names) == 0:
        print('No new variables')
    else:
        print('Initialized but did not modify:\n\t' + '\n\t'.join(unset_var_names))

    print('Restored %s' % save_file)


class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        self.cnn1 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(1, 4, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(4),
            
            nn.ReflectionPad2d(1),
            nn.Conv2d(4, 8, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(8),
        )

        self.fc1 = nn.Sequential(
            nn.Linear(8*227*227, 512),
            nn.ReLU(inplace=True),
            )

        self.fc2 = nn.Linear(512, 2)

    def forward_once(self, x):
        output = self.cnn1(x)
        output = output.view(output.size()[0], -1)
        output = self.fc1(output)
        return output

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        return self.fc2(torch.abs(output1 - output2))

class CrossEntropyLoss(torch.nn.Module):
    def __init__(self):
        super(CrossEntropyLoss, self).__init__()

    def forward(self, output, label):
        label = label.long()
        loss = F.cross_entropy(output, label)
        return loss

def test(model, device, test_loader, criterion):
    model.eval()
    num_of_zeros = 0
    num_of_ones = 0
    with torch.no_grad():
        for batch_idx, (data0, data1, label) in enumerate(tqdm.tqdm(test_loader)):
            data0, data1, label = data0.to(device), data1.to(device), label.to(device)
            out = model(data0, data1)
            loss_function = criterion(out, label)

            curr_pick = torch.argmax(out, dim=1)
            if curr_pick == 0:
            	num_of_zeros += 1
            else:
            	num_of_ones += 1
    ans = True
    if num_of_zeros > num_of_ones:
    	ans = False
    return ans


def are_the_samples_identical():
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print('Using device', device)
    import multiprocessing
    num_workers = multiprocessing.cpu_count()
    print('num workers:', num_workers)
    kwargs = {'num_workers': num_workers,
              'pin_memory': True} if use_cuda else {}

    folder_dataset_test_own = dset.ImageFolder(root='/Users/Arjun_Singh/Documents/MS_DataScience/CSE599_DeepLearning/Final_Project/testing_on_us')
    siamese_dataset = SiameseNetworkDataset(imageFolderDataset=folder_dataset_test_own,
	                                    transform=transforms.Compose([transforms.ToTensor()]),
	                                    should_invert=False)

    test_dataloader_own = DataLoader(siamese_dataset, batch_size=1, shuffle=False, **kwargs)

    net_new = SiameseNetwork().to(device)
    restore(net_new, '/Users/Arjun_Singh/Documents/MS_DataScience/CSE599_DeepLearning/Final_Project/saved_results/siamNet_crossEntropy_res/siamese_net_crossEntropy.pt')

    criterion = CrossEntropyLoss().to(device)

    ans = test(net_new, device, test_dataloader_own, criterion)
    print("Ans: ", ans)
    return ans
are_the_samples_identical()