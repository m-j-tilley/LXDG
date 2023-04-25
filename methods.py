import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import torch.utils.data
from copy import deepcopy

from tqdm import tqdm
import numpy as np

# Function to get a subset of data from a given loader
def get_old_data(loader, nsamples=100, seed=0):
    np.random.seed(seed)
    indx = np.random.randint(len(loader.dataset), size=(nsamples,))
    samp = torch.utils.data.SubsetRandomSampler(indx)
    temp_dl = torch.utils.data.DataLoader(loader.dataset, sampler=samp, batch_size=len(indx))
    _, (old_data, _) = next(enumerate(temp_dl))
    return old_data

# Function to get random data from a list of loaders
def get_random_data(loader_list, nsamples=100, seed=0):
    old_data_list = []
    for loader in loader_list:
        np.random.seed(seed)
        indx = np.random.randint(len(loader.dataset), size=(nsamples,))
        samp = torch.utils.data.SubsetRandomSampler(indx)
        temp_dl = torch.utils.data.DataLoader(loader.dataset, sampler=samp, batch_size=len(indx))
        _, (old_data, _) = next(enumerate(temp_dl))
        old_data_list.append(old_data)

    randomised_data = old_data.detach().clone()

    for i in range(nsamples):
        idx = np.random.randint(len(loader_list))
        randomised_data[i] = old_data_list[idx][i]

    return randomised_data

# EWC (Elastic Weight Consolidation) Class
class EWC(object):
    def __init__(self, model, dataset, device, batch_size=256, old_fisher_sum=None, old_dataloaders=[]):
        self.model = model
        self.dataloader = dataset
        self.batch_size = batch_size        
        self.device = device

        # Extract model parameters that require gradient
        self.params = {n: p for n, p in self.model.named_parameters() if p.requires_grad}
        self._means = {}
        self._precision_matrices = self._diag_fisher()

        # Add old Fisher sum to precision matrices if provided
        if old_fisher_sum is not None:
            for n, p in self.model.named_parameters():
                self._precision_matrices[n].data += old_fisher_sum[n].data

        # Save means of model parameters
        for n, p in deepcopy(self.params).items():
            self._means[n] = Variable(p.data).cpu()

        self.model = deepcopy(model).cpu()

        self.small_data_size = len(self.dataloader.dataset)
        
        # Get small dataset based on provided dataloaders
        if len(old_dataloaders) == 0:
            self.small_data = get_old_data(self.dataloader, nsamples=self.small_data_size)
        else:
            self.small_data = get_random_data([self.dataloader] + old_dataloaders, nsamples=self.small_data_size)

    # Function to compute diagonal Fisher information matrix
    def _diag_fisher(self):
        precision_matrices = {}
        for n, p in deepcopy(self.params).items():
            p.data.zero_()
            precision_matrices[n] = Variable(p.data).to(self.device)

        self.model.eval()
        for _, (batcheddata, _) in enumerate(self.dataloader):
            for ewc_data in batcheddata:        
                self.model.zero_grad()
                ewc_data = Variable(ewc_data).to(self.device)
                ewc_data = torch.unsqueeze(ewc_data, dim=0)
                output, _ = self.model(ewc_data)
                label = torch.argmax(output, 1)
                logprob = F.log_softmax(output, dim=1)[:, label]
                logprob.backward()

                # Update precision matrices with gradients
                for n, p in self.model.named_parameters():
                    if p.requires_grad:
                        precision_matrices[n].data += p.grad.data ** 2 / (len(self.dataloader.dataset))

        # Move precision matrices to CPU
        for n, p in self.model.named_parameters():
            if p.requires_grad:
                precision_matrices[n] = precision_matrices[n].to("cpu")

        precision_matrices = {n: p for n, p in precision_matrices.items()}

        output, label, ewc_data = output.to("cpu"), label.to("cpu"), ewc_data.to("cpu")
        del output, label, ewc_data

        return precision_matrices

    # Function to compute EWC penalty based on the given model
    def penalty(self, model: nn.Module):
        loss = 0
        for n, p in model.named_parameters():
            if p.requires_grad:
                # Move precision matrices and means to the device
                fish_info = self._precision_matrices[n].to(self.device)
                p_old = self._means[n].to(self.device)

                # Compute EWC penalty for the current parameter
                _loss = fish_info * torch.square(p - p_old)
                loss += _loss.sum()

        return loss.to(self.device)
