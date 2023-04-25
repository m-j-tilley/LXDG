import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import torch.utils.data
import numpy as np
import pickle

# Function for mean squared dot product
@torch.jit.script
def mean_squared_dotprod(gate, comparegate, olap):
    return torch.mean(torch.square(torch.mean(gate * comparegate, 1) - olap))

# Function for mean squared mean difference
@torch.jit.script
def mean_squared_meandiff(gate, olap):
    return torch.mean(torch.square(torch.mean(gate, 1) - olap))

# Train function for the model
def train(model, device, train_loader, optimizer, use_ewc=True, ewc_list=[], ewc_lambda=0.1, use_sparse=False, use_keepchange=False, sparse_lambda=100., keep_lambda=100., change_lambda=100., beta_change=0.):
    model.train()
    model = model.to(device)
    counter = 0
    data_seed = 0
    beta_sp = torch.Tensor([0.2]).to(device)
    beta_change = torch.Tensor([beta_change]).to(device)
    beta_keep = torch.Tensor([.3]).to(device)

    # Iterate through the batches
    for batch_idx, (data, target) in enumerate(train_loader):
        loss = 0

        data, target = data.to(device), target.to(device)
        target = target.reshape(-1)
        optimizer.zero_grad()
        output, gnewdnew = model(data)
        loss += F.nll_loss(F.log_softmax(output, dim=1), target)

        # Sparse regularization
        if use_sparse:
            for gate_id in range(len(gnewdnew)):
                loss += sparse_lambda * mean_squared_dotprod(gnewdnew[gate_id], gnewdnew[gate_id], beta_sp)

        # EWC and keep-change penalties
        for ewc in ewc_list:
            if use_ewc:
                loss += 0.5 * ewc_lambda * ewc.penalty(model).to(device)

            if use_keepchange:
                old_model = ewc.model.to(device)
                _, golddnew = old_model(data)
                id_max = np.shape(data)[0]

                np.random.seed(data_seed)
                indx = np.random.choice(np.arange(ewc.small_data_size), size=id_max)

                old_data = ewc.small_data[indx].to(device)
                data_seed += 1

                _, gnewdold = model(old_data)
                _, golddold = old_model(old_data)

                for gate_id in range(len(gnewdnew)):
                    loss += change_lambda * mean_squared_dotprod(gnewdnew[gate_id], gnewdold[gate_id], beta_change)
                    loss += keep_lambda * mean_squared_dotprod(gnewdold[gate_id], golddold[gate_id], beta_keep)

        loss.backward()
        optimizer.step()
        counter += 1

        # Memory management
        if len(ewc_list) > 1 and use_keepchange:
            for gate_id in range(len(gnewdnew)):
                gnewdnew[gate_id], gnewdold[gate_id], golddold[gate_id] = gnewdnew[gate_id].to("cpu"), gnewdold[gate_id].to("cpu"), golddold[gate_id].to("cpu")

            old_data, old_model = old_data.to("cpu"), old_model.to("cpu")
            del old_data, old_model, gnewdnew, gnewdold, golddold

        data, target, output = data.to("cpu"), target.to("cpu"), output.to("cpu")
        del data, target, output

        loss = loss.to("cpu")
        del loss
        torch.cuda.empty_cache()

    beta_sp, beta_change, beta_keep = beta_sp.to("cpu"), beta_change.to("cpu"), beta_keep.to("cpu")
    del beta_sp, beta_change, beta_keep

# Test function for the model
def test(model, device, test_loader, task_id=0, y_task_id=0, prep=0, dump_gates=True, name=''):
    with torch.no_grad():
        model.eval()
        test_loss = 0
        correct = 0
        totaldat = 0

        g1_all = np.zeros(model.nhidd1)
        g2_all = np.zeros(model.nhidd2)

        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output, gates = model(data)

            # Save gate values if dump_gates is True
            if dump_gates:
                g1, g2 = gates
                g1_all += np.sum(g1.cpu().detach().numpy(), axis=0)
                g2_all += np.sum(g2.cpu().detach().numpy(), axis=0)

            totaldat += len(data)
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

            data, target = data.to("cpu"), target.to("cpu")
            output = output.to("cpu")
            del data, target, output

        # Save gate values to files
        if dump_gates:
            fg1 = open(f'results/gate_vectors/{name}_gate_vector_1_trained_{task_id}_task_{y_task_id}.dat', 'wb')
            fg2 = open(f'results/gate_vectors/{name}_gate_vector_2_trained_{task_id}_task_{y_task_id}.dat', 'wb')

            pickle.dump(1. * g1_all / totaldat, fg1)
            pickle.dump(1. * g2_all / totaldat, fg2)

            fg1.close()
            fg2.close()

        model.train()
        return correct * 1. / totaldat