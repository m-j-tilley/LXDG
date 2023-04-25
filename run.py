import torch
import torchvision
from tqdm import tqdm
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pickle

from model import *
from methods import *
from task import *
from train import *

# Main function to run the experiments
def run_lxdg(config):
    
    num_epoch = config.get("num_epoch", 20)
    lr = config.get("lr", .001)
    TRAIN_BATCH = config.get("TRAIN_BATCH", 256)
    TEST_BATCH = config.get("TEST_BATCH", 1024)
    gate_lambdas = config.get("gate_lambdas", [1000.])
    ewc_lambdas = config.get("ewc_lambdas", [1000.])
    nhidd1 = config.get("nhidd1", 2000)
    nhidd2 = config.get("nhidd2", 2000)
    dump_gates = config.get("dump_gates", False)
    task_type = config.get("task_type", 'permuted_MNIST')
    input_size = config.get("npix_sq", 28*28)
    output_size = config.get("output_size", 10)
    use_ewc = config.get("use_ewc", True)
    use_sparse = config.get("use_sparse", True)
    use_keepchange = config.get("use_keepchange", True)
    online = config.get("online", False)
    ntasks = config.get("ntasks", 5)
    name = config.get("name", '')
    device = config.get("device", 0)
    rndseed = config.get("rndseed", 0) + 42
    include_gating_layers = config.get("include_gating_layers", True)
    random_xdg = config.get("random_xdg", False)
    
    # Iterate through gate and ewc lambdas
    for gate_lambda in gate_lambdas:
        for ewc_lambda in ewc_lambdas:
            # Initialize variables
            mean_task_acc_list = []
            acc_lists = []
            ewc_list = []
            mpf = 1.
            task_id = 0
            
            # Set include_gating_layers and random_xdg flags
            # Note: this is for the learned gate layers not the losses 
            if random_xdg:
                include_gating_layers=False
            
            # Create GatedModel instance
            model = GatedModel(include_gating_layers=include_gating_layers, input_size=input_size, output_size=output_size, nhidd1=nhidd1,nhidd2=nhidd2, random_xdg=random_xdg, device=device).to(device)

            ewc_old = None
            old_dataloaders = []

            # Training loop
            while mpf>0. and task_id<ntasks:
                
                optimizer = optim.Adam(model.parameters(), lr=lr)
                
                # Get train_loader based on task_type
                if task_type == 'permuted_MNIST':
                    train_loader = get_permuted_mnist_train_data(task_id, batch_size=TRAIN_BATCH, rndseed=rndseed)
                elif task_type == 'rotated_MNIST':
                    train_loader = get_rotated_mnist_train_data(task_id, batch_size=TRAIN_BATCH, rndseed=rndseed)
                else:
                    raise ValueError(f"Unsupported task_type: {task_type}")
                
                # Update task_id if random_xdg is True
                if random_xdg:
                    model.update_task_id(task_id)
                
                # Training for the current epoch
                for epoch in tqdm(range(num_epoch), desc=f"Training On Task {task_id}..."):
                    train(model, device, train_loader, optimizer, use_ewc=use_ewc, ewc_list=ewc_list, ewc_lambda=ewc_lambda, 
                          use_sparse=use_sparse, use_keepchange=use_keepchange, sparse_lambda=gate_lambda, keep_lambda=gate_lambda, 
                          change_lambda=gate_lambda)

                acc_list = []
                
                # Testing loop
                for task_test_id in range(task_id+1):
                    # Get test_loader based on task_type
                    if task_type == 'permuted_MNIST':
                        test_loader = get_permuted_mnist_test_data(task_test_id, batch_size=TRAIN_BATCH, rndseed=rndseed)
                    elif task_type == 'rotated_MNIST':
                        test_loader = get_rotated_mnist_test_data(task_test_id, batch_size=TRAIN_BATCH, rndseed=rndseed)
                    
                    # Update task_id if random_xdg is True
                    if random_xdg:
                        model.update_task_id(task_test_id)
                    
                    # Test the model and get accuracy
                    acc = test(model, device, test_loader, task_id=task_id, y_task_id=task_test_id, dump_gates=dump_gates, name=name)
                    acc_list.append(acc)

                acc_lists.append(acc_list)

                # Calculate mean task accuracy and update the list
                mean_task_acc_list.append(np.mean(acc_list))
                mpf = mean_task_acc_list[-1]
                print(f'Mean Accuracy Across {task_id+1} task(s): {mpf}')
                
                # Save results to a file
                with open(f'results/{task_type}_{name}.pkl', 'wb') as savf:
                    pickle.dump([mean_task_acc_list, acc_lists], savf)

                # Update ewc instances if use_ewc is True
                if use_ewc:
                    print('Using EWC... ')
                    if ewc_old == None:
                        ewc_new = EWC(model, train_loader, device, batch_size=TRAIN_BATCH)
                    else:
                        if online:
                            ewc_new = EWC(model, train_loader, device, batch_size=TRAIN_BATCH, old_fisher_sum=ewc_old._precision_matrices, old_dataloaders=old_dataloaders)
                        else:
                            ewc_new = EWC(model, train_loader, device, batch_size=TRAIN_BATCH)

                    old_dataloaders.append(train_loader)

                    if online:
                        ewc_list = [ewc_new]
                    else:
                        ewc_list.append(ewc_new)

                    ewc_old = ewc_new
                
                task_id += 1
                torch.cuda.empty_cache()
  

# Config for all experiments 

config = {
    "TRAIN_BATCH": 256,
    "output_size": 10,
    "device": 0,
    "rndseed": 0,
}

# Configs for permuted MNIST 

config_perm_LXDG_EWC = {
    "task_type": "permuted_MNIST",
    "name": f"perm_LXDG_EWC_{config['rndseed']}",
    "use_ewc": True,
    "use_sparse": True,
    "use_keepchange": True,
    "include_gating_layers": True,
    "random_xdg": False,
    "dump_gates": True,
}

config_perm_NOCON = {
    "task_type": "permuted_MNIST",
    "name": f"perm_NOCON_{config['rndseed']}",
    "use_ewc": False,
    "use_sparse": False,
    "use_keepchange": False,
    "include_gating_layers": True,
    "random_xdg": False,
    "dump_gates": False,
}

config_perm_EWC_only = {
    "task_type": "permuted_MNIST",
    "name": f"perm_EWC_only_{config['rndseed']}",
    "use_ewc": True,
    "use_sparse": False,
    "use_keepchange": False,
    "include_gating_layers": True,
    "random_xdg": False,
    "dump_gates": False,
}


config_perm_XDG_EWC = {
    "task_type": "permuted_MNIST",
    "name": f"perm_XDG_EWC_{config['rndseed']}",
    "use_ewc": True,
    "use_sparse": False,
    "use_keepchange": False,
    "include_gating_layers": False,
    "random_xdg": True,
    "dump_gates": False,
}


# Configs for rotated MNIST 

config_rot_LXDG_EWC = {
    "task_type": "rotated_MNIST",
    "name": f"rot_LXDG_EWC_{config['rndseed']}",
    "use_ewc": True,
    "use_sparse": True,
    "use_keepchange": True,
    "include_gating_layers": True,
    "random_xdg": False,
    "dump_gates": True,
}

config_rot_NOCON = {
    "task_type": "rotated_MNIST",
    "name": f"rot_NOCON_{config['rndseed']}",
    "use_ewc": False,
    "use_sparse": False,
    "use_keepchange": False,
    "include_gating_layers": True,
    "random_xdg": False,
    "dump_gates": False,
}

config_rot_EWC_only = {
    "task_type": "rotated_MNIST",
    "name": f"rot_EWC_only_{config['rndseed']}",
    "use_ewc": True,
    "use_sparse": False,
    "use_keepchange": False,
    "include_gating_layers": True,
    "random_xdg": False,
    "dump_gates": False,
}

config_rot_XDG_EWC = {
    "task_type": "rotated_MNIST",
    "name": f"rot_XDG_EWC_{config['rndseed']}",
    "use_ewc": True,
    "use_sparse": False,
    "use_keepchange": False,
    "include_gating_layers": False,
    "random_xdg": True,
    "dump_gates": False,
}


# Uncomment for various configurations

config_perm_LXDG_EWC.update(config)
run_lxdg(config_perm_LXDG_EWC)

#config_perm_NOCON.update(config)
#run_lxdg(config_perm_NOCON)

#config_perm_EWC_only.update(config)
#run_lxdg(config_perm_EWC_only)

#config_perm_XDG_EWC.update(config)
#run_lxdg(config_perm_XDG_EWC)

#config_rot_LXDG_EWC.update(config)
#run_lxdg(config_rot_LXDG_EWC)

#config_rot_NOCON.update(config)
#run_lxdg(config_rot_NOCON)

#config_rot_EWC_only.update(config)
#run_lxdg(config_rot_EWC_only)

#config_rot_XDG_EWC.update(config)
#run_lxdg(config_rot_XDG_EWC)
                