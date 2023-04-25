import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import torch.utils.data
import numpy as np

## Then define the model class
class GatedModel(nn.Module):
    def __init__(self,include_gating_layers=True, input_size = 784, output_size=10,nhidd1=2000,nhidd2=2000, random_xdg=False, device=0):
        super(GatedModel, self).__init__()
        
        self.nhidd1=nhidd1
        self.nhidd2=nhidd2
        
        self.random_xdg = random_xdg
        self.task_id = 0
        
        if self.random_xdg:
            max_tasks = 200
            self.gates = torch.Tensor(np.random.choice([0, 1], size=(max_tasks,2,2000), p=[0.8,0.2])).to(device)
      
        #fully connected layer
        self.fc1 = nn.Linear(input_size, nhidd1)
        self.fc2 = nn.Linear(nhidd1, nhidd2)
        self.out = nn.Linear(nhidd2, output_size)
        
        self.flat = nn.Flatten()
        
        self.include_gating_layers = include_gating_layers
        
        self.fc1drp = nn.Dropout(p=0.5)
        self.fc2drp = nn.Dropout(p=0.5)
    
        if self.include_gating_layers:
            
            self.g1fc1 = nn.Linear(input_size, 400)
            self.g1fc2 = nn.Linear(400, 400)
            self.g1out = nn.Linear(400, nhidd1)

            self.g2fc1 = nn.Linear(input_size,400)
            self.g2fc2 = nn.Linear(400, 400)
            self.g2out = nn.Linear(400, nhidd2)

            self.drp1 = nn.Dropout(p=0.5)
            self.drp2 = nn.Dropout(p=0.5)
            self.drp3 = nn.Dropout(p=0.5)
            self.drp4 = nn.Dropout(p=0.5)
            
    def update_task_id(self, tid):
        self.task_id=tid

    def forward(self, inp):

        x = self.flat(inp)
        g1 = x
        g2 = x 
        
        ###########
        # Gating layers
        if self.include_gating_layers:
            
            g1 = F.relu(self.g1fc1(g1))
            g1 = self.drp1(g1)
            g1 = F.relu(self.g1fc2(g1))
            g1 = self.drp2(g1)
            g1 = torch.sigmoid(self.g1out(g1))

            g2 = F.relu(self.g2fc1(g2))
            g2 =  self.drp3(g2)
            g2 = F.relu(self.g2fc2(g2))
            g2 =  self.drp4(g2)
            g2 = torch.sigmoid(self.g2out(g2))
        ############
        
        if self.random_xdg:
            g1 = self.gates[self.task_id, 0, :]
            g2 = self.gates[self.task_id, 1, :]
            
        x = F.relu(self.fc1(x))
        x = self.fc1drp(x)
        
        if self.include_gating_layers or self.random_xdg: 
            x = g1*x
        
        x = F.relu(self.fc2(x))
        x = self.fc2drp(x)
        
        if self.include_gating_layers or self.random_xdg:
            x = g2*x
        
        x = self.out(x)
        
        if self.include_gating_layers or self.random_xdg:
            return x, [g1, g2]
        else:
            return x, []
        

