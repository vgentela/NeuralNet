# -*- coding: utf-8 -*-
"""
Created on Sun Mar 31 14:37:12 2024

@author: Varshney
"""
import numpy as np
import pandas as pd
from pathlib import Path
import torch
from torch import nn
import tensorflow.keras.utils as tf
from torch.utils.data import TensorDataset, DataLoader
from matplotlib import pyplot as plt
path1 = Path(f"{Path.cwd()}\\stationary_output.csv")
path2 = Path(f"{Path.cwd()}\\vib_30hz_90degs.csv")
path3 = Path(f"{Path.cwd()}\\vib_30hz_upright.csv")
path4 = Path(f"{Path.cwd()}\\vib_30hz_rig.csv")

assert path1.exists()
assert path2.exists()
assert path3.exists()
assert path4.exists()
#%% Loading the data
correction_data = pd.read_csv(path1)

correction_df = correction_data.copy() # This offset is the 0g.
correction_df = correction_df.loc[:correction_df.index[correction_df.index<=121657].max()]

medium_1_data = pd.read_csv(path3)
medium_1_90_data = pd.read_csv(path2) 
medium_2_data = pd.read_csv(path4)

medium_1  = medium_1_data.copy()
medium_2 = medium_2_data.copy()
medium_1_90 = medium_1_90_data.copy()

medium_1 =  medium_1.loc[:medium_1.index[medium_1.index<=121657].max()]
medium_2 = medium_2.loc[:medium_2.index[medium_2.index<=121657].max()]

medium_1_90  = medium_1_90.loc[:medium_1_90.index[medium_1_90.index<=121658].max()]

medium_1 = medium_1 - correction_df
medium_2 = medium_2 - correction_df

medium_1['medium'] = 1
medium_2['medium'] = 2
#%%

#%% Labelling


def labelling(acc_thresholds,gyro_thresholds, df) -> pd.DataFrame():

    binary_labels = []
    num_cols = len(df.columns) - 2  # Subtracting 2 to exclude 'medium' and 'risk' columns
    
    for _, row in df.iterrows():
        acc_exceeds = sum(abs(row[col]) >= threshold for col, threshold in zip(df.columns, acc_thresholds) if col.startswith('acce'))
        gyro_exceeds = sum(abs(row[col]) >= threshold for col, threshold in zip(df.columns, gyro_thresholds) if col.startswith('gyro_'))
        total_exceeds = acc_exceeds + gyro_exceeds

        if total_exceeds > 0:
            binary_labels.append(1)  # Sample exceeds threshold
        else:
            binary_labels.append(0)  # Sample does not exceed threshold
    
    df['label'] = binary_labels
    return df


def pre_sampling(df,acc_risk,gyro_risk,medium):
    df_ret = pd.DataFrame()
    for i in range(0,len(df),1031):
        df_res = labelling(acc_risk, gyro_risk,df[i:i+1031])
        df_ret = pd.concat([df_ret,df_res])
    return df_ret

#%% Labelling
acc_risk_1 = [1.9,1.0,1.95] 
gyro_risk_1 = [120,300,40]

acc_risk_2 = [1, 0.9, 0.9]
gyro_risk_2 = [20,15,20]

medium_1 = pre_sampling(medium_1,acc_risk_1,gyro_risk_1,1)
medium_2 = pre_sampling(medium_2,acc_risk_2, gyro_risk_2, 2)

#%%% Sampling
def sampling(dfs)-> pd.DataFrame:
    train_data = []
    comb_df = pd.concat(dfs)  # Corrected concatenation
    print(len(comb_df))
    for i in range(0, len(comb_df), 1031):
        chunk = comb_df.iloc[i:i + 1031]
        proportion_label_1 = chunk['label'].sum() / len(chunk)

        chunk_label = 1 if proportion_label_1 >= (0.69 if chunk['medium'].iloc[0] == 1 else 0.679) else 0
        comb_df.at[i, 'risk'] = chunk_label

    comb_df.drop(['label'], axis=1, inplace=True)
    comb_df = pd.get_dummies(comb_df, columns=['medium'])
    
    for i in range(0, len(comb_df), 1031):
        chunk = comb_df.iloc[i:i + 1031]

        feature = torch.tensor(chunk.drop(['risk', 'avg_hz', 'medium_1.0', 'medium_2.0'], axis=1).values, dtype=torch.float32)
        risk_target = torch.tensor(chunk['risk'].iloc[0], dtype=torch.float32)  # Assuming binary label
        medium_target = torch.tensor(chunk[['medium_1.0', 'medium_2.0']].iloc[0].values, dtype=torch.float32)

        train_data.append((feature, torch.cat((risk_target.unsqueeze(0), medium_target), dim=0)))  # Concatenate risk and medium targets

    

    return train_data
#%%
sampled_data = sampling([medium_1,medium_2])
#%%
features = [sampled_data[i][0] for i in range(len(sampled_data)-1)] 
targets = [sampled_data[i][1] for i in range(len(sampled_data)-1)]

feature_tensor = torch.stack(features)
target_tensor = torch.stack(targets)

#%%

data_set = TensorDataset(feature_tensor,target_tensor)

train_data,test_data = torch.utils.data.random_split(data_set,[9/10,1/10])
train_loader =  DataLoader(train_data,batch_size = 20,shuffle = True,drop_last=True)
test_loader = DataLoader(test_data,batch_size=20,shuffle = True,drop_last=True)

#%% Network

class Classifier(nn.Module):
    
    def __init__(self,feature_nodes,hidden_nodes =3000):
        super(Classifier,self).__init__()
        self.net = nn.Sequential(nn.Linear(1031*feature_nodes,hidden_nodes),
                                 nn.ReLU(),
                                 nn.Linear(hidden_nodes,hidden_nodes),
                                 nn.ReLU(),
                                 nn.Linear(hidden_nodes,1500),
                                 nn.ReLU(),
                                 nn.Linear(1500, 750),
                                 nn.ReLU(),
                                 nn.Linear(750,350),
                                 nn.ReLU(),
                                 nn.Linear(350,50),
                                 nn.ReLU(),
                                 nn.Linear(50, 3)
                                 )
    
    def forward(self,data):
        data = data.view(data.size(0),-1)
        return torch.sigmoid(self.net(data))
    
    

#%% Train the model

def training(model,criterion,optimizer,train_loader,epochs = 40):
    
    losses =[]
    accs = []
    for _ in range(epochs):
        correct = 0
        print(f'Epoch {_}')
        print()
        for idx, (data,target) in enumerate(train_loader):
            
            optimizer.zero_grad()
            #binary_label = target[:, 0].max(dim=0)[0]
            output = model.forward(data)
            #print(output)
            loss = criterion(output,target)
            losses.append(loss.item())
            
            b_correct= torch.sum(torch.abs(output-target) <0.5)
            correct += b_correct
            
            loss.backward()
            optimizer.step()
            
        acc = float(correct)/len(train_loader)
        accs.append(acc)   
        
        print('Average Train Loss:',np.mean(losses),sep=':')
        print('Accuracy:',accs[_],sep =':')
        print('----------------------------------------------------------------')
    
    return model,losses,accs
#%%
model = Classifier(feature_tensor.shape[2])
optimizer = torch.optim.Adam(model.parameters(), lr = 0.0001)
criterion = nn.BCEWithLogitsLoss()
train_loader = train_loader

model,loss,acc = training(model, criterion, optimizer, train_loader)
#%%

