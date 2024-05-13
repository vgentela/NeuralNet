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
from torch.utils.data import TensorDataset, DataLoader
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.metrics import multilabel_confusion_matrix,confusion_matrix

#%%
path1 = Path(f"{Path.cwd()}\\stationary_output.csv")
path2 = Path(f"{Path.cwd()}\\vib_30hz_90degs.csv")
path3 = Path(f"{Path.cwd()}\\vib_30hz_upright.csv")
path4 = Path(f"{Path.cwd()}\\vib_30hz_rig.csv")

assert path1.exists()
assert path2.exists()
assert path3.exists()
assert path4.exists()
#%% Loading the data
def load_data(limit:int):
    """
    Load data from CSV files and preprocesses it for further analysis.

    Args:
        limit (int): Index limit for slicing the data.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: Tuple containing two pandas DataFrames
            representing the loaded and preprocessed data.
    """

    correction_data = pd.read_csv(path1)
    
    correction_df = correction_data.copy() # This offset is the 0g.
    correction_df = correction_df.loc[:correction_df.index[correction_df.index<=limit].max()]
    
    medium_1_data = pd.read_csv(path3)
    medium_1_90_data = pd.read_csv(path2) 
    medium_2_data = pd.read_csv(path4)
    
    medium_1  = medium_1_data.copy()
    medium_2 = medium_2_data.copy()
    medium_1_90 = medium_1_90_data.copy()
    
    medium_1 =  medium_1.loc[:medium_1.index[medium_1.index<=limit].max()]
    medium_2 = medium_2.loc[:medium_2.index[medium_2.index<=limit].max()]
    
    medium_1_90  = medium_1_90.loc[:medium_1_90.index[medium_1_90.index<=limit].max()]
    
    medium_1 = medium_1 - correction_df
    medium_2 = medium_2 - correction_df
    
    medium_1['medium'] = 1
    medium_2['medium'] = 2

    return medium_1, medium_2
#%% Labelling
def labelling(acc_thresholds,gyro_thresholds, df) -> pd.DataFrame():
    
    """
    Apply labeling to the dataset based on acceleration and gyroscope thresholds.

    Args:
        acc_thresholds (List[float]): Thresholds for acceleration.
        gyro_thresholds (List[float]): Thresholds for gyroscope.
        df (pd.DataFrame): DataFrame containing sensor data.

    Returns:
        pd.DataFrame: DataFrame with additional 'label' column indicating if the sample exceeds thresholds.
    """
    
    binary_labels = []
    #num_cols = len(df.columns) - 2  # Subtracting 2 to exclude 'medium' and 'risk' columns
    
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

#%%
def pre_sampling(df,acc_risk,gyro_risk,fs,medium):
    
    """
   Perform preprocessing and sampling of the data.

   Args:
       df (pd.DataFrame): DataFrame containing sensor data.
       acc_risk (List[float]): Acceleration thresholds for risk labeling.
       gyro_risk (List[float]): Gyroscope thresholds for risk labeling.
       fs (int): Sampling frequency.
       medium (int): Medium identifier.

   Returns:
       pd.DataFrame: Preprocessed and sampled DataFrame.
   """
    df_ret = pd.DataFrame()
    for i in range(0,len(df),fs):
        df_res = labelling(acc_risk, gyro_risk,df[i:i+fs])
        df_ret = pd.concat([df_ret,df_res])
    return df_ret

#%% Labelling
acc_risk_1 = [1.9,1.0,1.95] 
gyro_risk_1 = [120,300,40]

acc_risk_2 = [1, 0.9, 0.9]
gyro_risk_2 = [20,15,20]

medium_1,medium_2 = load_data(121657)

medium_1 = pre_sampling(medium_1,acc_risk_1,gyro_risk_1,1031,1)
medium_2 = pre_sampling(medium_2,acc_risk_2, gyro_risk_2, 1031,2)

#%%% Sampling
def sampling(dfs,fs)-> tuple[torch.Tensor, torch.Tensor]:
    
    """
   Perform sampling on the combined DataFrame.

   Args:
       dfs (List[pd.DataFrame]): List of DataFrames to be sampled.
       fs (int): Sampling frequency.

   Returns:
       tuple[torch.Tensor, torch.Tensor]: Tuples containing features and targets.
   """
   
    train_data = []
    comb_df = pd.concat(dfs)  
    print(len(comb_df))
    for i in range(0, len(comb_df), fs):
        chunk = comb_df.iloc[i:i + fs]
        proportion_label_1 = chunk['label'].sum() / len(chunk)

        chunk_label = 1 if proportion_label_1 >= (0.69 if chunk['medium'].iloc[0] == 1 else 0.679) else 0
        comb_df.at[i, 'risk'] = chunk_label

    comb_df.drop(['label'], axis=1, inplace=True)
    comb_df = pd.get_dummies(comb_df, columns=['medium'])
    
    for i in range(0, len(comb_df), fs):
        chunk = comb_df.iloc[i:i + fs]

        feature = torch.tensor(chunk.drop(['risk', 'avg_hz', 'medium_1.0', 'medium_2.0'], axis=1).values, dtype=torch.float32)
        risk_target = torch.tensor(chunk['risk'].iloc[0], dtype=torch.float32)  # Assuming binary label
        medium_target = torch.tensor(chunk[['medium_1.0', 'medium_2.0']].iloc[0].values, dtype=torch.float32)

        train_data.append((feature, torch.cat((risk_target.unsqueeze(0), medium_target), dim=0)))  # Concatenate risk and medium targets

    features = [train_data[i][0] for i in range(len(train_data)-1)] 
    targets = [train_data[i][1] for i in range(len(train_data)-1)]

    feature_tensor = torch.stack(features)
    target_tensor = torch.stack(targets)


    return feature_tensor,target_tensor
#%%
features,targets= sampling([medium_1,medium_2],1031)


#%% Network

class Classifier(nn.Module):
    
    """
    Neural network classifier model.

    Attributes:
        net (nn.Sequential): Sequential neural network layers.
    """

    def __init__(self,feature_nodes,hidden_nodes =3000):
        """
        Initialize the Classifier model.

        Args:
            feature_nodes (int): Number of input feature nodes.
            hidden_nodes (int): Number of nodes in the hidden layer.
        """
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
        
        """
        Forward pass through the neural network.

        Args:
            data (torch.Tensor): Input data.

        Returns:
            torch.Tensor: Output predictions.
        """
        
        data = data.view(data.size(0),-1)
        return torch.sigmoid(self.net(data))
    
    

#%% Train the model

def training(model,criterion,optimizer,train_data,epochs = 10):
    """
    Train the neural network model.

    Args:
        model (Classifier): Neural network model.
        criterion (nn.Module): Loss criterion.
        optimizer (torch.optim.Optimizer): Optimization algorithm.
        train_data (TensorDataset): Training data.
        epochs (int): Number of training epochs.

    Returns:
        tuple[Classifier, List[float], List[float]]: Tuple containing: trained model, list of epoch losses, and list of accuracies.
    """
    
    train_loader =  DataLoader(train_data,batch_size = 21,shuffle = True,drop_last=True)
    losses =[]
    epoch_loss =[]
    accs = []
    for _ in range(epochs):
        correct = 0
        total_samples = 0
        print(f'Epoch {_}')
        print()
        for idx, (data,target) in enumerate(train_loader):
            #print(len(data))            
            optimizer.zero_grad()
            #binary_label = target[:, 0].max(dim=0)[0]
            output = model.forward(data)
            #print(output)
            loss = criterion(output,target)
            losses.append(loss.item())
            
            correct += (output == target).sum().item()
            total_samples += target.size(0)

            
            loss.backward()
            optimizer.step()
            
        
        acc = correct/total_samples
        
        accs.append(acc)   
        epoch_loss.append(np.mean(losses))
        print('Average Train Loss:',epoch_loss[-1],sep=':')
        print('Accuracy:',acc,sep =':')
        print('----------------------------------------------------------------')
        
    plt.figure(figsize=(10, 5))
    plt.plot(range(len(epoch_loss)), epoch_loss, label='Loss', color='blue')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    plt.figure(figsize=(10, 5))
    plt.plot(range(len(accs)), accs, label='Accuracy', color='green')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.show()
    return model,epoch_loss,accs
#%%
data_set = TensorDataset(features,targets)

train_data,test_data = torch.utils.data.random_split(data_set,[9/10,1/10])

model = Classifier(6)
optimizer = torch.optim.Adam(model.parameters(), lr = 0.0001)
criterion = nn.BCEWithLogitsLoss()
train_data = train_data

model,loss,acc = training(model, criterion, optimizer, train_data)
#%% Evaluation
def eval(model,test_data):
    
    """
    Evaluate the trained model on test data and plot confusion matrix.

    Args:
        model (Classifier): Trained neural network model.
        test_data (TensorDataset): Test data.
    """
    
    test_loader = DataLoader(test_data,batch_size=21,shuffle = True,drop_last=True)

    true_labels = []
    predicted_labels = []
    
    # Evaluate the model on the test set
    model.eval()
    for data, target in test_loader:
        output = model(data)
        predicted_labels.extend(output.view(-1, 3).tolist())
        true_labels.extend(target.view(-1, 3).tolist())
    
    
    true_labels_binary_risk = np.array([1 if label[0] == 1.0 else 0 for label in true_labels])
    true_labels_binary_mediums = np.array([[1, 0] if (label[1], label[2]) == (1.0, 0.0) else [0, 1] for label in true_labels])
    
    # Convert predicted labels to binary format
    predicted_labels_binary_risk = np.array([1 if label[0] >= 0.5 else 0 for label in predicted_labels])
    predicted_labels_binary_mediums = np.array([[1, 0] if (label[1], label[2]) == (1.0, 0.0) else [0, 1] for label in predicted_labels])
    

    true_labels_binary = np.vstack((true_labels_binary_risk, true_labels_binary_mediums.T)).T
    predicted_labels_binary = np.vstack((predicted_labels_binary_risk, predicted_labels_binary_mediums.T)).T
    
    # Calculate confusion matrix
    conf_matrix = multilabel_confusion_matrix(true_labels_binary, predicted_labels_binary)
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    for i, label in enumerate(['Risk', ['Medium 1', 'Medium 2']]):
        plt.subplot(1, 2, i + 1)
        if i<1:
            sns.heatmap(conf_matrix[i], annot=True, fmt='d', cmap='Blues', xticklabels=['No' + label, label], yticklabels=['No' + label, label])
        else:
            sns.heatmap(conf_matrix[i], annot=True, fmt='d', cmap='Blues', xticklabels=[label[0], label[1]], yticklabels=[label[0], label[1]])
            
        plt.xlabel('Predicted Labels')
        plt.ylabel('True Labels')
        plt.title(f'Confusion Matrix for {label}')
    plt.tight_layout()
    plt.show()
    

#%%
eval(model,test_data)