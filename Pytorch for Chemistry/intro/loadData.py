import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset
from sklearn.model_selection import train_test_split

from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem

target_enzyme = '2c19' # total 5 enzymes in repo [1a2, 2c19, 2c9, 2d6, 3a4]
data_path = 'https://raw.githubusercontent.com/shuan4638/MCdropout-CYP450Classifcation/master/data/CYP_%s.csv' % target_enzyme
data = pd.read_csv(data_path, header = None)
data.columns = ['SMILES', 'label']

def smiles2fp(smiles):
    mol = Chem.MolFromSmiles(smiles)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
    arr = np.zeros((1,))
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr

data["FPs"] = data.SMILES.apply(smiles2fp)
X = np.stack(data.FPs.values)
y = np.stack(data.label.values)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size = 128

def numpy2tensor(array, device):
    return torch.tensor(array, device = device).float()

X_train, X_test, y_train, y_test = train_test_split(X, y,  test_size=0.10)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train,  test_size=0.11)

train_dataset = TensorDataset(numpy2tensor(X_train, device), numpy2tensor(y_train, device))
val_dataset = TensorDataset(numpy2tensor(X_val, device), numpy2tensor(y_val, device))
test_dataset = TensorDataset(numpy2tensor(X_test, device), numpy2tensor(y_test, device))

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                          batch_size=batch_size,
                                          shuffle=True)
val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)