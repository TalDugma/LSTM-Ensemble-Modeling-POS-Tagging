#NOTE : In order to run we need the  "nested_array" files, therefore before the run we need to run prepreocessing.py and then calc_embed.py

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score, recall_score
torch.manual_seed(16)
np.random.seed(16)
class linear_FFN(nn.Module):
    
    def __init__(self,input_dim,output_dim) -> None:
        super().__init__()
        self.hidden = nn.Sequential(nn.Linear(input_dim, 32),
                                    nn.Linear(32,16),
                                    nn.Linear(16,4))

        self.output_layer =nn.Linear(4,output_dim)
    
    def forward(self,x):
        x = self.output_layer(self.hidden(x))
        return torch.sigmoid(x)
        

train_file_path = "data/train.tagged"
dev_file_path = "data/dev.tagged"  
embedded_data= np.load("nested_array_train_25.npy",allow_pickle=True)

#load data
X_train = []
y_train = []
for sentence in embedded_data:
    for word_embedding_and_tag_index in range(len(sentence)):
        cur_word = sentence[word_embedding_and_tag_index][0]
        tag = sentence[word_embedding_and_tag_index][1]
        X_train.append(cur_word)
        y_train.append(int(tag))

X_train = np.array(X_train)
y_train = np.array(y_train)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

data=TensorDataset(torch.tensor(X_train, dtype=torch.float), torch.tensor(y_train, dtype=torch.float)) 

#activate model and choose hyperparameters
model = linear_FFN(input_dim = 25,output_dim=1)
model.to(device)
learning_rate = 0.0005
criterion = nn.BCELoss()
optimizer= optim.Adam(model.parameters(), lr=learning_rate)
num_epochs=20
batch_size=32
train_loader = DataLoader(data, batch_size=batch_size, shuffle=True)
epoch_losses = []

#train the model
for epoch in range(num_epochs):
    model.train()
    batch_losses = []
    for word,tag in train_loader:
        word,tag = word.to(device),tag.to(device)
        output = model(word)
        loss = criterion(output.view(-1), tag)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        batch_losses.append(loss.item())
    epoch_losses.append(np.mean(batch_losses))
    if (epoch+1) % 1 == 0:
        print(f'Epoch {epoch+1}, loss: {epoch_losses[-1]}')

#load dev data
embedded_data_dev= np.load("nested_array_dev_25.npy",allow_pickle=True)

X_dev=[]
y_dev=[]
for sentence in embedded_data_dev:
    for word_embedding_and_tag_index in range(len(sentence)):
        cur_word = sentence[word_embedding_and_tag_index][0]
        tag = sentence[word_embedding_and_tag_index][1]
        X_dev.append(cur_word)
        y_dev.append(int(tag))
X_dev = np.array(X_dev)
y_dev = np.array(y_dev)
data = TensorDataset(torch.tensor(X_dev, dtype=torch.float), torch.tensor(y_dev, dtype=torch.float))
test_loader = DataLoader(data, batch_size=batch_size, shuffle=True)
model.eval() 

# f1 score for the dev data
with torch.no_grad():
    y_pred_dev = model(torch.tensor(X_dev, dtype=torch.float).to(device))
    y_pred_dev = y_pred_dev.cpu().numpy() 
    threshold = 0.3
    y_pred_dev = y_pred_dev > threshold

f1_dev = f1_score(y_dev, y_pred_dev)


# Calculate the F1 score on the training data
with torch.no_grad():
    y_pred_train = model(torch.tensor(X_train, dtype=torch.float).to(device))
    y_pred_train = y_pred_train.cpu().numpy() 
    threshold = 0.3
    y_pred_train = y_pred_train > threshold

f1_train = f1_score(y_train, y_pred_train)
#print recall and precision
precision_train = precision_score(y_train, y_pred_train)
recall_train = recall_score(y_train, y_pred_train)

print("F1 score on train data:", f1_train)
print("F1 score on dev data:", f1_dev)
print("Precision on train data:", precision_train)
print("Recall on train data:", recall_train)
