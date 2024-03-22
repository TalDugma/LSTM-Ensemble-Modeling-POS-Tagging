#NOTE : In order to run we need the  "nested_array" files, therefore before the run we need to run prepreocessing.py and then calc_embed.py

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import f1_score
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score
#random seed
torch.manual_seed(16)
np.random.seed(16)

train_file_path = "data/train.tagged"
dev_file_path = "data/dev.tagged"  
embedded_data= np.load("nested_array_train_25.npy",allow_pickle=True)



class LSTM(nn.Module):
    def __init__(self, input_size, hidden_dim, num_layers, output_dim):
        super(LSTM, self).__init__()  # Correctly call the superclass initializer
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True)  # Ensure batch_first=True for compatibility with your data
        self.fc = nn.Linear(hidden_dim, output_dim)  # Assuming output_dim is the correct size for your task
        self.sigmoid = nn.Sigmoid()  # Instantiate Sigmoid here for use in forward method

    def forward(self, x):
        # Automatically initialize hidden and cell states to zeros
        h0 = torch.zeros(self.lstm.num_layers, x.size(0), self.hidden_dim)
        c0 = torch.zeros(self.lstm.num_layers, x.size(0), self.hidden_dim)

        # Forward propagate the LSTM
        output, _ = self.lstm(x, (h0, c0))
        
        # Decode the hidden state of the last time step, assume x is of shape (batch, sequence, feature)
        output = self.fc(output)
        
        # Apply the sigmoid activation
        output = self.sigmoid(torch.squeeze(output,-1))
        
        return output
    

X_train = []
y_train = []
for sentence in embedded_data:
    sen_x =[]
    sen_y =[]
    for word_embedding, tag in sentence:
        sen_x.append(word_embedding)
        sen_y.append(int(tag))
    X_train.append(sen_x)
    y_train.append(sen_y)



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#pad the data and create the data loader
X_train = [torch.tensor(sentence, dtype=torch.float) for sentence in X_train]
X_train = pad_sequence(X_train, batch_first=True, padding_value=0.0)
y_train = [torch.tensor(sentence, dtype=torch.int) for sentence in y_train]
y_train = pad_sequence(y_train, batch_first=True, padding_value=0.0)
data=TensorDataset(torch.tensor(X_train, dtype=torch.float), torch.tensor(y_train, dtype=torch.float))
#set hyperparameters and call model
learning_rate =0.0005
num_epochs = 40
model = LSTM(input_size=25,hidden_dim=32,num_layers=3,output_dim=1)  
model.to(device)
batch_size=32
train_loader = DataLoader(data, batch_size=batch_size, shuffle=True)
lengths = torch.tensor([len(sentence) for sentence in embedded_data], dtype=torch.long)
optimizer= optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.BCELoss()
epoch_losses = []
#train model
for epoch in tqdm(range(num_epochs)):
    model.train()
    batch_losses = []
    for words,tags in train_loader:
        words,tags = words.to(device),tags.to(device)
        output = model(words)
        loss = criterion(output, tags)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        batch_losses.append(loss.item())
    epoch_losses.append(np.mean(batch_losses))
    if (epoch+1) % 10 == 0:
        print(f'Epoch {epoch+1}, loss: {epoch_losses[-1]}')

#predict on the train data
model.eval()
with torch.no_grad():
    X_train = np.array(X_train)
    y_pred_train = model(torch.tensor(X_train, dtype=torch.float).to(device))
    y_pred_train = y_pred_train.cpu().numpy() 
    y_pred_train = y_pred_train > 0.15     

# Calculate the F1 score on the training data
embedded_data_dev= np.load("nested_array_dev_25.npy",allow_pickle=True)
y_pred_train = y_pred_train.flatten()
y_train = y_train.cpu().numpy()
y_train = y_train.flatten()
y_pred_train = y_pred_train.astype(int)

f1_train = f1_score(y_train, y_pred_train,average= 'binary')
print("F1 score on train data:", f1_train)

X_dev=[]
y_dev=[]
for sentence in embedded_data_dev:
    sen_x =[]
    sen_y =[]
    for word_embedding, tag in sentence:
        sen_x.append(word_embedding)
        sen_y.append(int(tag))
    X_dev.append(sen_x)
    y_dev.append(sen_y)

#pad the data and create the data loader
X_dev = [torch.tensor(sentence, dtype=torch.float) for sentence in X_dev]
X_dev = pad_sequence(X_dev, batch_first=True, padding_value=0.0)
y_dev = [torch.tensor(sentence, dtype=torch.int) for sentence in y_dev]
y_dev = pad_sequence(y_dev, batch_first=True, padding_value=0.0)
data=TensorDataset(torch.tensor(X_dev, dtype=torch.float), torch.tensor(y_dev, dtype=torch.float))
test_loader = DataLoader(data, batch_size=batch_size, shuffle=True)

#  f1 score for the dev
with torch.no_grad():
    X_dev = np.array(X_dev)
    y_pred_dev = model(torch.tensor(X_dev, dtype=torch.float).to(device))
    y_pred_dev = y_pred_dev.cpu().numpy() 
    y_pred_dev = y_pred_dev > 0.15


y_dev = y_dev.cpu().numpy()
y_pred_dev = y_pred_dev.flatten()
y_dev = y_dev.flatten()
y_pred_dev = y_pred_dev.astype(int)


f1_dev = f1_score(y_dev, y_pred_dev,average= 'binary')
print("F1 score on dev data:", f1_dev)

precision = precision_score(y_dev, y_pred_dev, average='binary')  # Use 'micro', 'macro', 'weighted', or 'samples' for multi-class
recall = recall_score(y_dev, y_pred_dev, average='binary')  # Same as above for averaging

# Print the precision and recall

print(f'Precision: {precision}')
print(f'Recall: {recall}')