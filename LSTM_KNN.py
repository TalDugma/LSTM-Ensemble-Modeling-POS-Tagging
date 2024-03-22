import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
from sklearn.neighbors import KNeighborsClassifier
#random seed
torch.manual_seed(16)
np.random.seed(16)

train_file_path = "data/train.tagged"
dev_file_path = "data/dev.tagged"  
embedded_data= np.load("nested_array_train_25.npy",allow_pickle=True)
embedded_data_test= np.load("nested_array_test_25.npy",allow_pickle=True)

def lenths(X_dev):#for getting the length of the sentences
    lenthim=[]
    for i in range(len(X_dev)):
        lenth=len(X_dev[i])
        lenthim.append(lenth)
    return lenthim
    
def unpad(y_pred_test,x_dev):#for unpading the predictions
    lenthim=lenths(x_dev)
    unpaded_y_pred=[]
    for i in range(len(y_pred_test)):
        j=int(lenthim[i])
        unpaded_y_pred.append(y_pred_test[i][:j])
    return unpaded_y_pred

def list_of_lists_to_list(y):
    a=[]
    for i in range(len(y)):
        for j in range(len(y[i])):
            a.append(y[i][j])
    return a

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_dim, num_layers, output_dim):
        super(LSTM, self).__init__()  
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
        
        # Pass the output of the last time step to the fully connected layer
        output = self.fc(output)
        
        
        output = self.sigmoid(torch.squeeze(output,-1))
        
        return output
    


X_train_knn = []
y_train_knn = []
for sentence in embedded_data:
    for word_embedding, tag in sentence:
        X_train_knn.append(word_embedding)
        y_train_knn.append(int(tag))


knn = KNeighborsClassifier(n_neighbors=11, metric='euclidean')
knn.fit(X_train_knn, y_train_knn)
X_test_knn = []


for sentence in embedded_data_test:
    for word_embedding, tag in sentence:
        X_test_knn.append(word_embedding)
       

# Predict the tags for the test data
y_pred_test_knn = knn.predict(X_test_knn)

X_train_lstm = []
y_train_lstm = []
for sentence in embedded_data:
    sen_x =[]
    sen_y =[]
    for word_embedding, tag in sentence:
        sen_x.append(word_embedding)
        sen_y.append(int(tag))
    X_train_lstm.append(sen_x)
    y_train_lstm.append(sen_y)

X_test_lstm=[]
y_test_lstm=[]
for sentence in embedded_data_test:
    sen_x =[]
    sen_y =[]
    for word_embedding, tag in sentence:
        sen_x.append(word_embedding)
        sen_y.append(int(tag))
    X_test_lstm.append(sen_x)
    y_test_lstm.append(sen_y)
X_test_lstm = [torch.tensor(sentence, dtype=torch.float) for sentence in X_test_lstm]
X_test_lstm = pad_sequence(X_test_lstm, batch_first=True, padding_value=0.0)


X_train_lstm = [torch.tensor(sentence, dtype=torch.float) for sentence in X_train_lstm]
X_train_lstm = pad_sequence(X_train_lstm, batch_first=True, padding_value=0.0)
y_train_lstm = [torch.tensor(sentence, dtype=torch.int) for sentence in y_train_lstm]
y_train_lstm = pad_sequence(y_train_lstm, batch_first=True, padding_value=0.0)
data=TensorDataset(torch.tensor(X_train_lstm, dtype=torch.float), torch.tensor(y_train_lstm, dtype=torch.float)) 


def lstm_knn(num_epochs,learning_rate,threshold,data,X_test,y_test):
    """
    function to train lstm model and then use the knn model to change the predictions
    """
 
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    model = LSTM(input_size=25,hidden_dim=32,num_layers=3,output_dim=1)  
    model.to(device)
    batch_size=32
    train_loader = DataLoader(data, batch_size=batch_size, shuffle=True)
    
    optimizer= optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.BCELoss()
    epoch_losses = []
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



    model.eval() 

    # prediction for the test
    with torch.no_grad():
        X_test = np.array(X_test)
        y_pred_test = model(torch.tensor(X_test, dtype=torch.float).to(device))
        y_pred_test = y_pred_test.cpu().numpy() 
        y_pred_test = y_pred_test > threshold

    y_pred_test = y_pred_test.astype(int)
    
    y_pred_test= unpad(y_pred_test,y_test)

    y_pred_test=list_of_lists_to_list(y_pred_test)

    y_final_pred_test=[]
    count = 0
    #ensembling the predictions
    for i in range(len(y_pred_test)):
        
            if y_pred_test_knn[i] == 1 and y_pred_test[i] == 0: 
                y_final_pred_test.append(1)
                count +=1
            else:
                y_final_pred_test.append(y_pred_test[i])
    
    y_test = list_of_lists_to_list(y_test)

    return y_final_pred_test



lr =0.0005
num_epochs = 40
threshold=0.20

y=lstm_knn(num_epochs=40,learning_rate=lr,threshold=threshold,data=data,X_test=X_test_lstm,y_test=y_test_lstm)
