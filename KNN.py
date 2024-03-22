#NOTE : In order to run we need the  "nested_array" files, therefore before the run we need to run prepreocessing and then calc_embed.py
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score, recall_score
embedded_data = np.load("nested_array_train_25.npy",allow_pickle=True) #changed from embedded_train to nested_array_train.npy

#implemet KNN using sklearn:
knn = KNeighborsClassifier(n_neighbors=7, metric='euclidean')
# Fit the KNN model
X_train = []
y_train = []
for sentence in embedded_data:
    for word_embedding, tag in sentence:
        X_train.append(word_embedding)
        y_train.append(int(tag))

knn.fit(X_train, y_train)
print("Accuracy on training data:", knn.score(X_train, y_train))
            
embedded_data_dev = np.load("nested_array_dev_25.npy",allow_pickle=True)
         
X_dev = []
y_dev = []
for sentence in embedded_data_dev:
    for word_embedding, tag in sentence:
        X_dev.append(word_embedding)
        y_dev.append(int(tag))
# Predict the tags for the training data
y_pred_dev = knn.predict(X_dev)
# Calculate the F1 score on the training data
f1_dev = f1_score(y_dev, y_pred_dev)
print("F1 score on dev data:", f1_dev)
#calculate percision and recall
precision_dev = precision_score(y_dev, y_pred_dev)
recall_dev = recall_score(y_dev, y_pred_dev)
print("Recall on dev data:", recall_dev)
print("Precision on dev data:", precision_dev)
