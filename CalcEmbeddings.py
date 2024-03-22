#here we calculate the embedding of the data for the train and test, so we don't need to calculate it every time we run the model
#we will use the same embedding for all the models
import Preprocessing as pr
import numpy as np
import gensim.downloader


train_file_path = "data/train.tagged"
dev_file_path = "data/dev.tagged"  
embedded_train_data = pr.embedded_sentences(pr.preprocess(pr.file_to_sentences(train_file_path)),pr.get_unique_words(pr.preprocess(pr.file_to_sentences(train_file_path))))
# Convert the data into a structured NumPy array
# Convert the nested lists to a structured format (list of tuples)
structured_data = [[(word, tag) for word, tag in sentence] for sentence in embedded_train_data]

# Convert the structured data into a NumPy array
nested_array_train= np.array(structured_data, dtype=object)

# Save the structured NumPy array using np.save
np.save('nested_array_train_25.npy', nested_array_train, allow_pickle=True)

embedded_dev_data = pr.embedded_sentences(pr.preprocess(pr.file_to_sentences_dev(dev_file_path)),pr.get_unique_words(pr.preprocess(pr.file_to_sentences_dev(dev_file_path))))
# Convert the nested lists to a structured format (list of tuples)
structured_data_dev = [[(word, tag) for word, tag in sentence] for sentence in embedded_dev_data]

# Convert the structured data into a NumPy array
nested_array_dev= np.array(structured_data_dev, dtype=object)

# Save the structured NumPy array using np.save
np.save('nested_array_dev_25.npy', nested_array_dev, allow_pickle=True)


print("done")

                
