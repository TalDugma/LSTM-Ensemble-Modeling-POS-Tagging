import os
import numpy as np
from Preprocessing import embedded_sentences, preprocess, file_to_sentences_dev, get_unique_words, file_to_sentences 
import shutil
# model = gensim.downloader.load('glove-twitter-200')
# model.save_word2vec_format('glove-twitter-25.txt', binary=False)
# model = gensim.models.KeyedVectors.load_word2vec_format('glove-twitter-25.txt', binary=False)
#copy test.untaggedntag test so it will be in the same shape as dev and train
#create copy of test.untagged file
shutil.copy("data/test.untagged","data/test copy.untagged")

with open("data/test copy.untagged","r") as testfile:
    lines = testfile.readlines()
    line_counter=-1
    for line in lines:
        line_counter+=1
        if line!="\n" and line!="\t" and line!=" " and line!="" and line!="\t\n":
            lines[line_counter] = line[:-1] + "\tO\n"
        else:
            lines[line_counter] = line
with open("data/test copy.tagged","w") as testfile:    
    #delete testfile.txt`content`
    testfile.truncate(0)
    #write the new content
    testfile.writelines(lines)
embedded_train = embedded_sentences(preprocess(file_to_sentences("data/train.tagged")),get_unique_words(preprocess(file_to_sentences("data/train.tagged"))))
# Convert the nested lists to a structured format (list of tuples)
structured_data_train = [[(word, tag) for word, tag in sentence] for sentence in embedded_train]

                                    

# Convert the structured data into a NumPy array
nested_array_test= np.array(structured_data_train, dtype=object)

# Save the structured NumPy array using np.save
np.save('nested_array_train_25.npy', nested_array_test, allow_pickle=True)



embedded_test = embedded_sentences(preprocess(file_to_sentences_dev("data/test copy.tagged")),get_unique_words(preprocess(file_to_sentences_dev("data/test copy.tagged"))))
# Convert the nested lists to a structured format (list of tuples)
structured_data_test = [[(word, tag) for word, tag in sentence] for sentence in embedded_test]

# Convert the structured data into a NumPy array
nested_array_test= np.array(structured_data_test, dtype=object)

# Save the structured NumPy array using np.save
np.save('nested_array_test_25.npy', nested_array_test, allow_pickle=True)

print("done embedding test")
    

