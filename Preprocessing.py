import numpy as np
import gensim.downloader

# Download and load the GloVe model
model = gensim.downloader.load('glove-twitter-25')

# Save the model in Word2Vec format
model.save_word2vec_format('glove-twitter-25.txt', binary=False)

model = gensim.models.KeyedVectors.load_word2vec_format('glove-twitter-25.txt', binary=False)


def subwords(word):
    """
    function gets a word and returns the sub-words if the word is more than one word (like StarWars) for example. 
    """
    sub_words = []
    for i in range(len(word)):
        if word[i].isupper() and i>0:
            sub_words.append(word[:i])
            sub_words.append(word[i:])
            return sub_words
    if len(sub_words) == 0:
        sub_words.append(word)
    return sub_words
    
def file_to_sentences(tagged_file_path):
    """
    fuction gets a tagged file path and returns a list of the sentences split 
    """
    with open(tagged_file_path,"r") as file:
        sentences_list = []
        sentence = []
        for line in file:
            if line !="\t\n" and line != "\n":
                sentence.append(line)
            else:
                sentences_list.append(sentence)
                sentence = []
        # sentences_list.append(sentence)
    return sentences_list

def file_to_sentences_dev(tagged_file_path):
    """
    fuction gets a tagged file path and returns a list of the sentences split 
    """
    with open(tagged_file_path,"r") as file:
        sentences_list = []
        sentence = []
        count = 0
        for line in file:
            if line !="\n" and line != "\t\n":
                count+=1
                sentence.append(((line)))
            else:
                sentences_list.append(sentence)
                sentence = []
        # sentences_list.append(sentence)
    return sentences_list    


def sentence_to_pairs(sentence):
    """
    function gets a sentence (list) and returns it devided to mini-lists that contain a pair of word-tag
    """
    for pair_index in range(len(sentence)):
        sentence[pair_index] = sentence[pair_index].split("\t")
        sentence[pair_index][1] = sentence[pair_index][1].replace("\n","")
    return sentence

def username(word):
    """
    function gets a word and returns True if the word is a username, else False
    """
    if word[0] == "@" and len(word)>1:
        return True
    return False
def hashtag(word):
    if word[0] == "#":
        return True
    return False 
def link(word):
    if word[0:4] == "http":
        return True
    return False

def glove(model, target_word):
    try: 
        return model[target_word]
    except:
        try:
            # try lower case (for words that are not in the model but their lower case is in the model)
            return model[target_word.lower()]
        except:
            if len(target_word)>0: 
                if len(subwords(target_word))>1:
                    sub_words = subwords(target_word)
                    return (glove(model,sub_words[0])+glove(model,sub_words[1]))
                if target_word[0].isupper():
                    return model["<unknown>"]+np.array([0.5]*25)
            return model["<unknown>"]

def preprocess(sentences):
    """
    function gets the sentences list and returns a list of the sentences after preprocessing (without embedding)
    """
    new_sentences_list = []
    for i in range(len(sentences)):
        new_sentence = []
        original_sentence= sentence_to_pairs(sentences[i])
        for j in range(len(original_sentence)):
            word,tag = original_sentence[j][0],original_sentence[j][1]
            if tag!="O":
                tag = '1'
            else:
                tag = '0'
            if username(word):
                new_sentence.append(["@@",tag])
            elif hashtag(word):
                new_sentence.append([word[1:],tag])
            elif link(word):
                new_sentence.append(["http",tag])
            # elif word.isdigit():
            #      new_sentence.append(["nine",tag])
            else:   
                new_sentence.append([word,tag])

        new_sentences_list.append(new_sentence)   
    return new_sentences_list 

# embedding only once for each word
def get_unique_words(sentences):
    """
    function gets the sentences list and returns a list of the unique words in the sentences
    """
    unique_words_embedding = []
    for i in range(len(sentences)):
        for j in range(len(sentences[i])):
            if sentences[i][j][0] not in unique_words_embedding:
                unique_words_embedding.append([sentences[i][j][0],glove(model,sentences[i][j][0])])
    return unique_words_embedding

def get_idx_of_word(word,unique_words_embedding):
    """
    function gets a word and a list of unique words and returns the index of the word in the list
    """
    for i in range(len(unique_words_embedding)):
        if unique_words_embedding[i][0] == word:
            return i
    return None


def embedded_sentences(sentences,unique_words_embedding):
    """
    function gets the sentences list and the words' embedding list and returns the sentences with the embedding
    """  
    new_sentences_list = []
    for sentence_index in range(len(sentences)):
        for word_index in range(len(sentences[sentence_index])):
            word= sentences[sentence_index][word_index][0]
            sentences[sentence_index][word_index][0] =unique_words_embedding[get_idx_of_word(word,unique_words_embedding)][1]
    return sentences

