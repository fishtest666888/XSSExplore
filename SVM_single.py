import csv, random, pickle
import time
import numpy as np
import preprocess
from keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from utils import GeneSeg

batch_size = 50
maxlen = 200
vec_dir = "./file/word2vec1.pickle"
epochs_num = 1
model_dir = "./file/p_SVM_model"


def pre_process(data):
    with open(vec_dir, "rb") as f:
        word2vec = pickle.load(f)
        dictionary = word2vec["dictionary"]
        embeddings = word2vec["embeddings"]
        reverse_dictionary = word2vec["reverse_dictionary"]
    payload = GeneSeg(data)

    def to_index(data):
        d_index = []
        for word in data:
            if word in dictionary.keys():
                d_index.append(dictionary[word])
            else:
                d_index.append(dictionary["UNK"])
        return d_index

    new = to_index(payload)
    new_payload = []
    new_payload.append(new)
    datas_index = pad_sequences(new_payload, value=-1, maxlen=maxlen)
    datas_embed = []
    dims = len(embeddings["UNK"])
    for data in datas_index:
        data_embed = []
        for d in data:
            if d != -1:
                data_embed.extend(embeddings[reverse_dictionary[d]])
            else:
                data_embed.extend([0.0] * dims)
        datas_embed.append(data_embed)
    return datas_embed


def svm_test_result(model, data):
    payload = pre_process(data)
    result = model.predict(payload)
    if result[0] == 1:  # XSS
        return True
    return False


# data = '<script>alert(123)</script>asdhfdebh'
# payload = pre_process(data)

# file = open(model_dir, 'rb')
# xss_svm = pickle.load(file)
# file.close()

# result = xss_svm.predict(payload)
# print("result: ", result)
