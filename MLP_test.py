import pickle
import numpy as np
from utils import GeneSeg
# from keras.preprocessing.sequence import pad_sequences
from keras.utils import pad_sequences


detection_dir = "./file/detection.csv"
model_dir = "./file/MLP_model.h5"
vec_dir = "./file/word2vec1.pickle"
log_dir = "./log/MLP.log"


def process_data(data):
    new_data = []
    with open(vec_dir, "rb") as f:
        word2vec = pickle.load(f)
        dictionary = word2vec["dictionary"]
        reverse_dictionary = word2vec["reverse_dictionary"]
        embeddings = word2vec["embeddings"]
        dims_num = word2vec["dims_num"]
        input_num = word2vec["input_num"]

    payload = GeneSeg(data)

    def to_index(data):
        d_index = []
        for word in data:
            if word in dictionary.keys():  # 如果这个词存在于前面挑选出来的vocabulary_size-1个词中
                d_index.append(dictionary[word])
            else:
                d_index.append(dictionary["UNK"])
        return d_index

    data_index = to_index(payload)
    new_data.append(data_index)
    datas = pad_sequences(new_data, maxlen=532, value=-1)
    da = np.array(datas[0])
    data_embed = []
    for d in da:
        if d != -1:
            data_embed.append(embeddings[reverse_dictionary[d]])  # 每个词用128位的向量表示
        else:
            data_embed.append([0.0] * len(embeddings["UNK"]))
    d = []
    d.append(data_embed)
    return np.array(d)


def mlp_detection(model, data):
    labels_pre = []  # 预测结果
    labels_pre.extend(model.predict_on_batch(data))  # 函数返回模型在一个batch上的预测结果
    labels_pre = np.array(labels_pre).round()

    if labels_pre[0][0] == 1:  # 正常样本
        return False
    else:  # 恶意样本
        return True


def mlp_test_result(data, model):
    datas = process_data(data)
    result = mlp_detection(model, datas)
    return result



