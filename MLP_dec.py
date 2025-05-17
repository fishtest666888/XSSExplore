import pickle
import numpy as np
from utils import GeneSeg
import preprocess
from keras.models import Sequential
from keras.layers import Dense, InputLayer, Dropout, Flatten, LSTM
from keras.optimizers import Adam
from keras.preprocessing.sequence import pad_sequences
detection_dir = "./file/detection.csv"
model_dir = "./file/MLP_model.h5"
vec_dir = "./file/word2vec1.pickle"
log_dir = "./log/MLP.log"


# 修改后完整的 process_data 函数
def process_data(data):
    payload = GeneSeg(data)

    def to_index(words):
        return [dictionary[word] if word in dictionary else dictionary["UNK"] for word in words]

    data_index = to_index(payload)
    padded = pad_sequences([data_index], maxlen=532, value=-1)[0]

    unk_vector = embeddings["UNK"]
    data_embed = [embeddings[reverse_dictionary[d]] if d != -1 else unk_vector for d in padded]

    return np.array([data_embed])  # shape: (1, maxlen, embed_dim)


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


def mlp_model(input_num, dims_num, batch_size):
    inputs = InputLayer(input_shape=(input_num, dims_num), batch_size=batch_size)
    layer1 = Dense(100, activation="relu")
    layer2 = Dense(20, activation="relu")
    flatten = Flatten()
    layer3 = Dense(2, activation="softmax", name="Output")
    optimizer = Adam()
    model = Sequential()
    model.add(inputs)
    model.add(layer1)
    model.add(Dropout(0.5))
    model.add(layer2)
    model.add(Dropout(0.5))
    model.add(flatten)
    model.add(layer3)
    model.compile(optimizer, loss="categorical_crossentropy", metrics=["accuracy"])
    model.load_weights(model_dir, by_name=False)
    return model


num = 0
data = preprocess.data_process()
model = mlp_model(532, 128, 1)
for item in data:
    result = mlp_test_result(item, model)
    if result:
        num += 1
print("检测率：", num/len(data))
