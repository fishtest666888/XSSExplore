from utils import GeneSeg
from datetime import datetime
import csv, pickle, random, json
import numpy as np
# from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import tensorflow as tf
from keras.utils import pad_sequences

vec_dir = "./file/word2vec1.pickle"
pre_datas_train = "./file/pre_datas_train.csv"
pre_datas_test = "./file/pre_datas_test.csv"
process_datas_dir = "./file/process_datas.pickle"


def pre_process():
    with open(vec_dir, "rb") as f:
        word2vec = pickle.load(f)
        dictionary = word2vec["dictionary"]
        reverse_dictionary = word2vec["reverse_dictionary"]
        embeddings = word2vec["embeddings"]
    xssed_data = []
    normal_data = []
    with open("./data/xssed.csv", "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, fieldnames=["payload"])
        for row in reader:
            payload = row["payload"]
            word = GeneSeg(payload)
            xssed_data.append(word)
    with open("./data/normal.csv", "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        reader = csv.DictReader(f, fieldnames=["payload"])
        for row in reader:
            payload = row["payload"]
            word = GeneSeg(payload)
            normal_data.append(word)
    xssed_num = len(xssed_data)
    normal_num = len(normal_data)
    xssed_labels = [1] * xssed_num
    normal_labels = [0] * normal_num
    datas = xssed_data + normal_data
    labels = xssed_labels + normal_labels
    labels = to_categorical(labels)

    def to_index(data):
        d_index = []
        for word in data:
            if word in dictionary.keys():
                d_index.append(dictionary[word])
            else:
                d_index.append(dictionary["UNK"])
        return d_index

    datas_index = [to_index(data) for data in datas]
    datas_index = pad_sequences(datas_index, value=-1)
    random.seed(datetime.now())
    rand = random.sample(range(len(datas_index)), len(datas_index))
    datas = [datas_index[index] for index in rand]
    labels = [labels[index] for index in rand]
    train_datas, test_datas, train_labels, test_labels = train_test_split(datas, labels, test_size=0.3)
    train_size = len(train_labels)
    test_size = len(test_labels)
    input_num = len(train_datas[0])
    dims_num = embeddings["UNK"].shape[0]
    word2vec["train_size"] = train_size
    word2vec["test_size"] = test_size
    word2vec["input_num"] = input_num  # 532
    word2vec["dims_num"] = dims_num  # 128
    with open(vec_dir, "wb") as f:
        pickle.dump(word2vec, f)
    print("Saved word2vec to:", vec_dir)
    print("Write trian datas to:", pre_datas_train)
    with open(pre_datas_train, "w") as f:
        for i in range(train_size):
            data_line = str(train_datas[i].tolist()) + "|" + str(train_labels[i].tolist()) + "\n"
            f.write(data_line)
    print("Write test datas to:", pre_datas_test)
    with open(pre_datas_test, "w") as f:
        for i in range(test_size):
            data_line = str(test_datas[i].tolist()) + "|" + str(test_labels[i].tolist()) + "\n"
            f.write(data_line)
    print("Write datas over!")


def data_generator(data_dir):
    dataset = tf.data.TextLineDataset(data_dir)  # ✅ 提供文件路径作为参数
    dataset = dataset.shuffle(buffer_size=10000)  # 可选：打乱数据
    iterator = iter(dataset.repeat())  # 循环读取
    return iterator

def batch_generator(datas_dir, datas_size, batch_size, embeddings, reverse_dictionary, train=True):
    generator = data_generator(datas_dir)
    while True:
        batch_data = []
        batch_label = []
        for _ in range(batch_size):
            try:
                line = next(generator)
                data_str, label_str = line.numpy().decode("utf-8").split("|")
                data = json.loads(data_str)
                label = json.loads(label_str)
                data_embed = []
                for d in data:
                    if d != -1:
                        data_embed.append(embeddings[reverse_dictionary[d]])
                    else:
                        data_embed.append([0.0] * len(embeddings["UNK"]))
                batch_data.append(data_embed)
                batch_label.append(label)
            except StopIteration:
                if not train:
                    break
                generator = data_generator(datas_dir)  # 重新开始
                line = next(generator)
                data_str, label_str = line.numpy().decode("utf-8").split("|")
                data = json.loads(data_str)
                label = json.loads(label_str)
                data_embed = []
                for d in data:
                    if d != -1:
                        data_embed.append(embeddings[reverse_dictionary[d]])
                    else:
                        data_embed.append([0.0] * len(embeddings["UNK"]))
                batch_data.append(data_embed)
                batch_label.append(label)

        yield (np.array(batch_data), np.array(batch_label))
        if not train and len(batch_label) < batch_size:
            break


def build_dataset(batch_size):
    with open(vec_dir, "rb") as f:
        word2vec = pickle.load(f)
    embeddings = word2vec["embeddings"]
    reverse_dictionary = word2vec["reverse_dictionary"]
    train_size = word2vec["train_size"]
    test_size = word2vec["test_size"]
    dims_num = word2vec["dims_num"]
    input_num = word2vec["input_num"]
    train_generator = batch_generator(pre_datas_train, train_size, batch_size, embeddings, reverse_dictionary)
    test_generator = batch_generator(pre_datas_test, test_size, batch_size, embeddings, reverse_dictionary, train=False)
    return train_generator, test_generator, train_size, test_size, input_num, dims_num


if __name__ == "__main__":
    pre_process()
