import time
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, InputLayer, Dropout, Conv1D, Flatten, MaxPool1D
from keras.optimizers import Adam
from processing import build_dataset
from utils import init_session
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import accuracy_score, f1_score
from keras.utils import pad_sequences

init_session()
batch_size = 500
epochs_num = 1
log_dir = "./log/Conv.log"
model_dir = "./file/Conv_model"


def train(train_generator, train_size, input_num, dims_num):
    print("Start Train Job! ")
    start = time.time()
    inputs = InputLayer(input_shape=(input_num, dims_num), batch_size=batch_size)
    layer1 = Conv1D(64, 3, activation="relu")
    layer2 = Conv1D(64, 3, activation="relu")
    layer3 = Conv1D(128, 3, activation="relu")
    layer4 = Conv1D(128, 3, activation="relu")
    layer5 = Dense(128, activation="relu")
    output = Dense(2, activation="softmax", name="Output")
    optimizer = Adam()
    model = Sequential()
    model.add(inputs)
    model.add(layer1)
    model.add(layer2)
    model.add(MaxPool1D(pool_size=2))
    model.add(Dropout(0.5))
    model.add(layer3)
    model.add(layer4)
    model.add(MaxPool1D(pool_size=2))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(layer5)
    model.add(Dropout(0.5))
    model.add(output)
    model.compile(optimizer, loss="categorical_crossentropy", metrics=["accuracy"])
    model.fit(train_generator, steps_per_epoch=train_size // batch_size, epochs=epochs_num)
    model.save_weights(model_dir+ ".h5")
    end = time.time()
    print("Over train job in %f s" % (end - start))


def test(test_generator, test_size, input_num, dims_num, batch_size):
    inputs = InputLayer(input_shape=(input_num, dims_num), batch_size=batch_size)
    layer1 = Conv1D(64, 3, activation="relu")
    layer2 = Conv1D(64, 3, activation="relu")
    layer3 = Conv1D(128, 3, activation="relu")
    layer4 = Conv1D(128, 3, activation="relu")
    layer5 = Dense(128, activation="relu")
    output = Dense(2, activation="softmax", name="Output")
    optimizer = Adam()
    model = Sequential()
    model.add(inputs)
    model.add(layer1)
    model.add(layer2)
    model.add(MaxPool1D(pool_size=2))
    model.add(Dropout(0.5))
    model.add(layer3)
    model.add(layer4)
    model.add(MaxPool1D(pool_size=2))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(layer5)
    model.add(Dropout(0.5))
    model.add(output)
    model.compile(optimizer, loss="categorical_crossentropy", metrics=["accuracy"])
    model.load_weights(model_dir, by_name=False)
    labels_pre = []  # 预测结果
    labels_true = []  # 真实的标签
    batch_num = test_size // batch_size + 1
    steps = 0
    for batch, labels in test_generator:  # test_generator是一批数据，[[攻击向量1，攻击向量2，..],[标签1，标签2，..]]
        if len(labels) == batch_size:  # batch, labels都是数组
            labels_pre.extend(model.predict_on_batch(batch))  # 函数返回模型在一个batch上的预测结果
        else:
            batch = np.concatenate((batch, np.zeros((batch_size - len(labels), input_num, dims_num))))
            labels_pre.extend(model.predict_on_batch(batch)[0:len(labels)])
        labels_true.extend(labels)
        steps += 1
        print("%d/%d batch" % (steps, batch_num))
    labels_pre = np.array(labels_pre).round()

    def to_y(labels):
        y = []
        for i in range(len(labels)):  # 正常是0，恶意是1
            if labels[i][0] == 1:
                y.append(0)  # 正常
            else:
                y.append(1)  # 恶意
        return y

    y_true = to_y(labels_true)
    y_pre = to_y(labels_pre)
    precision = precision_score(y_true, y_pre)
    recall = recall_score(y_true, y_pre)
    accuracy = accuracy_score(y_true, y_pre)
    f1 = f1_score(y_true, y_pre)
    print("Precision score is :", precision)
    print("Recall score is :", recall)
    print("Accuracy score is :", accuracy)
    print("F1 score is :", f1)


if __name__ == "__main__":
    train_generator, test_generator, train_size, test_size, input_num, dims_num = build_dataset(batch_size)
    train(train_generator, train_size, input_num, dims_num)
    test(test_generator, test_size, input_num, dims_num, batch_size)
