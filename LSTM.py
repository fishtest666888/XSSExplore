import time
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, InputLayer, Dropout, LSTM
from keras.callbacks import TensorBoard
from keras.optimizers import Adam
from processing import build_dataset
from utils import init_session
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import accuracy_score, f1_score


init_session()
batch_size = 350
epochs_num = 1
process_datas_dir = "./file/process_datas.pickle"
log_dir = "./log/LSTM.log"
model_dir = "./file/LSTM_model.h5"


def train(train_generator, train_size, input_num, dims_num):
    print("Start Train Job! ")
    start = time.time()
    inputs = InputLayer(input_shape=(input_num, dims_num), batch_size=batch_size)
    layer1 = LSTM(128)
    output = Dense(2, activation="softmax", name="Output")
    optimizer = Adam()
    model = Sequential()
    model.add(inputs)
    model.add(layer1)
    model.add(Dropout(0.5))
    model.add(output)
    call = TensorBoard(log_dir=log_dir, write_grads=True, histogram_freq=0)
    model.compile(optimizer, loss="categorical_crossentropy", metrics=["accuracy"])
    model.fit(train_generator, steps_per_epoch=train_size // batch_size, epochs=epochs_num, callbacks=[call])
    model.save_weights(model_dir)
    end = time.time()
    print("Over train job in %f s" % (end - start))


def test(model_dir, test_generator, test_size, input_num, dims_num, batch_size):
    inputs = InputLayer(input_shape=(input_num, dims_num), batch_size=batch_size)
    layer1 = LSTM(128)
    output = Dense(2, activation="softmax", name="Output")
    optimizer = Adam()
    model = Sequential()
    model.add(inputs)
    model.add(layer1)
    model.add(Dropout(0.5))
    model.add(output)
    call = TensorBoard(log_dir=log_dir, write_grads=True, histogram_freq=0)
    model.compile(optimizer, loss="categorical_crossentropy", metrics=["accuracy"])
    model.load_weights(model_dir, by_name=False)
    labels_pre = []  # 预测结果
    labels_true = []  # 真实的标签
    batch_num = test_size // batch_size + 1
    steps = 0
    for batch, labels in test_generator:
        if len(labels) == batch_size:
            labels_pre.extend(model.predict_on_batch(batch))
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
    print("precision:", precision)
    print("recall:", recall)
    print("Accuracy score is :", accuracy)
    print("F1 score is :", f1)
    # return precision, recall


if __name__ == "__main__":
    train_generator, test_generator, train_size, test_size, input_num, dims_num = build_dataset(batch_size)
    train(train_generator, train_size, input_num, dims_num)
    test(model_dir, test_generator, test_size, input_num, dims_num, batch_size)

# def test_lstm(epochs_num):
#     train_generator, test_generator, train_size, test_size, input_num, dims_num = build_dataset(batch_size)
#     json_string = train(train_generator, train_size, input_num, dims_num, epochs_num)
#     precision, recall = test(model_dir, test_generator, test_size, input_num, dims_num, batch_size)
#     return precision, recall
