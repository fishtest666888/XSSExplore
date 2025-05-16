import csv
import xlwt
import codecs
import re
from urllib.parse import unquote


def deduplicate(data):
    new_data = []
    for i in range(len(data)):
        if data[i] not in new_data:
            new_data.append(data[i])
    return new_data


def pre_process(payload):
    pay_load = re.sub(r'<br/>', '', payload)
    pay_load = pay_load.replace('+', ' ')
    pay_load = unquote(pay_load)
    pay_load = pay_load.lower()
    return pay_load


def data_write_excel(datas):
    f = xlwt.Workbook()
    sheet1 = f.add_sheet(u'sheet1', cell_overwrite_ok=True)
    i = 0
    for data in datas:
        sheet1.write(i, 0, data)
        i = i + 1
    f.save("data\\xss_preprocess.xlsx")


def data_write_csv(datas):
    file_csv = codecs.open("./data/chen.csv", 'w+', 'utf-8')  # 追加
    writer = csv.writer(file_csv, delimiter=' ', quotechar=' ', skipinitialspace=False, quoting=csv.QUOTE_MINIMAL)
    for data in datas:
         writer.writerow(data)
    print("over")


def data_process():
    data = []
    with open("./data/ss.csv", "r", encoding="utf-8", errors="ignore") as f:
        reader = csv.DictReader(f, fieldnames=["payload"])  # 读csv文件，把每行中的信息映射到一个字典,字典的键由fieldnames给出
        for row in reader:
            payload = row["payload"]
            data.append(payload)
    # new_data = deduplicate(data)
    return data


def new_process():
    data = []
    with open("./data/new.csv", "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, fieldnames=["payload"])  # 读csv文件，把每行中的信息映射到一个字典,字典的键由fieldnames给出
        for row in reader:
            payload = row["payload"]
            word = pre_process(payload)
            data.append(word)
    return data

# datas = data_process()
# print(datas)
# data_write_excel(datas)
