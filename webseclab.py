# import matplotlib.pyplot as plt
# import matplotlib
# import numpy as np
#
# matplotlib.rcParams['font.sans-serif'] = ['SimHei']
# num_list = [0.750, 0.538, 1.000, 1.000,  0.714, 0.636, 1.000, ]  # 柱状图第一组数据
# num_list1 = [0.316, 0.737, 0.316, 0.700,  0.476, 0.737, 0.850]  # 柱状图第二组数据
# name = ["XSpear", "XSSer", "XSSmap", "Suggester", "wapiti", "w3af", "XSSExplore"]
# x = list(range(len(name)))
# width = 0.3
# index = np.arange(len(name))
# plt.bar(index, num_list, width, color='steelblue', tick_label=name, label='precision')
# plt.bar(index + width, num_list1, width, color='red', hatch='\\', label='recall')
# plt.legend(['precision', 'recall'], labelspacing=1)
# plt.xlabel("扫描器")
# # plt.xticks(rotation=15)
# plt.legend()
# plt.savefig("Webseclab.jpg", dpi=300)
# plt.show()

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

def drawHistogram():
    matplotlib.rc("font", family='MicroSoft YaHei')
    list1 = np.array([0.750, 0.538, 1.000, 1.000,  0.714, 0.636, 1.000])   # 柱状图第一组数据
    list2 = np.array([0.316, 0.737, 0.316, 0.700,  0.476, 0.737, 0.800])   # 柱状图第二组数据
    length = len(list1)
    x = np.arange(length)   # 横坐标范围
    listDate = ["XSpear", "XSSer", "XSSmap", "Suggester", "wapiti", "w3af", "XSSExplore"]

    plt.figure()
    total_width, n = 0.6, 2   # 柱状图总宽度，有几组数据
    width = total_width / n   # 单个柱状图的宽度
    x1 = x - width / 2   # 第一组数据柱状图横坐标起始位置
    x2 = x1 + width   # 第二组数据柱状图横坐标起始位置

    # plt.title("一周每天吃悠哈软糖颗数柱状图")   # 柱状图标题
    plt.xlabel("扫描器")   # 横坐标label 此处可以不添加
    # plt.ylabel("唯一结构URL数")   # 纵坐标label
    plt.bar(x1, list1, width, color='steelblue', tick_label=listDate, label='precision')
    plt.bar(x2, list2, width, color='red', label='recall')
    plt.xticks(x, listDate)   # 用星期几替换横坐标x的值
    plt.legend()   # 给出图例
    plt.savefig("Webseclab.jpg", dpi=300)
    plt.show()

if __name__ == '__main__':
    drawHistogram()