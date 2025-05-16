# import numpy as np
# import matplotlib.mlab as mlab
# import matplotlib
# import matplotlib.pyplot as plt
# import numpy as np
#
# def drawHistogram():
#     matplotlib.rc("font", family='MicroSoft YaHei')
#     list1 = np.array([0.08, 0.70, 0.82, 0.56])   # 柱状图第一组数据
#     list2 = np.array([0.85, 0.85, 0.88, 0.81])   # 柱状图第二组数据
#     length = len(list1)
#     x = np.arange(length)   # 横坐标范围
#     listDate = ["Webseclab", "XSS-labs", "Firing Range", "OWASP Benchmark"]
#
#     plt.figure()
#     total_width, n = 0.7, 2   # 柱状图总宽度，有几组数据
#     width = total_width / n   # 单个柱状图的宽度
#     x1 = x - width / 2   # 第一组数据柱状图横坐标起始位置
#     x2 = x1 + width   # 第二组数据柱状图横坐标起始位置
#
#     # plt.title("一周每天吃悠哈软糖颗数柱状图")   # 柱状图标题
#     plt.xlabel("靶机")   # 横坐标label 此处可以不添加
#     plt.ylabel("约登指数")   # 纵坐标label
#     plt.bar(x1, list1, width=width, color='xkcd:turquoise', label="VulExplore")
#     plt.bar(x2, list2, width=width, color='xkcd:lavender', label="Wapiti")
#     plt.xticks(x, listDate)   # 用星期几替换横坐标x的值
#     plt.legend()   # 给出图例
#     plt.savefig("yuedeng.jpg", dpi=300)
#     plt.show()
#
# if __name__ == '__main__':
#     drawHistogram()

import matplotlib.pyplot as plt
import matplotlib

x = ["Webseclab", "XSS-labs", "Firing Range", "OWASP Benchmark"]
y1 = [0.08, 0.70, 0.82, 0.56]
y2 = [0.85, 0.85, 0.88, 0.81]

# plt.title('扩散速度')  # 折线图标题
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
plt.xlabel('靶标')  # x轴标题
plt.ylabel('约登指数')  # y轴标题
plt.plot(x, y1, marker='o', markersize=3, linewidth=2)  # 绘制折线图，添加数据点，设置点的大小
plt.plot(x, y2, marker='o', markersize=3, linewidth=2)

for a, b in zip(x, y1):
    plt.text(a, b, b, ha='center', va='bottom', fontsize=10)  # 设置数据标签位置及大小
for a, b in zip(x, y2):
    plt.text(a, b, b, ha='center', va='bottom', fontsize=10)

plt.legend(['Wapiti', 'XSSExplore'])  # 设置折线名称
plt.savefig("yuedeng.jpg", dpi=300)
plt.show()  # 显示折线图
