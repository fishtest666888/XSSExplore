# import matplotlib
# import matplotlib.pyplot as plt
# import numpy as np
# from matplotlib.ticker import MultipleLocator
#
# matplotlib.rcParams['font.sans-serif'] = ['SimHei']
#
# x = [25, 50, 75, 100, 125]
# y = [0.6685, 0.8611, 0.9167, 0.9213, 0.9259]
# z = [881.94, 1820.79, 3202.85, 4752, 6203.56]
# plt.bar(x, y, width=15, label="绕过率", color="Coral", alpha=0.9)
# plt.legend(loc="upper left")
# plt.xlabel("最大步长")
# plt.ylabel("绕过率")
#
# ax2 = plt.twinx()
# ax2.set_ylabel("时间")
#
# plt.plot(x, z, "r", marker='.', c='r', ms=5, linewidth='3', label="时间")
#
# for a, b in zip(x, y):
#     plt.text(a, b, b, ha='center', va='bottom', fontsize=8)
# # 在右侧显示图例
# plt.legend(loc="upper left")
# x_major_locator=MultipleLocator(25)
# ax=plt.gca()
# ax.xaxis.set_major_locator(x_major_locator)
# plt.savefig("MaxStep.jpg", dpi=300)
#
# plt.show()


import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as mtick
from matplotlib.font_manager import FontProperties

# font = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=14)
b = [0.6685, 0.9167, 0.9199, 0.9213, 0.9259]  # 数据
a = [881.94, 1820.79, 3202.85, 4752, 6203.56]
l = [i for i in range(5)]

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签

fmt = '%.2f%%'
yticks = mtick.FormatStrFormatter(fmt)  # 设置百分比形式的坐标轴
lx = ['25', '50', '75', '100', '125']

fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.plot(l, a, 'Hr-', label=u'训练时间', linewidth=3)

# ax1.yaxis.set_major_formatter(yticks)
# for i, (_x, _y) in enumerate(zip(l, b)):
#     plt.text(_x, _y, b[i], color='black', fontsize=10)  # 将数值显示在图形上
ax1.legend(loc="upper center")
ax1.set_ylim([0, 6300])
ax1.set_ylabel('训练时间(s)')
ax1.set_xlabel("最大步长")
plt.legend(prop={'family': 'SimHei', 'size': 8})  # 设置中文
ax2 = ax1.twinx()  # this is the important function
plt.bar(l, b, width=0.55, alpha=0.45, color='blue', label=u'绕过率')
# ax2.legend(loc=2)
ax2.set_ylim([0, 1])  # 设置y轴取值范围
ax2.set_ylabel('绕过率')

plt.legend(prop={'family': 'SimHei', 'size': 8}, loc="upper center")
plt.xticks(l, lx)
plt.savefig("MaxStep.jpg", dpi=300)
plt.show()
