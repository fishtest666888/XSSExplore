import matplotlib.pyplot as plt
import matplotlib

x = ["MLP", "LSTM", "SafeDog", "XSSChop"]
y1 = [0.981, 0.918, 0.979, 0.990]
y2 = [0.953, 0.862, 0.942, 0.969]

# plt.title('扩散速度')  # 折线图标题
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
plt.xlabel('检测模型')  # x轴标题
plt.ylabel('绕过率ER')  # y轴标题
plt.plot(x, y1, marker='o', markersize=3, linewidth=2)  # 绘制折线图，添加数据点，设置点的大小
plt.plot(x, y2, marker='o', markersize=3, linewidth=2)

for a, b in zip(x, y1):
    plt.text(a, b, b, ha='center', va='bottom', fontsize=10)  # 设置数据标签位置及大小
for a, b in zip(x, y2):
    plt.text(a, b, b, ha='center', va='bottom', fontsize=10)

plt.legend(['XSSExplore', 'Wang'])  # 设置折线名称
plt.savefig("adversarial.jpg", dpi=300)
plt.show()  # 显示折线图
