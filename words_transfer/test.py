import matplotlib.pyplot as plt

# 创建一些示例数据
x = [1, 2, 3, 4, 5]
y = [1, 4, 9, 16, 25]

# 使用十六进制颜色字符串
plt.plot(x, y, color='#00ff0080')  # 绿色, 透明度 0.5 (80 是 128 的十六进制表示)
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Example Plot with Hex Color and Transparency')
plt.show()