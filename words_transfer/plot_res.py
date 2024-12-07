import matplotlib.pyplot as plt

def plot_sensitivity_scores(scores, final_score):
    plt.figure(figsize=(10, 6))  # 创建一个新的图形窗口，设置大小
    
    # 找到最终分数在列表中的索引
    final_score_index = next((i for i, score in enumerate(scores) if score == final_score), None)
    
    # 如果找到了最终分数的索引，绘制两条线：红色和绿色
    if final_score_index is not None:
        # 绘制红色线（最终分数之前）
        plt.plot(range(len(scores[:final_score_index+1])), scores[:final_score_index+1], marker="o", linestyle="-", color="red", label='Before Final Score')
        # 绘制绿色线（包括最终分数及之后）
        plt.plot(range(final_score_index, len(scores)), scores[final_score_index:], marker="o", linestyle="-", color="green", label='After Final Score')
    else:
        # 如果没有找到最终分数，只绘制一条蓝线
        plt.plot(range(len(scores)), scores, marker="o", linestyle="-", color="blue", label='Sensitive Scores')
    
    plt.title("Sensitive Score Changes Over Iterations")
    plt.xlabel("Iteration")
    plt.ylabel("Sensitive Score")
    plt.grid(True)
    plt.legend()  # 显示图例

    # 保存图表到硬盘E
    plt.savefig('E:/sensitivity_scores.png')

    # 显示图表
    plt.show()

# 示例分数列表
sensitivity_scores = [96.17931842803955, 94.47479844093323, 69.25705075263977, 48.78421127796173, 26.12912952899933, 20.3071728348732, 15.836913883686066, 12.135674059391022, 9.766027331352234, 6.981152296066284, 3.9491720497608185, 2.922307141125202, 1.6801755875349045, 1.4109758660197258, 1.2327684089541435, 0.7489688228815794, 0.58809625916183, 0.5198541097342968, 0.49888836219906807, 0.472057331353426, 0.46960869804024696, 0.3866532351821661, 0.3483325242996216, 0.32808338291943073, 0.26649029459804296, 0.2502472838386893, 0.24251011200249195, 0.22383728064596653, 0.19837617874145508, 0.1586818601936102, 0.15742944087833166, 0.1418496249243617, 0.11991389328613877, 0.11024604318663478, 0.10247275931760669, 0.06979791796766222, 0.0581124855671078, 0.047465128591284156]
# 示例最终分数
final_score = 0.5198541097342968

# 调用函数绘制并保存图像
plot_sensitivity_scores(sensitivity_scores, final_score)