import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

# 设置参数
sampling_rate = 100  # 采样率，每秒100个点
duration = 5  # 持续时间5秒
t = np.linspace(0, duration, int(sampling_rate * duration), endpoint=False)

# 生成基础正弦波（代表心跳的基本频率）
base_freq = 1.2  # 基础频率为1.2Hz，即每分钟约72次心跳
signal = np.sin(2 * np.pi * base_freq * t)

# 添加一些细节以更接近真实的脉搏波形
# 添加一个二次谐波以增加波形复杂度
signal += 0.3 * np.sin(2 * np.pi * 2 * base_freq * t)

# 添加一些随机波动来模拟个体差异和噪声
np.random.seed(0)  # 固定随机种子以便结果可复现
noise = np.random.normal(0, 0.1, signal.shape)
signal += noise

# 对信号进行平滑处理以模拟实际测量中的滤波效果
window_size = 10
smoothed_signal = np.convolve(signal, np.ones(window_size)/window_size, mode='same')

# 绘制生成的平滑后的信号
plt.figure(figsize=(12, 6))
plt.plot(t, smoothed_signal, label='模拟平滑脉搏信号', color='red')
plt.xlabel('时间 (秒)')
plt.ylabel('脉搏幅度')
plt.title('模拟生成的平滑脉搏信号')
plt.legend()
plt.show()

# 频率特征
fft_signal = np.fft.fft(smoothed_signal)
freq = np.fft.fftfreq(len(smoothed_signal), d=t[1] - t[0])
plt.figure(figsize=(12, 6))
plt.plot(freq[:len(freq)//2], np.abs(fft_signal[:len(fft_signal)//2]), label='频率特征')  # 只显示正频率部分
plt.xlabel('频率 (Hz)')
plt.ylabel('幅度')
plt.title('脉搏信号的频率特征')
plt.legend()
plt.show()

def envelope(signal):
    peaks, _ = find_peaks(signal)
    valleys, _ = find_peaks(-signal)
    env = np.zeros_like(signal)
    env[peaks] = signal[peaks]
    env[valleys] = signal[valleys]
    return env

# 波形特征
env_signal = envelope(smoothed_signal)
plt.figure(figsize=(12, 6))
plt.plot(t, env_signal, label='波形特征', color='green')
plt.xlabel('时间 (秒)')
plt.ylabel('脉搏幅度')
plt.title('脉搏信号的波形特征')
plt.legend()
plt.show()

# 周期特征
peak_indices, _ = find_peaks(smoothed_signal)
periods = np.diff(t[peak_indices])
average_period = np.mean(periods)
plt.figure(figsize=(12, 6))
plt.hist(periods, bins=10, label='周期特征')
plt.axvline(average_period, color='r', linestyle='dashed', linewidth=2, label=f'平均周期: {average_period:.2f} 秒')
plt.xlabel('周期 (秒)')
plt.ylabel('次数')
plt.title('脉搏信号的周期特征')
plt.legend()
plt.show()