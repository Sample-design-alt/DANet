import matplotlib.pyplot as plt
import torch

x = ['WA+SA+SWM', 'WA+SA', 'WA+SWM', 'WA', 'SA+SWM', 'SA', 'SWM', 'WiWo transformer']

# accuracy
y1 = [0.8150, 0.8324, 0.8786, 0.8150, 0.8843, 0.8323, 0.8439, 0.8381]  # 32
y2 = [0.8439, 0.8497, 0.8265, 0.8497, 0.8554, 0.8843, 0.8323, 0.8612]  # 48
y3 = [0.8439, 0.8670, 0.8728, 0.8554, 0.8612, 0.8265, 0.8554, 0.8912]  # 64
y4 = [0.8554, 0.8092, 0.8497, 0.8728, 0.8612, 0.8208, 0.8381, 0.8728]  # 96

plt.plot(x, y1, marker='*', markersize=8, linestyle='dotted', label='window size=32')
plt.plot(x, y2, marker='v', markersize=8, linestyle='dashed', label='window size=48')
plt.plot(x, y3, marker='x', markersize=8, linestyle='dashdot', label='window size=64')
plt.plot(x, y4, marker='p', markersize=8, linestyle='solid', label='window size=96')
plt.xticks(rotation=45)
plt.xlabel('Model')
plt.ylabel('Accuracy')
plt.title('Impact of Window Size')
plt.legend()
plt.show()
