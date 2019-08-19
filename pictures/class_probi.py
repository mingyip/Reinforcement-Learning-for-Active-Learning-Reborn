import numpy as np
import matplotlib.pyplot as plt

np.random.seed(19680801)
n_bins = 10
x = np.random.randn(1000, 3)

# param
n_groups = 10
bar_width = 0.65
opacity = 0.8
index = np.arange(n_groups)

# Dataset
step0   = (0.09, 0.13, 0.13, 0.14, 0.09, 0.10, 0.09, 0.08, 0.11, 0.10)
step10  = (0.10, 0.07, 0.45, 0.11, 0.10, 0.12, 0.08, 0.05, 0.09, 0.02)
step50  = (0.05, 0.05, 0.63, 0.10, 0.02, 0.045, 0.09, 0.054, 0.043, 0.038)
step100 = (0.04, 0.05, 0.85, 0.12, 0.02, 0.02, 0.08, 0.02, 0.01, 0.03)

fig, axes = plt.subplots(nrows=2, ncols=2)
ax0, ax1, ax2, ax3 = axes.flatten()

rects0 = ax0.bar(index, step0, bar_width,
alpha=opacity,
color='g',
label='Step 0')
ax0.set_xticks(np.arange(n_groups))
ax0.set_xlabel('Class')
ax0.set_ylabel('Predicted Probability')
ax0.set_ylim(top=1)
ax0.legend()

rects1 = ax1.bar(index, step10, bar_width,
alpha=opacity,
color='g',
label='Step 10')
ax1.set_xticks(np.arange(n_groups))
ax1.set_xlabel('Class')
ax1.set_ylim(top=1)
ax1.legend()

rects2 = ax2.bar(index, step50, bar_width,
alpha=opacity,
color='g',
label='Step 50')
ax2.set_xticks(np.arange(n_groups))
ax2.set_xlabel('Class')
ax2.set_ylabel('Predicted Probability')
ax2.set_ylim(top=1)
ax2.legend()

rects3 = ax3.bar(index, step100, bar_width,
alpha=opacity,
color='g',
label='Step 100')
ax3.set_xticks(np.arange(n_groups))
ax3.set_xlabel('Class')
# ax3.set_ylabel('Predicted Probability')
ax3.set_ylim(top=1)
ax3.legend()

fig.suptitle('Predicted Probability in different states', fontsize=12)
fig.tight_layout()
fig.subplots_adjust(top=0.91)
plt.show()

