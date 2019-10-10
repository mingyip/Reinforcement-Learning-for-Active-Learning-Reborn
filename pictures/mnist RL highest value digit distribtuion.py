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


dist0 = (67, 74, 88, 109, 98, 95, 71, 102, 144, 152)
dist1 = (65, 74, 92, 114, 100, 112, 80, 76, 141, 146)
dist2 = (61, 65, 99, 115, 110, 111, 65, 88, 132, 154)
dist3 = (68, 67, 91, 106, 98, 107, 74, 98, 146, 145)
dist4 = (60, 67, 92, 123, 95, 110, 63, 96, 141, 153)
dist5 = (63, 68, 90, 105, 103, 117, 74, 92, 141, 147)
dist6 = (65, 73, 90, 124, 100, 105, 70, 88, 141, 144)
dist7 = (62, 65, 92, 121, 99, 108, 63, 85, 149, 156)
dist8 = (67, 67, 86, 107, 107, 109, 78, 87, 141, 151)
dist9 = (62, 70, 87, 117, 101, 107, 65, 93, 147, 151)
dist10 = (67, 67, 93, 116, 103, 111, 67, 86, 147, 143)
dist11 = (69, 62, 88, 123, 97, 114, 65, 84, 149, 149)


value0 = 0.955
value1 = 0.941
value2 = 0.946
value3 = 0.957
value4 = 0.958
value5 = 0.964
value6 = 0.961
value7 = 0.957
value8 = 0.948
value9 = 0.960
value10 = 0.967
value11 = 0.941



#dist0 = (96, 105, 79, 70, 77, 66, 74, 80, 77, 76)
#dist1 = (76, 101, 87, 86, 62, 76, 82, 79, 69, 82)
#dist2 = (94, 88, 71, 85, 73, 73, 79, 91, 78, 68)
#dist3 = (86, 95, 69, 71, 71, 74, 81, 91, 75, 87)
#dist4 = (75, 81, 81, 86, 74, 91, 79, 73, 84, 76)
#dist5 = (69, 95, 71, 98, 66, 76, 77, 85, 79, 84)
#dist6 = (77, 87, 70, 80, 75, 65, 81, 85, 105, 75)
#dist7 = (84, 87, 87, 70, 75, 70, 100, 76, 77, 74)
#dist8 = (75, 75, 95, 87, 97, 61, 72, 85, 80, 73)
#dist9 = (80, 103, 85, 70, 80, 67, 61, 83, 94, 77)
#dist10 = (83, 86, 87, 72, 84, 65, 77, 78, 72, 96)
#dist11 = (88, 88, 77, 80, 78, 82, 71, 74, 82, 80)


#value0 = 0.899
#value1 = 0.864
#value2 = 0.898
#value3 = 0.911
#value4 = 0.860
#value5 = 0.924
#value6 = 0.835
#value7 = 0.902
#value8 = 0.937
#value9 = 0.842
#value10 = 0.929
#value11 = 0.907



#dist0  = (56, 78, 110, 97, 72, 98, 75, 129, 153, 132)
#dist1  = (79, 64, 98, 108, 54, 115, 88, 148, 145, 101)
#dist2  = (76, 117, 123, 105, 74, 75, 64, 114, 122, 130)
#dist3  = (84, 86, 64, 134, 60, 125, 65, 63, 194, 125)
#dist4  = (80, 74, 89, 65, 100, 67, 61, 99, 89, 276)
#dist5  = (67, 87, 101, 108, 90, 103, 63, 105, 117, 159)
#dist6  = (46, 106, 58, 97, 267, 63, 51, 62, 116, 134)
#dist7  = (57, 92, 64, 106, 84, 231, 60, 95, 106, 105)
#dist8  = (87, 85, 63, 143, 71, 122, 62, 67, 174, 126)
#dist9  = (53, 80, 52, 113, 57, 105, 59, 286, 92, 103)
#dist10 = (74, 138, 130, 122, 67, 127, 71, 75, 122, 74)
#dist11 = (64, 89, 105, 114, 105, 95, 70, 134, 120, 104)


#value0  = 0.959
#value1  = 0.961
#value2  = 0.946
#value3  = 0.957
#value4  = 0.949
#value5  = 0.958
#value6  = 0.930
#value7  = 0.941
#value8  = 0.956
#value9  = 0.936
#value10 = 0.955
#value11 = 0.955



# TOP
# dist0  = (9, 9, 718, 56, 10, 11, 152, 9, 21, 5)
# dist1  = (75, 92, 115, 74, 141, 49, 233, 63, 77, 81)
# dist2  = (8, 5, 9, 6, 349, 20, 152, 71, 14, 366)
# dist3  = (457, 5, 38, 153, 9, 109, 193, 10, 13, 13)
# dist4  = (659, 0, 19, 6, 13, 8, 283, 7, 4, 1)
# dist5  = (358, 9, 394, 108, 3, 66, 38, 5, 14, 5)
# dist6  = (14, 16, 586, 300, 4, 39, 7, 7, 22, 5)
# dist7  = (6, 38, 68, 8, 428, 5, 26, 362, 8, 51)
# dist8  = (44, 1, 73, 213, 14, 98, 495, 4, 51, 7)
# dist9  = (232, 2, 11, 15, 18, 11, 692, 5, 8, 6)
# dist10 = (17, 25, 362, 25, 16, 21, 7, 188, 302, 37)
# dist11 = (20, 4, 9, 18, 554, 28, 21, 48, 17, 281)


# value0  = 0.602
# value1  = 0.896
# value2  = 0.488
# value3  = 0.459
# value4  = 0.431
# value5  = 0.466
# value6  = 0.441
# value7  = 0.453
# value8  = 0.549
# value9  = 0.540
# value10 = 0.480
# value11 = 0.384



font = {
        'color':  'darkred',
        'weight': 'normal',
        'size': 12,
        }
fig, axes = plt.subplots(nrows=3, ncols=4)
ax0, ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9, ax10, ax11 = axes.flatten()

rects0 = ax0.bar(index, dist0, bar_width,
alpha=opacity,
color='g',
label='seed 0')
ax0.set_xticks(np.arange(n_groups))
ax0.set_xlabel("acc: " + str(value0))
ax0.set_ylabel('Predicted Probability')
ax0.set_ylim(top=1000)
ax0.legend()

rects1 = ax1.bar(index, dist1, bar_width,
alpha=opacity,
color='g',
label='seed 1')
ax1.set_xticks(np.arange(n_groups))
ax1.set_xlabel("acc: " + str(value1))
ax1.set_ylim(top=1000)
ax1.legend()

rects2 = ax2.bar(index, dist2, bar_width,
alpha=opacity,
color='g',
label='seed 2')
ax2.set_xticks(np.arange(n_groups))
ax2.set_xlabel("acc: " + str(value2))
# ax2.set_ylabel('Predicted Probability')
ax2.set_ylim(top=1000)
ax2.legend()

rects3 = ax3.bar(index, dist3, bar_width,
alpha=opacity,
color='g',
label='seed 3')
ax3.set_xticks(np.arange(n_groups))
ax3.set_xlabel("acc: " + str(value3))
# ax3.set_ylabel('Predicted Probability')
ax3.set_ylim(top=1000)
ax3.legend()

rects4 = ax4.bar(index, dist4, bar_width,
alpha=opacity,
color='g',
label='seed 4')
ax4.set_xticks(np.arange(n_groups))
ax4.set_xlabel("acc: " + str(value4))
ax4.set_ylabel('Predicted Probability')
ax4.set_ylim(top=1000)
ax4.legend()

rects5 = ax5.bar(index, dist5, bar_width,
alpha=opacity,
color='g',
label='seed 5')
ax5.set_xticks(np.arange(n_groups))
ax5.set_xlabel("acc: " + str(value5))
# ax5.set_ylabel('Predicted Probability')
ax5.set_ylim(top=1000)
ax5.legend()

rects6 = ax6.bar(index, dist6, bar_width,
alpha=opacity,
color='g',
label='seed 6')
ax6.set_xticks(np.arange(n_groups))
ax6.set_xlabel("acc: " + str(value6))
# ax6.set_ylabel('Predicted Probability')
ax6.set_ylim(top=1000)
ax6.legend()

rects7 = ax7.bar(index, dist7, bar_width,
alpha=opacity,
color='g',
label='seed 7')
ax7.set_xticks(np.arange(n_groups))
ax7.set_xlabel("acc: " + str(value7))
# ax7.set_ylabel('Predicted Probability')
ax7.set_ylim(top=1000)
ax7.legend()

rects8 = ax8.bar(index, dist8, bar_width,
alpha=opacity,
color='g',
label='seed 8')
ax8.set_xticks(np.arange(n_groups))
ax8.set_xlabel("acc: " + str(value8))
ax8.set_ylabel('Predicted Probability')
ax8.set_ylim(top=1000)
ax8.legend()

rects9 = ax9.bar(index, dist9, bar_width,
alpha=opacity,
color='g',
label='seed 9')
ax9.set_xticks(np.arange(n_groups))
ax9.set_xlabel("acc: " + str(value9))
# ax9.set_ylabel('Predicted Probability')
ax9.set_ylim(top=1000)
ax9.legend()

rects10 = ax10.bar(index, dist10, bar_width,
alpha=opacity,
color='g',
label='seed 10')
ax10.set_xticks(np.arange(n_groups))
ax10.set_xlabel("acc: " + str(value10))
# ax10.set_ylabel('Predicted Probability')
ax10.set_ylim(top=1000)
ax10.legend()

rects11 = ax11.bar(index, dist11, bar_width,
alpha=opacity,
color='g',
label='seed 11')
ax11.set_xticks(np.arange(n_groups))
ax11.set_xlabel("acc: " + str(value11))
# ax11.set_ylabel('Predicted Probability')
ax11.set_ylim(top=1000)
ax11.legend()


fig.suptitle('12 MNIST digit distribtion of BVSB agent set', fontsize=14)
fig.tight_layout()
fig.subplots_adjust(top=0.91)
plt.show()

