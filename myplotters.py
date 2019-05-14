import numpy as np
import matplotlib.pyplot as plt

# Bar Plot
def myBar(dfi, pTitle, pSubTitle, xColN, xColLab, yCol1N, yCol2N, yColLab, 
          legendL, barWidth, stackedBool, y2ColN, y2ColLab, y2ColLeg, data_labels):
    fig, ax = plt.subplots(figsize=(10,6))
    xVar = np.arange(len(dfi[xColN])) # for stacked vs. side-by-side determination below
    plt.bar(xVar, dfi[yCol1N], color='#63646a', edgecolor='white', width=barWidth) #63646a dark-gray
    if yCol2N != None:
        if stackedBool == False:
            xVar = xVar + barWidth # allows side-by-side bar
        plt.bar(xVar, dfi[yCol2N], color='#d9d9d9', edgecolor='white', width=barWidth) #d9d9d9 light-gray
    plt.xlabel(xColLab, fontsize=16)
    plt.ylabel(yColLab, fontsize=16)
    plt.suptitle(pTitle, y=.96, horizontalalignment='center', fontsize=20, fontweight='bold')
    plt.title(pSubTitle, horizontalalignment='center', fontsize=15, style='italic')
    plt.xticks(range(dfi[xColN].count()), dfi[xColN]) # show all x-value labels on x-axis
    plt.legend(legendL, loc=1)
    
    # Add Line to 2nd Axis
    if y2ColN is not None:
        ax2 = plt.twinx()
        ax2.plot(dfi[xColN], dfi[y2ColN], color='#9e2e62', label=y2ColLab) #9e2e62 dark purple
        ax2.set_ylabel(y2ColLab, fontsize=16, rotation=270, labelpad=15)
        ax2.legend([y2ColLeg], loc=2) 

    # Add data labels
    rects = ax.patches
    for rect, label in zip(rects, data_labels):
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width() / 2, height + 5, label,
                ha='center', va='bottom')

    plt.xlim([-0.5, dfi[xColN].size - 0.5]) # remove white space on left/right of plot
    fig.tight_layout(rect=[0, 0, 1, 0.95]) # keeps from pushing text offscreen
    return plt.show()

# Training Plot History
def plot_history(history):
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    x = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(x, acc, 'b', label='Training acc')
    plt.plot(x, val_acc, 'r', label='Validation acc')
    plt.title('Training and Validation Acccuracy')
    plt.xlabel('Epoch')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(x, loss, 'b', label='Training loss')
    plt.plot(x, val_loss, 'r', label='Validation loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.legend()
    return plt.show()