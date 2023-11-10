import numpy as np
import matplotlib.pyplot as plt

def accuracy(model, x, y):
    pred = model.predict(x)
    correct_pred = np.where(pred == y, 1, 0).sum()
    return correct_pred / len(pred[:])

def visualize_training(avg_loss, accuracies):
    fig, axs = plt.subplots(2)

    # Make the y-label, ticks and tick labels match the line color.
    axs[0].set_ylabel('Loss')
    axs[0].plot(avg_loss, 'tab:blue')

    axs[1].set_ylabel('Accuracy')
    axs[1].set_xlabel('Epochs')
    axs[1].plot(accuracies, 'tab:orange')

    #plt.tight_layout()
    plt.show()