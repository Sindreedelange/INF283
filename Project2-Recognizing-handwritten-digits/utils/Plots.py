import matplotlib.pyplot as plt
import numpy as np
import itertools


class Plots:

    def __init__(self):
        pass

    def plot_confusion_matrix(self, conf_mat):
        plt.figure()
        plt.imshow(conf_mat, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion matrix')
        plt.colorbar()
        class_names = [0,1,2,3,4,5,6,7,8,9]
        tick_marks = np.arange(len(class_names))
        plt.xticks(tick_marks, class_names, rotation=45)
        plt.yticks(tick_marks, class_names)

        thresh = conf_mat.max() / 2.
        for i, j in itertools.product(range(conf_mat.shape[0]), range(conf_mat.shape[1])):
            plt.text(j, i, format(conf_mat[i, j], 'd'),
                    horizontalalignment="center",
                    color="white" if conf_mat[i, j] > thresh else "black")

        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.tight_layout()
        plt.show()