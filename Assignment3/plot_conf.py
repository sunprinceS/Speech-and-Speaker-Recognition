#!/usr/bin/env python
# -- coding: utf-8 --

import itertools
from sklearn.metrics import confusion_matrix
from states import *
import matplotlib.pyplot as plt

def plot_confusion_matrix(cm, classes=None,
                          title='Confusion matrix',
                          cmap=plt.cm.jet):
    """
    This function prints and plots the confusion matrix.
    """
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    # thresh = cm.max() / 2.
    # for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        # plt.text(j, i, '{:.2f}'.format(cm[i, j]), horizontalalignment="center",
                 # color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def main():
    pred = np.load('pred_y.npz')['pred']
    ans = np.load('data/test_y.npz')['y']
    pred = np.array([stateList[st][:-2] for st in pred])
    ans = np.array([stateList[st][:-2] for st in ans])
    conf_mat = confusion_matrix(ans,pred)

    plt.figure()
    plot_confusion_matrix(conf_mat, classes=phonemeList)
    plt.show()
if __name__ == "__main__":
    main()
