
from __future__ import print_function
import sys

import itertools
import numpy as np
import time
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.lines as mlines

import math

import os
import util.dataprocessing
from sklearn import metrics
import load_class_2
dp = util.dataprocessing.DataPlotter()



base_dir_path = "{}/".format(os.path.dirname(os.path.abspath(__file__))) #"/home/jasper/oneshot-gestures/

BASE_DIR        =   "{}/".format(os.path.dirname(os.path.abspath(__file__)))
MODEL_VERS      =   "model-19x1-temp"
MODEL_EXCLUDING =   "model-19"
ONESHOT_CLASS   =   15

OUTPUT_DIRECTORY=   "{}output/{}/class-{}/".format(BASE_DIR,MODEL_VERS,ONESHOT_CLASS)
PARAM_DIRECTORY =   "{}convnet_params/{}/class-{}/".format(BASE_DIR,MODEL_VERS,ONESHOT_CLASS)
EXCLUDING_PARAM_PATH   \
                =   "{}convnet_params/{}/excluding-{}".format(BASE_DIR,MODEL_EXCLUDING,ONESHOT_CLASS)

# x_labels = (1,2,3,4,5,25,100,200)
# y_test = np.load("{}output/y_tests/class{}.npy".format(base_dir_path,ONESHOT_CLASS))
# y_predictions = np.empty((len(x_labels),len(y_test)))
#
# i = 0
# for num_samples in x_labels:
#     y_predictions[i] = np.load("{}layers1-samples{}/y_predictions.npy".format(OUTPUT_DIRECTORY,num_samples))
#     # dp.plotConfusionMatrix(y_test,y_predictions[i],"{}layers1-samples{}/conf-matr".format(OUTPUT_DIRECTORY,num_samples))
#     i+=1
#
#
# dp.plotAccF1(y_test,y_predictions,x_labels,19,"Learning gesture 15, retrain level 1")

def cm2inch(*tupl):
    inch=0.393700787
    if isinstance(tupl[0], tuple):
        return tuple(i * inch for i in tupl[0])
    else:
        return tuple(i * inch for i in tupl)

def sigmoid(x):
    a = []
    for item in x:
        a.append(1/(1+math.exp(-item)))
    return a


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, "\centering{:2.0f}".format(cm[i, j]*100.0),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
    else:
        print('Confusion matrix, without normalization')
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, cm[i, j],
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    # plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def plotConfusionMatrix(y_test, y_pred,title="Confusion Matrix", savePath=None):
    class_names = ["0"]
    for i in xrange(1,20):
        class_names.append("{}".format(i))

    cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
    np.set_printoptions(precision=2)


    # Plot non-normalized confusion matrix
    plt.figure(figsize=(8,6))
    plot_confusion_matrix(cnf_matrix, classes=class_names,
                          title=title)
    if savePath != None:
        # plt.savefig("{}.pgf".format(savePath))
        plt.savefig("{}.pdf".format(savePath))
    plt.show()

    # Plot normalized confusion matrix
    plt.figure(figsize=(8,6))
    plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                          title="{}".format(title))
    if savePath != None:
        # plt.savefig("{}-norm.pgf".format(savePath))
        plt.savefig("{}-norm.pdf".format(savePath))
    plt.show()


def plotPrecRec(y_test, y_predictions, x_labels, baseline_prec, baseline_rec, title="F1 score", oneshot_class=19):

    # plt.grid(True)
    prec_scores = np.zeros(len(y_predictions))
    recall_scores = np.zeros(len(y_predictions))

    i = 0
    for y_pred in y_predictions:
        prec_scores[i] = metrics.precision_score(y_test, y_pred, labels=[oneshot_class], average=None)
        recall_scores[i] = metrics.recall_score(y_test, y_pred, labels=[oneshot_class], average=None)
        i += 1

    print(prec_scores)
    print(recall_scores)

    plt.xlabel(r"Aantal samples")

    plt.plot(x_labels,prec_scores, 'b')
    plt.plot(x_labels,recall_scores, 'r')

    plt.plot(x_labels,prec_scores, 'b.')
    plt.plot(x_labels,recall_scores, 'r.')

    plt.axhline(y=baseline_prec, xmin=0, xmax=1, linewidth=1, color='blue', linestyle=':', hold=None)
    plt.axhline(y=baseline_rec, xmin=0, xmax=1, linewidth=1, color='red', linestyle=':', hold=None)

    blue_line = mlines.Line2D([], [], color='blue', marker='.', label='precision')
    red_line = mlines.Line2D([], [], color='red', marker='.', label="recall")
    blue_dash = mlines.Line2D([], [], color='blue', linestyle=':', linewidth=1, label='precision baseline')
    red_dash = mlines.Line2D([], [], color='red', linestyle=':', linewidth=1, label='recall baseline')


    plt.legend(handles=(blue_line,blue_dash,red_line,red_dash))


    ax = plt.subplot(111)
    ax.set_xscale('log')
    ax.set_xticks(x_labels)
    ax.get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter())

    plt.title(title,size=12)
    plt.tight_layout()

def plotNaive15():
    y_test = np.load("/home/jasper/oneshot-gestures/output/y_tests/class19.npy")
    num_samples_array = [1, 2, 3, 4, 5, 10, 25, 50, 100, 200]
    y_predictions = np.empty((len(num_samples_array), 2000))
    i = 0
    for num_samples in num_samples_array:
        y_predictions[i] = np.load(
            "{}output/naive_model/class-15/layers1-samples{}/y_predictions.npy".format(BASE_DIR, num_samples))
        i += 1
    plt.figure(figsize=cm2inch(14, 9))
    plotPrecRec(y_test, y_predictions, num_samples_array, 0.97, 0.95,title="", oneshot_class=15)
    plt.show()

def plotNaive15Layers2():
    y_test = np.load("/home/jasper/oneshot-gestures/output/y_tests/class19.npy")
    num_samples_array = [1, 2, 3, 4, 5, 10, 25, 50, 100, 200]
    y_predictions = np.empty((len(num_samples_array), 2000))
    i = 0
    for num_samples in num_samples_array:
        y_predictions[i] = np.load(
            "{}output/naive_model/class-15/layers2-samples{}/y_predictions.npy".format(BASE_DIR, num_samples))
        i += 1
    plt.figure(figsize=cm2inch(14, 9))
    plotPrecRec(y_test, y_predictions, num_samples_array, 0.97, 0.95,title="", oneshot_class=15)
    plt.show()

def plotNaiveLayers():
    y_test = np.load("/home/jasper/oneshot-gestures/output/y_tests/class19.npy")
    num_samples_array = [1, 2, 3, 4, 5, 10, 25, 50, 100, 200]
    y_predictions_1 = np.empty((len(num_samples_array), 2000))
    y_predictions_2 = np.empty((len(num_samples_array), 2000))

    i = 0
    for num_samples in num_samples_array:
        y_predictions_1[i] = np.load(
            "{}output/naive_model/class-15/layers1-samples{}/y_predictions.npy".format(BASE_DIR, num_samples))
        y_predictions_2[i] = np.load(
            "{}output/naive_model/class-15/layers2-samples{}/y_predictions.npy".format(BASE_DIR, num_samples))

        i += 1
    plt.figure(figsize=cm2inch(16, 11))


    # plt.grid(True)
    prec_scores_1 = np.zeros(len(y_predictions_1))
    recall_scores_1 = np.zeros(len(y_predictions_1))

    prec_scores_2 = np.zeros(len(y_predictions_2))
    recall_scores_2 = np.zeros(len(y_predictions_2))

    i = 0
    for y_pred in y_predictions_1:
        prec_scores_1[i] = metrics.precision_score(y_test, y_pred, labels=[15], average=None)
        recall_scores_1[i] = metrics.recall_score(y_test, y_pred, labels=[15], average=None)
        i += 1
    i=0
    for y_pred in y_predictions_2:
        prec_scores_2[i] = metrics.precision_score(y_test, y_pred, labels=[15], average=None)
        recall_scores_2[i] = metrics.recall_score(y_test, y_pred, labels=[15], average=None)
        i += 1


    plt.xlabel(r"Aantal samples")

    x_labels=num_samples_array
    # plt.ylabels(np.arange(0,1,0.1))
    plt.yticks(np.arange(0,1.1,0.1))

    plt.plot(x_labels,prec_scores_1, 'b')
    plt.plot(x_labels,recall_scores_1, 'r')

    plt.plot(x_labels,prec_scores_1, 'b.')
    plt.plot(x_labels,recall_scores_1, 'r.')

    plt.plot(x_labels, prec_scores_2, 'c')
    plt.plot(x_labels, recall_scores_2, 'm')

    plt.plot(x_labels, prec_scores_2, 'cx')
    plt.plot(x_labels, recall_scores_2, 'mx')

    plt.axhline(y=0.97, xmin=0, xmax=1, linewidth=0.5, color='blue', linestyle=':', hold=None)
    plt.axhline(y=0.95, xmin=0, xmax=1, linewidth=0.5, color='red', linestyle=':', hold=None)

    blue_line = mlines.Line2D([], [], color='blue', marker='.', label='precision 1 laag')
    red_line = mlines.Line2D([], [], color='red', marker='.', label="recall 1 laag")
    blue_line_2 = mlines.Line2D([], [], color='cyan', marker='.', linestyle='--', label='precision 2 lagen')
    red_line_2 = mlines.Line2D([], [], color='magenta', marker='.', linestyle='--', label="recall 2 lagen")
    blue_dash = mlines.Line2D([], [], color='blue', linestyle=':', linewidth=0.5, label='precision baseline')
    red_dash = mlines.Line2D([], [], color='red', linestyle=':', linewidth=0.5, label='recall baseline')


    plt.legend(handles=(blue_line,blue_line_2,red_line_2,red_line,blue_dash,red_dash))


    ax = plt.subplot(111)
    ax.set_xscale('log')
    ax.set_xticks(x_labels)
    ax.get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter())

    plt.title("",size=12)
    plt.tight_layout()
    plt.show()


def plot15():
    y_test = np.load("/home/jasper/oneshot-gestures/output/y_tests/class15.npy")
    num_samples_array = [1, 2, 3, 4, 5, 10, 25, 50, 100, 200]
    y_predictions = np.empty((len(num_samples_array), 2000))
    i = 0
    for num_samples in num_samples_array:
        y_predictions[i] = np.load(
            "{}output/model-19x1-temp/class-15/layers1-samples{}/y_predictions.npy".format(BASE_DIR, num_samples))
        i += 1
    plt.figure(figsize=cm2inch(14, 9))
    plotPrecRec(y_test, y_predictions, num_samples_array, 0.97, 0.95, "Oneshot gebaar 15")
    plt.show()

def plot14():
    y_test = np.load("/home/jasper/oneshot-gestures/output/y_tests/class14.npy")
    num_samples_array = [1, 2, 3, 4, 5, 10, 25, 50, 100, 200]
    y_predictions = np.empty((len(num_samples_array), 2000))
    i = 0
    for num_samples in num_samples_array:
        y_predictions[i] = np.load(
            "{}output/model-19x1-temp/class-14/layers1-samples{}/y_predictions.npy".format(BASE_DIR, num_samples))
        i += 1
    plt.figure(figsize=cm2inch(14, 9))
    plotPrecRec(y_test, y_predictions, num_samples_array, 0.74, 0.65, "Oneshot gebaar 14")
    plt.show()

def plot14augm():
    y_test = np.load("/home/jasper/oneshot-gestures/output/y_tests/class14.npy")
    num_samples_array = [1, 2, 3, 4, 5,10,25,50,100,200]
    y_predictions = np.empty((len(num_samples_array), 2000))
    i = 0
    for num_samples in num_samples_array:
        y_predictions[i] = np.load(
            "{}output/model-19x1-redo/class-14/layers1-samples{}augm/y_predictions.npy".format(BASE_DIR, num_samples))
        i += 1
    plt.figure(figsize=cm2inch(14, 9))
    plotPrecRec(y_test, y_predictions, num_samples_array, 0.74, 0.65)
    plt.show()

def plot15augm():
    y_test = np.load("/home/jasper/oneshot-gestures/output/y_tests/class15.npy")
    num_samples_array = [1, 2, 3, 4, 5,10,25,50,100,200]
    y_predictions = np.empty((len(num_samples_array), 2000))
    i = 0
    for num_samples in num_samples_array:
        y_predictions[i] = np.load(
            "{}output/model-19x1-redo/class-15/layers1-samples{}augm/y_predictions.npy".format(BASE_DIR, num_samples))
        i += 1
    plt.figure(figsize=cm2inch(14, 9))
    plotPrecRec(y_test, y_predictions, num_samples_array, 0.97, 0.95, title="")
    plt.show()

def plotVgl15Augm():
    y_test = np.load("/home/jasper/oneshot-gestures/output/y_tests/class15.npy")
    num_samples_array = [1, 2, 3, 4, 5, 10, 25, 50, 100, 200]
    y_predictions_1 = np.empty((len(num_samples_array), 2000))
    y_predictions_2 = np.empty((len(num_samples_array), 2000))

    i = 0
    for num_samples in num_samples_array:
        y_predictions_1[i] = np.load(
            "{}output/model-19x1-temp/class-15/layers1-samples{}/y_predictions.npy".format(BASE_DIR, num_samples))
        y_predictions_2[i] = np.load(
            "{}output/model-19x1-redo/class-15/layers1-samples{}augm/y_predictions.npy".format(BASE_DIR, num_samples))

        i += 1
    plt.figure(figsize=cm2inch(16, 11))


    # plt.grid(True)
    prec_scores_1 = np.zeros(len(y_predictions_1))
    recall_scores_1 = np.zeros(len(y_predictions_1))

    prec_scores_2 = np.zeros(len(y_predictions_2))
    recall_scores_2 = np.zeros(len(y_predictions_2))

    i = 0
    for y_pred in y_predictions_1:
        prec_scores_1[i] = metrics.precision_score(y_test, y_pred, labels=[19], average=None)
        recall_scores_1[i] = metrics.recall_score(y_test, y_pred, labels=[19], average=None)
        i += 1
    i=0
    for y_pred in y_predictions_2:
        prec_scores_2[i] = metrics.precision_score(y_test, y_pred, labels=[19], average=None)
        recall_scores_2[i] = metrics.recall_score(y_test, y_pred, labels=[19], average=None)
        i += 1


    plt.xlabel(r"Aantal samples")

    x_labels=num_samples_array
    # plt.ylabels(np.arange(0,1,0.1))
    plt.yticks(np.arange(0,1.1,0.1))

    plt.plot(x_labels,prec_scores_1, 'b')
    plt.plot(x_labels,recall_scores_1, 'r')

    plt.plot(x_labels,prec_scores_1, 'b.')
    plt.plot(x_labels,recall_scores_1, 'r.')

    plt.plot(x_labels, prec_scores_2, 'c')
    plt.plot(x_labels, recall_scores_2, 'm')

    plt.plot(x_labels, prec_scores_2, 'cx')
    plt.plot(x_labels, recall_scores_2, 'mx')

    plt.axhline(y=0.97, xmin=0, xmax=1, linewidth=0.5, color='blue', linestyle=':', hold=None)
    plt.axhline(y=0.95, xmin=0, xmax=1, linewidth=0.5, color='red', linestyle=':', hold=None)

    blue_line = mlines.Line2D([], [], color='blue', marker='.', label='precision')
    red_line = mlines.Line2D([], [], color='red', marker='.', label="recall")
    blue_line_2 = mlines.Line2D([], [], color='cyan', marker='.', linestyle='--', label='precision met augmentatie')
    red_line_2 = mlines.Line2D([], [], color='magenta', marker='.', linestyle='--', label="recall met augmentatie")
    blue_dash = mlines.Line2D([], [], color='blue', linestyle=':', linewidth=0.5, label='precision baseline')
    red_dash = mlines.Line2D([], [], color='red', linestyle=':', linewidth=0.5, label='recall baseline')


    plt.legend(handles=(blue_line,red_line,blue_line_2,red_line_2,blue_dash,red_dash))

    ax = plt.subplot(111)
    ax.set_xscale('log')
    ax.set_xticks(x_labels)
    ax.get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter())

    plt.show()

def plot18x2():
    y_test = np.load("/home/jasper/oneshot-gestures/output/y_tests/class13-18.npy")
    num_samples_array = [1,10,100]
    y_predictions_1 = np.empty((len(num_samples_array), 2000))
    y_predictions_2 = np.empty((len(num_samples_array), 2000))

    i = 0
    for num_samples in num_samples_array:
        y_predictions_1[i] = np.load(
            "{}output/model-18x2/class-13-18/layers1-samples{}/y_predictions.npy".format(BASE_DIR, num_samples))
        i += 1


    # plt.grid(True)
    prec_scores_1 = np.zeros(len(y_predictions_1))
    recall_scores_1 = np.zeros(len(y_predictions_1))

    prec_scores_2 = np.zeros(len(y_predictions_1))
    recall_scores_2 = np.zeros(len(y_predictions_1))

    i = 0
    for y_pred in y_predictions_1:
        prec_scores_1[i] = metrics.precision_score(y_test, y_pred, labels=[18], average=None)
        recall_scores_1[i] = metrics.recall_score(y_test, y_pred, labels=[18], average=None)
        prec_scores_2[i] = metrics.precision_score(y_test, y_pred, labels=[19], average=None)
        recall_scores_2[i] = metrics.recall_score(y_test, y_pred, labels=[19], average=None)
        i += 1

    OS_1 = mlines.Line2D([], [], color='red', marker='.', label='gebaar 13')
    OS_10 = mlines.Line2D([], [], color='blue', marker='.', label="gebaar 18")
    base = mlines.Line2D([], [], color='green', linewidth=0.5, linestyle=':', label='baseline 13')
    base2= mlines.Line2D([], [], color='orange', linewidth=0.5, linestyle=':', label='baseline 18')

    fig, (ax0, ax1) = plt.subplots(nrows=2)

    ax1.legend(handles=(OS_1, OS_10, base, base2))
    ax0.set_ylabel("Precision")
    ax0.set_color_cycle(['r', 'r', 'b', 'b', 'g', 'g'])
    ax0.set_xscale('log')
    ax0.set_xticks([1,10,100])
    ax0.set_xticklabels([1,10,100])
    ax0.get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter())

    ax1.set_ylabel("Recall")
    ax1.set_color_cycle(['r', 'r', 'b', 'b', 'g', 'g'])
    ax1.set_xscale('log')
    ax1.set_xticks([1,10,100])
    ax1.set_xticklabels([1,10,100])
    ax1.get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
    ax1.set_xlabel("Klassenummer")


    ax0.plot([1,10,100], prec_scores_1, '-')
    ax0.plot([1,10,100], prec_scores_1, '.')
    ax0.plot([1,10,100], prec_scores_2, '-')
    ax0.plot([1,10,100], prec_scores_2, '.')

    ax1.plot([1,10,100], recall_scores_1, '-')
    ax1.plot([1,10,100], recall_scores_1, '.')
    ax1.plot([1,10,100], recall_scores_2, '-')
    ax1.plot([1,10,100], recall_scores_2, '.')

    ax0.axhline(y=0.8039, xmin=0, xmax=1, linewidth=1, color='green', linestyle=':')
    ax0.axhline(y=0.9069, xmin=0, xmax=1, linewidth=1, color='yellow', linestyle=':')
    ax1.axhline(y=0.7961, xmin=0, xmax=1, linewidth=1, color='green', linestyle=':')
    ax1.axhline(y=0.8387, xmin=0, xmax=1, linewidth=1, color='yellow', linestyle=':')

    # plt.tight_layout()
    # plt.show()
    plt.tight_layout()
    plt.show()

def plotDataAugm():
    # -----------------------------#
    # Vergelijking data-augmentatie#
    # -----------------------------#
    augm = mlines.Line2D([], [], color='red', marker='.', label='Met augmentatie')
    noaugm = mlines.Line2D([], [], color='blue', marker='.', label="Zonder augmentatie")
    base = mlines.Line2D([], [], color='green', linestyle=':', marker='.', label='baseline')
    recall_line = mlines.Line2D([], [], color='red', linestyle='', marker='.', label='recall')
    prec_line = mlines.Line2D([], [], color='blue', linestyle='', marker='.', label='prec')

    fig, (ax0, ax1) = plt.subplots(nrows=2)
    fig2, (ax3, ax4) = plt.subplots(nrows=2)

    ax0.legend(handles=(augm, noaugm))
    ax0.set_ylabel("Precision")
    ax0.set_color_cycle(['r', 'r', 'b', 'b', 'g', 'g'])
    ax0.set_xticks(xrange(20))
    ax0.set_xticklabels(xrange(20))

    ax1.set_ylabel("Recall")
    ax1.set_color_cycle(['r', 'r', 'b', 'b', 'g', 'g'])
    ax1.set_xticks(xrange(20))
    ax1.set_xticklabels(xrange(20))
    ax1.set_xlabel("Klassenummer")

    ax3.legend(handles=(prec_line, recall_line))
    ax3.set_ylabel("Precision")
    ax3.set_color_cycle(['r', 'b', 'g', 'y'])
    ax3.set_xticks(xrange(20))
    ax3.set_xticklabels(xrange(20))

    ax4.set_ylabel("Recall")
    ax4.set_color_cycle(['r', 'b', 'g', 'y'])
    ax4.set_xticks(xrange(20))
    ax4.set_xticklabels(xrange(20))
    ax4.set_xlabel("Klassenummer")

    y_pred = np.load("/home/jasper/oneshot-gestures/output/conf_matrix_data/all.npy")
    y_test = np.load("/home/jasper/oneshot-gestures/output/y_tests/class19.npy")
    all_prec = metrics.precision_score(y_test, y_pred, average=None)
    all_recall = metrics.recall_score(y_test, y_pred, average=None)

    for num_samples in [1, 10]:

        prec_scores = np.empty(20)
        recall_scores = np.empty(20)
        prec_scores_noaugm = np.empty(20)
        recall_scores_noaugm = np.empty(20)
        i = 0
        for class_num in xrange(2, 20):
            y_test = np.load("/home/jasper/oneshot-gestures/output/y_tests/class{}.npy".format(class_num))
            y_pred = np.load(
                "/home/jasper/oneshot-gestures/output/model-19x1-redo/class-{}/layers1-samples{}/y_predictions.npy".format(
                    class_num, num_samples))
            y_pred2 = np.load(
                "/home/jasper/oneshot-gestures/output/model-19x1-redo/class-{}/layers1-samples{}noaugm/y_predictions.npy".format(
                    class_num, num_samples))
            prec_scores[i] = metrics.precision_score(y_test, y_pred, labels=[19], average=None)
            recall_scores[i] = metrics.recall_score(y_test, y_pred, labels=[19], average=None)
            prec_scores_noaugm[i] = metrics.precision_score(y_test, y_pred2, labels=[19], average=None)
            recall_scores_noaugm[i] = metrics.recall_score(y_test, y_pred2, labels=[19], average=None)
            i = i + 1
        if (num_samples == 1):
            ax0.plot(xrange(20), prec_scores, )
            ax0.plot(xrange(20), prec_scores, '.')
            ax0.plot(xrange(20), prec_scores_noaugm, )
            ax0.plot(xrange(20), prec_scores_noaugm, '.')

            ax1.plot(xrange(20), recall_scores)
            ax1.plot(xrange(20), recall_scores, '.')
            ax1.plot(xrange(20), recall_scores_noaugm, )
            ax1.plot(xrange(20), recall_scores_noaugm, '.')

        elif (num_samples == 10):
            ax3.plot(xrange(20), prec_scores, )
            ax3.plot(xrange(20), prec_scores, '.')
            ax3.plot(xrange(20), prec_scores_noaugm, )
            ax3.plot(xrange(20), prec_scores_noaugm, '.')

            ax4.plot(xrange(20), recall_scores)
            ax4.plot(xrange(20), recall_scores, '.')
            ax4.plot(xrange(20), recall_scores_noaugm, )
            ax4.plot(xrange(20), recall_scores_noaugm, '.')

    # plt.tight_layout()
    plt.show()

def plotOneshotAll():
    #----------------------------#
    #     Alle klassen P+R       #
    #----------------------------#
    # Vergelijken klasse precision recall
    # plt.figure(figsize=cm2inch(9,7))

    OS_1 = mlines.Line2D([], [], color='red', marker='.', label='1 sample')
    OS_10 = mlines.Line2D([], [], color='blue', marker='.', label="10 samples")
    base = mlines.Line2D([], [], color='green', linestyle=':',marker='.', label='baseline')

    fig, (ax0, ax1) = plt.subplots(nrows=2)

    ax0.legend(handles=(OS_1,OS_10,base))
    ax0.set_ylabel("Precision")
    ax0.set_color_cycle(['r', 'r', 'b', 'b', 'g', 'g'])
    ax0.set_xticks(xrange(20))
    ax0.set_xticklabels(xrange(20))

    ax1.set_ylabel("Recall")
    ax1.set_color_cycle(['r', 'r', 'b', 'b', 'g', 'g'])
    ax1.set_xticks(xrange(20))
    ax1.set_xticklabels(xrange(20))
    ax1.set_xlabel("Klassenummer")

    y_pred=np.load("/home/jasper/oneshot-gestures/output/conf_matrix_data/all.npy")
    y_test = np.load("/home/jasper/oneshot-gestures/output/y_tests/class19.npy")
    all_prec=metrics.precision_score(y_test,y_pred,average=None)
    all_recall=metrics.recall_score(y_test,y_pred,average=None)

    for num_samples in [1,10]:

        prec_scores = np.empty(20)
        recall_scores = np.empty(20)
        i = 0
        for class_num in xrange(20):
            y_test=np.load("/home/jasper/oneshot-gestures/output/y_tests/class{}.npy".format(class_num))
            y_pred=np.load("/home/jasper/oneshot-gestures/output/model-19x1-redo/class-{}/layers1-samples{}/y_predictions.npy".format(class_num,num_samples))
            prec_scores[i] = metrics.precision_score(y_test, y_pred, labels=[19], average=None)
            recall_scores[i] = metrics.recall_score(y_test, y_pred, labels=[19], average=None)
            i=i+1
        ax0.plot(xrange(20),prec_scores,)
        ax0.plot(xrange(20), prec_scores, '.')
        ax1.plot(xrange(20),recall_scores,label="recall score")
        ax1.plot(xrange(20), recall_scores, '.')


    ax0.plot(xrange(20),all_prec,':')
    ax0.plot(xrange(20), all_prec,'.')
    ax1.plot(xrange(20),all_recall,':')
    ax1.plot(xrange(20), all_recall,'.')


    # plt.tight_layout()
    # plt.show()
    plt.show()

def plotActivationFunctions():
    # ----------------------------#
    # Plot van activatiefuncties #
    # ----------------------------#

    xaxis = np.arange(-2, 2, 0.1)
    sig = sigmoid(xaxis)
    tanh = np.tanh(xaxis)
    relu = xaxis * (xaxis > 0)
    plt.figure(figsize=cm2inch(9, 7))
    plt.xlabel(r'$x$')
    plt.ylabel(r'$a(x)$')
    plt.yticks([-1, 0, 1, 2])
    plt.tick_params(axis='both', which='major', labelsize=8)
    plt.tick_params(axis='both', which='minor', labelsize=8)
    plt.plot(xaxis, sig, 'C0--')
    plt.plot(xaxis, tanh, 'C1:')
    plt.plot(xaxis, relu, 'C2')

    plt.legend(handles=(mlines.Line2D([], [], color='C0', linestyle='--', label=r'$Sigmo\ddot{\imath}de$'),
                        mlines.Line2D([], [], color='C1', linestyle=':', label=r'$Tanh$'),
                        mlines.Line2D([], [], color='C2', label=r'$ReLU$')))
    plt.tight_layout()
    plt.show()

def plotVgl19x1Naive():
    y_test = np.load("/home/jasper/oneshot-gestures/output/y_tests/class15.npy")
    y_test_2 = np.load("/home/jasper/oneshot-gestures/output/y_tests/class19.npy")
    num_samples_array = [1, 2, 3, 4, 5, 10, 25, 50, 100, 200]
    y_predictions_1 = np.empty((len(num_samples_array), 2000))
    y_predictions_2 = np.empty((len(num_samples_array), 2000))

    i = 0
    for num_samples in num_samples_array:
        y_predictions_1[i] = np.load(
            "{}output/model-19x1-temp/class-15/layers1-samples{}/y_predictions.npy".format(BASE_DIR, num_samples))
        y_predictions_2[i] = np.load(
            "{}output/naive_model/class-15/layers1-samples{}/y_predictions.npy".format(BASE_DIR, num_samples))

        i += 1
    plt.figure(figsize=cm2inch(16, 11))

    # plt.grid(True)
    prec_scores_1 = np.zeros(len(y_predictions_1))
    recall_scores_1 = np.zeros(len(y_predictions_1))

    prec_scores_2 = np.zeros(len(y_predictions_2))
    recall_scores_2 = np.zeros(len(y_predictions_2))

    i = 0
    for y_pred in y_predictions_1:
        prec_scores_1[i] = metrics.precision_score(y_test, y_pred, labels=[19], average=None)
        recall_scores_1[i] = metrics.recall_score(y_test, y_pred, labels=[19], average=None)
        i += 1
    i = 0
    for y_pred in y_predictions_2:
        prec_scores_2[i] = metrics.precision_score(y_test_2, y_pred, labels=[15], average=None)
        recall_scores_2[i] = metrics.recall_score(y_test_2, y_pred, labels=[15], average=None)
        i += 1

    print(recall_scores_1)
    print(recall_scores_2)

    plt.xlabel(r"Aantal samples")

    x_labels = num_samples_array
    # plt.ylabels(np.arange(0,1,0.1))
    plt.yticks(np.arange(0, 1.1, 0.1))

    plt.plot(x_labels, prec_scores_1, 'b')
    plt.plot(x_labels, recall_scores_1, 'r')

    plt.plot(x_labels, prec_scores_1, 'b.')
    plt.plot(x_labels, recall_scores_1, 'r.')

    plt.plot(x_labels, prec_scores_2, 'c:')
    plt.plot(x_labels, recall_scores_2, 'm:')

    # plt.plot(x_labels, prec_scores_2, 'cx')
    # plt.plot(x_labels, recall_scores_2, 'mx')

    plt.axhline(y=0.97, xmin=0, xmax=1, linewidth=0.5, color='blue', linestyle=':', hold=None)
    plt.axhline(y=0.95, xmin=0, xmax=1, linewidth=0.5, color='red', linestyle=':', hold=None)

    blue_line = mlines.Line2D([], [], color='blue', marker='.', label='precision softmax19+1')
    red_line = mlines.Line2D([], [], color='red', marker='.', label="recall softmax19+1")
    blue_line_2 = mlines.Line2D([], [], color='cyan', linestyle=':', label='precision softmax20')
    red_line_2 = mlines.Line2D([], [], color='magenta', linestyle=':', label="recall softmax20")
    blue_dash = mlines.Line2D([], [], color='blue', linestyle=':', linewidth=0.5, label='precision baseline')
    red_dash = mlines.Line2D([], [], color='red', linestyle=':', linewidth=0.5, label='recall baseline')

    plt.legend(handles=(blue_line,red_line , blue_line_2, red_line_2, blue_dash, red_dash))

    ax = plt.subplot(111)
    ax.set_xscale('log')
    ax.set_xticks(x_labels)
    ax.get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter())

    plt.title("", size=12)
    plt.tight_layout()
    plt.show()


def main():
    mpl.rc('font', **{'size':'11','family': 'serif', 'sans-serif': ['Computer Modern']})
    mpl.rc('text', usetex='true')
    mpl.rc('lines',linewidth=1)




    # plotNaiveLayers()
    # plotNaive15Layers2()
    # plotNaive15()
    # plotOneshotAll()
    # plot14()
    # plot15()
    # plot15augm()
    # plotVgl15Augm()
    plot18x2()
    # plotVgl19x1Naive()



    # plotNaive15()
    # plot14()
    # plot15()
    # plotDataAugm()
    # plotOneshotAll()
    # plotActivationFunctions()

    # plotNaive15()
    #
    # plotNaive15Layers2()

    #
    # y_test = np.load("/home/jasper/oneshot-gestures/output/y_tests/class19.npy")
    # y_pred=np.load("/home/jasper/oneshot-gestures/output/all.npy")
    #
    #
    # print(metrics.classification_report(y_test,y_pred,digits=6))
    #
    # print(metrics.accuracy_score(y_test,y_pred))





if __name__ == "__main__":
    main()

