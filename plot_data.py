
from __future__ import print_function
import sys
import numpy as np
import time
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.lines as mlines
import math

import os
import util.dataprocessing
from sklearn import metrics
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



def plotPrecRec(y_test, y_predictions, x_labels, baseline_prec, baseline_rec, title="F1 score"):

    # plt.grid(True)
    prec_scores = np.zeros(len(y_predictions))
    recall_scores = np.zeros(len(y_predictions))

    i = 0
    for y_pred in y_predictions:
        prec_scores[i] = metrics.precision_score(y_test, y_pred, labels=[19], average=None)
        recall_scores[i] = metrics.recall_score(y_test, y_pred, labels=[19], average=None)

        i += 1

    plt.xlabel(r"Aantal samples")

    plt.plot(x_labels,prec_scores, 'b')
    plt.plot(x_labels,recall_scores, 'r')

    plt.plot(x_labels,prec_scores, 'bo')
    plt.plot(x_labels,recall_scores, 'ro')

    plt.axhline(y=baseline_prec, xmin=0, xmax=1, linewidth=1, color='blue', linestyle=':', hold=None)
    plt.axhline(y=baseline_rec, xmin=0, xmax=1, linewidth=1, color='red', linestyle=':', hold=None)

    blue_line = mlines.Line2D([], [], color='blue', marker='o', label='precision')
    red_line = mlines.Line2D([], [], color='red', marker='o', label="recall")
    blue_dash = mlines.Line2D([], [], color='blue', linestyle=':', linewidth=1, label='precision baseline')
    red_dash = mlines.Line2D([], [], color='red', linestyle=':', linewidth=1, label='recall baseline')


    plt.legend(handles=(blue_line,blue_dash,red_line,red_dash))


    ax = plt.subplot(111)
    ax.set_xscale('log')
    ax.set_xticks(x_labels)
    ax.get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter())

    plt.title(title,size=12)
    plt.tight_layout()



def main():
    mpl.rc('font', **{'size':'11','family': 'serif', 'sans-serif': ['Computer Modern']})
    mpl.rc('text', usetex='true')


    xaxis = np.arange(-2,2,0.1)
    sig=sigmoid(xaxis)

    tanh=np.tanh(xaxis)
    relu=xaxis*(xaxis>0)


    plt.figure(figsize=cm2inch(9, 7))
    plt.xlabel(r'$x$')
    plt.ylabel(r'$a(x)$')
    plt.yticks([-1,0,1,2])
    plt.tick_params(axis='both', which='major', labelsize=8)
    plt.tick_params(axis='both', which='minor', labelsize=8)
    plt.plot(xaxis,sig,'C0--')
    plt.plot(xaxis,tanh,'C1:')
    plt.plot(xaxis,relu,'C2')

    plt.legend(handles=(mlines.Line2D([], [], color='C0', linestyle='--', label=r'$Sigmo\ddot{\imath}de$'),
                        mlines.Line2D([], [], color='C1', linestyle=':', label=r'$Tanh$'),
                        mlines.Line2D([], [], color='C2',  label=r'$ReLU$')))
    plt.tight_layout()
    plt.show()

    y_test = np.load("/home/jasper/oneshot-gestures/output/y_tests/class15.npy")
    num_samples_array = [1,2,3,4,5,25,50,100,200]
    y_predictions = np.empty((len(num_samples_array),2000))
    i = 0
    for num_samples in num_samples_array:
        y_predictions[i] = np.load("{}output/model-19x1-temp/class-15/layers1-samples{}/y_predictions.npy".format(BASE_DIR,num_samples))
        i+=1
    plt.figure(figsize=cm2inch(14, 9))
    plotPrecRec(y_test,y_predictions,num_samples_array,0.97,0.95,"Oneshot gebaar 15")

    # plt.show()

    y_test = np.load("/home/jasper/oneshot-gestures/output/y_tests/class14.npy")
    num_samples_array = [1, 2, 3, 4, 5, 200]
    y_predictions = np.empty((len(num_samples_array), 2000))
    i = 0
    for num_samples in num_samples_array:
        y_predictions[i] = np.load("{}output/model-19x1-temp/class-14/layers1-samples{}/y_predictions.npy".format(BASE_DIR, num_samples))
        i += 1
    plt.figure(figsize=cm2inch(14, 9))
    plotPrecRec(y_test, y_predictions, num_samples_array, 0.56, 0.74, "Oneshot gebaar 14")

    # plt.show()




if __name__ == "__main__":
    main()