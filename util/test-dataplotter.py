import dataprocessing
import numpy as np
if __name__=='__main__':
    dp = dataprocessing.DataPlotter()

    losses = []
    val_accuracies = []
    train_losses=[]
    for num_samples in [200,100,50]:
        losses.append(np.load("/home/jasper/oneshot-gestures/output/data-17-retrain-1-samples-{}/train_loss.npy".format(num_samples)))
        losses.append(np.load("/home/jasper/oneshot-gestures/output/data-17-retrain-1-samples-{}/val_loss.npy".format(num_samples)))
        val_accuracies.append(np.load("/home/jasper/oneshot-gestures/output/data-17-retrain-1-samples-{}/val_acc.npy".format(num_samples)))

    #dp.plotAccLoss(val_loss,val_acc)
    dp.plotCompare(losses)
    dp.plotCompare(val_accuracies)