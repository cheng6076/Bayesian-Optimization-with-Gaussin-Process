__author__ = 'mdenil'

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn import cross_validation
import pylab as pl
import random

def visualize_predictions(rf_params=None):
    """
    Train an Random Forest with input parameters and predict on 9 examples
    as well as visualize the examples against predictions.
    """
    X = np.load("data/mnistFeatures.npy")
    y = np.load("data/mnistlabels.npy")

    #print 'X='+str(X)##############

    clf = RandomForestClassifier(**rf_params)
    clf.fit(X, y)

    #3 fold cross-validation
    scores = cross_validation.cross_val_score(
	clf, X, y,
	scoring='accuracy')
    print 'average accuracy = '+ str( scores.mean() )
    print 'scores= ' + str( scores )

    predictions = clf.predict(X)

    #print 'predictions= ' + str(predictions)################
    #print 'y=' + str(y) #############

    ''' use nonzero to get the index array'''
    correct, = (predictions == y).nonzero()
    errors, = (predictions != y).nonzero()

    '''
    print  '(predictions == y):::'
    print  (predictions == y)
    print  '(predictions == y).nonzero():::'
    print  (predictions == y).nonzero()
    print 'correct= '
    print correct ############

    print  '(predictions != y):::'
    print  (predictions == y)
    print  '(predictions != y).nonzero():::'
    print  (predictions == y).nonzero()
    print 'errors= '
    print errors ######
    '''

    # show some correct and some with errors
    show = np.concatenate([correct[:4], errors[:5]])

    #print 'show= ' + str(show) ######

    f, axarr = pl.subplots(3, 3)
    for i in range(9):
        teesss = np.reshape(X[show[i]], (28, 28))
	#print 'teesss= ' + str(teesss) ######
        axarr[i/3, i%3].imshow(teesss, cmap='gray')
        axarr[i/3, i%3].set_title('prediction: {}'.format(int(predictions[i])))
        axarr[i/3, i%3].axis('off')

    pl.show()


def visualize():
    """
    Show some random digits from MNIST.
    """
    X = np.load("data/mnistFeatures.npy")
    y = np.load("data/mnistlabels.npy")

    idxs = random.sample(xrange(X.shape[0]), 9)

    f, axarr = pl.subplots(3, 3)
    for i,idx in enumerate(idxs):
        teesss = np.reshape(X[idx], (28, 28))
        axarr[i/3, i%3].imshow(teesss, cmap='gray')
        axarr[i/3, i%3].set_title('label: {}'.format(int(y[idx])))
        axarr[i/3, i%3].axis('off')

    pl.show()


if __name__ == "__main__":
    #visualize()
    visualize_predictions(
	{'max_depth': 200,
 	'max_features': 100,
 	'min_samples_leaf': 1,
 	'min_samples_split': 10,
	'n_estimators': 10}
	)
