## ####################################################
import sys
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score, precision_score, recall_score

## ###################################################
# for demonstration
import matplotlib
import matplotlib.pyplot as plt

## ##############################################################


class svm_primal_cls:

    def __init__(self, C=1000, eta=0.1, xlambda=0.01, nitermax=10):

        self.C = C                 ## penalty constant
        self.eta = eta             ## learning rate for the primal method
        self.nitermax = nitermax   ## maximum number of iterations
        self.xlambda = xlambda     ## lambda  = 1/C
        self.w = None              ## weight vector

        return

    # ---------------------------------------------------------
    def fit(self, X, y):
        """
        Task: to solve the support Vector Machine
              by applying stochastic gradient descent method
        Input:  X      2d array of input examples in the rows
                y      1d(vector) array of +1,-1 labels
        """
        m, n = X.shape

        # initialize the primal variables
        w = np.zeros(n)

        # iteration on the full training data
        for t in range(self.nitermax):
            # process all sample examples sequentially
            for i in range(m):
              # Stochastic gradient descent algorithm for SVM

        return

    # ------------------------------------------
    def predict(self, Xtest):
        """
        Task: to predict the labels for the given examples based on the self.w
        Input:  Xtest   2d array of input examples in the rows
        Output: y       1d array of predicted labels
        """
        # prediction

        return (y)

## ##############################################################


class svm_dual_cls:

    def __init__(self, C=1, nitermax=10):

        self.C = C                 ## penalty constant
        self.nitermax = nitermax   ## maximum number of iterations
        self.alpha = None          ## vector of Lagrangians, dual variables 
        self.Xtrain = None         ## the training examples, inputs
        self.ytrain = None         ## the training examples, outputs

        return

    # ---------------------------------------------------------
    def fit(self, X, y):
        """
        Task: to solve the support Vector Machine
              by applying stochastic dual coordinate ascent method
        Input:  X      2d array of input examples in the rows
                y      1d(vector) array of +1,-1 labels
        """
        m, n = X.shape
        self.Xtrain = X
        self.ytrain = y

        # initialize the dual variables
        alpha = np.zeros(m)
        # compute the linear kernel
        K = np.dot(X, X.T)

        # iteration on the full training data
        for t in range(self.nitermax):
            # process all sample examples sequentially
            for i in range(m):
              # Stochastic Dual Coordinate Ascent for SVM

        return

    # ------------------------------------------
    def predict(self, Xtest):
        """
        Task: to predict the labels for the given examples based on the self.alpha
        Input:  Xtest   2d array of input test examples in the rows
        Output: y       1d array of predicted labels
        """

        # predict

        return (y)

## ##############################################################
## ##############################################################


def main(iworkmode):

    igraph = 1  # =1 show graph, =0 not

    # Load the data
    X, y = load_breast_cancer(return_X_y=True)  # X input, y output
    print(X.shape, y.shape)
    # to convert the {0,1} output into {-1,+1}
    y = 2 * y - 1

    mdata, ndim = X.shape  # size of the data

    # scale the input data
    X /= np.outer(np.ones(mdata), np.max(np.abs(X), 0))

    # fix the random number generator
    rng = np.random.default_rng(12345)

    # fixed hyper parameters
    # primal learning speed
    eta = 0.1

    # penalty constant
    C = 2000

    # number of iteration
    literations = [2*(i+1) for i in range(20)]
    niteration = len(literations)

    nfold = 5
    nmethod = 2  # 0 svm_primal, 1 svm_dual

    # split the data into 5-folds
    cselection_outer = KFold(n_splits=nfold, random_state=None,
                             shuffle=False)

    # collect scores for folds, number of iterations for each learner

    # Run the cross-validation
    ifold = 0
    for index_train, index_test in cselection_outer.split(X):
        # select the folds

        for iiter in range(niteration):
            # select the iteration number

            # training of both models

            # prediction of both models

            # compute the scores: f1, precision,recall

            # end of the iteration selection loop

            # end of the fold loop

            # average the scores on the folds
            # for all methods and for all number of iterations

    if igraph == 1:

        # you might plot the results

    print('Bye')

    return


## ####################################################
## ###################################################
if __name__ == "__main__":
    if len(sys.argv) == 1:
        iworkmode = 0
    elif len(sys.argv) >= 2:
        iworkmode = eval(sys.argv[1])
    main(iworkmode)
