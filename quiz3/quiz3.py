# # Quiz 3 - Programming Excercises
#
# ## Logistic Regression
#
# In this exercise, you will implement a Logistic Regression model and explore different stepsize policies for the Stochastic Gradient Descent algorithm using the [Breast Cancer Wisconsin dataset](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html).
#
# The code below prepares the data required for this exercise.

# +
import numpy as np
from sklearn.datasets import load_breast_cancer

# load the data
X, y = load_breast_cancer(return_X_y=True)

print(f"Shape of input data: {X.shape}")
print(f"Shape of output data: {y.shape}")

mdata, ndim = X.shape 

# convert the output space from {0,1} into {-1,+1}
y = 2*y - 1

# normalization
X /= np.outer(np.ones(mdata), np.max(np.abs(X),0))


# -

# **Task 1**: Complete the `logreg_sgd_cls` class provided below to implement a Logistic Regression class.
#
# <div class="alert alert-info">
# <b>Note:</b> Be aware that the Logistic Regression algorithm presented in Lecture 5 may differ from the one implemented in Sklearn. Please use the version presented in the Lecture.
# </div>

class logreg_sgd_cls:
    """ 
    Logistic Regression Classifier
     
    """
    def __init__(self):
        self.nitermax = 100     # maximum iteration
        self.eta = 0.1          # initial step size
        self.w = None           # weights to learn

    ## ------------------------------------------
    def fit(self, X, y, eta, nitermax, diminish=0):
        """ Train the logistic regression model
            with stochastic gradient descent algorithm

        Input:  X         2D array where each row represents an input example
                y         1D array(vector) of +1,-1 labels
                eta       initial step size
                nitermax  maximum number of iterations
                diminish  =0 constant step size, =1 diminishing step size
        """
        mtrain, ndim = X.shape

        # initialize the weights
        w = np.zeros(ndim)

        # iterations on the full data
        for t in range(nitermax):

            # select the stepsize
            if diminish == 1:       # diminishing stepsize policy
                etat = eta/(t+1)    # t+1 to avoid division by zero
            else:                   # constant stepsize policy
                etat = eta

            # perform one step of stochastic gradient descent on each training example
            for i in range(mtrain):
                # TODO: implement the weight update
                pass



                

        # save the final weights to the model
        self.w  = w

        return(w)

    ## ------------------------------------------
    def predict(self, X):
        """ Predict the labels

        Input:
            X   2D array where each row represents an input example

        Output:
            y   1D array of predicted labels
        """

        # TODO: implement the prediction process X, self.w -> y
        y = None
        
        return(y)

# **Task 2**: Complete the helper function below for running n-Fold cross validation.

# +
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score

def learning_cycle(X, y, niteration, eta, stepsize_type, nfold):
    """
    Helper function for running n-Fold cross validation
    Input:  X              2d array of inputs
            y              1d array of outputs
            niteration     number of iteration for gradient descent
            eta            initial stepsize
            stepsize_type  =0 constant, =1 diminishing
            nfold          number of folds for cross-validation
    Output: xf1score       1D array of size [nfold] containing the f1 scores for each fold        
    """

    # split the data into n folds
    cselection = KFold(n_splits=nfold, random_state=None, shuffle=False)

    # array to collect the results for each fold
    xf1score = np.zeros(nfold)
    
    # create an instance of the logistic regression model
    clogreg = logreg_sgd_cls()

    # perform n-fold cross-validation
    for ifold, (index_train, index_test) in enumerate(cselection.split(X)):
        # TODO: train the model, and evaluate using F1 score for each fold

        # xf1score[ifold] = ...



        
        
    return(xf1score)
# -

# **Task 3**: Execute n-fold cross-validation and answer question 4 according to the results.

# +
def main(iworkmode):
    # Learning hyperparameters
    # create the list of initial step sizes (learning rates)
    neta = 40   # number of different step size
    eta0 = 0.2
    # list of initial step sizes
    leta = [ eta0*(i+1) for i in range(neta)]

    # number of iterations in gradient descent
    iteration = 50

    # number of folds 
    nfold = 5

    # fix the random seed
    rng = np.random.default_rng(12345)

    # number of different stepsize policies
    nstepsize_type = 2

    # array to collect the test scores for the two stepsize policies
    xmean_test_results = np.zeros((neta, nstepsize_type))


    # Run cross-validation
    # enumerate stepsize policies
    for istepsize in [0, 1]:    # 0 - constant, 1 - diminishing

        xf1score_eta = np.zeros(neta)

        # enumerate the (initial) stepsizes
        for ieta in range(neta):
            eta = leta[ieta]

            # n-fold cross-validation
            xf1score = learning_cycle(X, y, iteration, eta, istepsize, nfold)
            # compute the mean of the test accuracies
            xf1score_eta[ieta] = np.mean(xf1score)

        xmean_test_results[:, istepsize] = xf1score_eta

    # values of the curves in the graph  
    for i in range(nstepsize_type):
        for j in range(neta):
            print('%6.4f'%leta[j],'%6.4f'%xmean_test_results[j, i])
