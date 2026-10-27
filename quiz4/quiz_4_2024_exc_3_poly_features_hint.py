## ####################################################
import sys
import numpy as np

from sklearn.model_selection import KFold
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
## ###################################################
# for demonstration
import matplotlib
import matplotlib.pyplot as plt
from math import factorial

## ##############################################################


def explicit_poly_features(Xin, c=1, degree=2):
    """
    Task: to transform the input features into explicit polynomial ones
    Input: Xin     2d array of input features
           c       constant term in (<x,y>+c)**degree
           degree  the degree of the polynomial kernel
    Output: Xpoly  2d array of the explicit polynomial feataures       
    """
    pf = PolynomialFeatures(degree=degree, include_bias=True)
    Xpf = pf.fit_transform(Xin)
    coefs = []
    for powers in pf.powers_:
        r = degree - powers.sum()  # leftover power for the constant c
        multinom = factorial(degree) / (np.prod([factorial(p) for p in powers]) * factorial(r))
        coefs.append(np.sqrt(multinom * c**r))
    Xpoly = Xpf * np.array(coefs)

    return (Xpoly)

## ##############################################################


def main(iworkmode):

    igraph = 1  # =1 show graph, =0 not

    m = 100  # sample size

    rng = np.random.default_rng(12345)

    # loading the data file
    dataname = 'quiz_4_exc_3_data_x_y.npz'
    data = np.load(dataname)
    X = data['x']
    y = data['y']

    nfold = 5  # number of folds
    ndegree = 3  # number of different dgerees tested

    # fix constant c in the polynomials
    c = 1

    # collector of the results
    xerrors = np.zeros((ndegree+1, nfold))

    # split the data into folds
    cselection = KFold(n_splits=nfold, random_state=None, shuffle=False)
    ifold = 0
    for index_train, index_test in cselection.split(X):
        Xtrain = X[index_train]
        ytrain = y[index_train]
        Xtest = X[index_test]
        ytest = y[index_test]
        mtrain = Xtrain.shape[0]
        mtest = Xtest.shape[0]
        print('Training size:', mtrain, 'Test size:', mtest)

        for idegree in range(1, ndegree+1):
            Xtrain_poly = explicit_poly_features(Xtrain, c=c, degree=idegree)
            Xtest_poly = explicit_poly_features(Xtest, c=c, degree=idegree)

            reg = LinearRegression().fit(Xtrain_poly, ytrain)
            ypred = reg.predict(Xtest_poly)

            xerrors[idegree, ifold] = mean_squared_error(ytest, ypred)

        ifold += 1

    xmeanerror = np.mean(xerrors, 1)

    for idegree in range(1, ndegree+1):
        print('Degree:', idegree, 'Error:', '%8.4f' % xmeanerror[idegree])

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
