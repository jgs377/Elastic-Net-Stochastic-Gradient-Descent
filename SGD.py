import numpy as np
from sklearn import preprocessing
from sklearn.utils import shuffle
import math


class SGD:
    def __init__(self, lambda_=0.1, alpha=0.5, learning_rate=0.05, batch_size=10, epochs=200, normalize=True):
        self.lambda_ = lambda_
        self.alpha = alpha
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.beta = None
        self.epochs = epochs
        self.pre = preprocessing.MinMaxScaler()
        self.normalize = normalize

    def fit(self, X, Y):
        # initialize variables
        lambda_ = self.lambda_
        alpha = self.alpha
        learning_rate = self.learning_rate
        batch_size = self.batch_size
        beta = np.ones(shape=(X.shape[1], 1))
        epochs = self.epochs
        iterations = int(math.ceil(X.shape[0] / batch_size))

        # normalize x values
        if self.normalize:
            X = self.pre.fit_transform(X)

        for epoch in range(epochs):
            # shuffle X, Y for every epoch
            X, Y = shuffle(X, Y)

            # for every batch in data
            for i in range(iterations):
                # cut the current batch
                x = X[i*batch_size:(i+1)*batch_size]
                y = Y[i*batch_size:(i+1)*batch_size]

                # update step
                regularization = lambda_ * (2 * alpha * beta)
                sum_of_batches = np.zeros(shape=(x.shape[1], 1))
                for j in range(x.shape[0]):
                    xi = x[j].copy()
                    xi.resize((1, x.shape[1]))
                    yi = y[j].copy()
                    yi.resize((1, 1))
                    sum_of_batches += 2 * np.dot(xi.T, np.dot(xi, beta) - yi) + regularization
                gradient = sum_of_batches
                beta = self.proximal(beta - (learning_rate / x.shape[0]) * gradient)

            self.beta = beta

        self.beta = beta

    # l1 regularization, by proximal gradient descent / iterative soft thresholding
    def proximal(self, beta):
        t = self.lambda_ * self.learning_rate * (1-self.alpha)

        for i in range(beta.shape[0]):
            if beta[i][0] >= t:
                beta[i] = beta[i][0] - t
            elif -t <= beta[i][0] <= t:
                beta[i] = 0
            elif beta[i][0] <= -t:
                beta[i] = beta[i][0] + t

        return beta

    # return predictions using beta from saved model
    def predict(self, x, flag=False):
        # if predict is called before fit, exit
        if self.beta is None:
            print("You must fit before you predict.")
            return None

        # normalize x using the same scale as before to maintain relevance of beta
        # if flag is True, then predict was called from inside the class, and doesnt
        # need to be normalized again
        if self.normalize and not flag:
            x = self.pre.transform(x)

        predictions = list()
        for row in x:
            predictions.append(np.dot(row, self.beta))

        predictions = np.array(predictions)
        predictions.resize((predictions.shape[0],))
        return predictions
