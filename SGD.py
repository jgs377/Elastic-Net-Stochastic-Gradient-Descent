import numpy as np
import pandas as pd
from sklearn import metrics, preprocessing
from sklearn.utils import shuffle
import math
import matplotlib.pyplot as plt


class SGD:
    def __init__(self, lambda_=0.0, alpha=0.5, learning_rate=0.25, batch_size=10, epochs=200, normalize=True):
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
        rmse = list()
        iterations = int(math.ceil(X.shape[0] / batch_size))

        # normalize x values
        if self.normalize:
            X = self.pre.fit_transform(X)

        # epoch is a pass through whole data
        for epoch in range(epochs):
            # shuffle X, Y for every epoch
            X, Y = shuffle(X, Y)

            # for every batch in data
            for i in range(iterations):
                # cut the current batch
                x = X[i*batch_size:(i+1)*batch_size]
                y = Y[i*batch_size:(i+1)*batch_size]

                # update step
                regularization = lambda_ * (((0.0) * self.proximal(beta)) + (2 * alpha * beta))
                sum_of_batches = np.zeros(shape=(x.shape[1], 1))
                for j in range(x.shape[0]):
                    xi = x[j].copy()
                    xi.resize((1, x.shape[1]))
                    yi = y[j].copy()
                    yi.resize((1, 1))
                    sum_of_batches += 2 * np.dot(xi.T, np.dot(xi, beta) - yi) + regularization
                gradient = sum_of_batches
                beta = beta - (learning_rate / x.shape[0]) * gradient

            self.beta = beta
            rmse.append(self.loss(X, Y))

        # create graph and return plt, so you can show() later
        plt.plot(range(1, epochs+1), rmse, 'ro')
        plt.ylabel("Loss")
        plt.xlabel("Epochs")
        plt.title("Lambda = {}, Aplha = {}, Learning rate = {}".format(self.lambda_, alpha, learning_rate))
        self.beta = beta

        return plt

    # return RMSE
    def loss(self, X, Y):
        l = metrics.mean_squared_error(y_pred=self.predict(X, flag=True), y_true=Y)
        return np.sqrt(l)

    # l1 regularization, by proximal gradient descent / iterative soft thresholding
    def proximal(self, beta):
        t = self.learning_rate

        for i in range(beta.shape[0]):
            if beta[i][0] >= t:
                beta[i] = beta[i][0] - t
            elif -t <= beta[i][0] <= t:
                beta[i] = 0
            elif beta[i][0] <= -t:
                beta[i] = beta[i][0] + t

        return beta

    # return ndarray of predictions using beta from saved model
    def predict(self, x, flag=False):
        # if predict is called before fit, exit
        if self.beta is None:
            print("You must fit before you predict.")
            exit()

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


# helper function to print RMSE and R^2
def results(y_pred, y_true):
    print "{} {:.2f}".format('RMSE:', np.sqrt(metrics.mean_squared_error(y_true, y_pred)))
    print "{} {:.4f}".format('R-sq:', metrics.r2_score(y_true, y_pred))
    return


data = pd.read_csv('./train.csv')

X = data[['lights', 'T1', 'RH_1', 'T2', 'RH_2', 'T3', 'RH_3', 'T4', 'RH_4', 'T5',
          'RH_5', 'T6', 'RH_6', 'T7', 'RH_7', 'T8', 'RH_8', 'T9', 'RH_9', 'T_out',
          'Press_mm_hg', 'RH_out', 'Windspeed', 'Visibility', 'Tdewpoint']]

Y = data['Appliances']

X = X.values
Y = Y.values

reg = SGD(batch_size=100, alpha=0, lambda_=0.03, learning_rate=0.005, epochs=100)
plt = reg.fit(X, Y)

data2 = pd.read_csv('./test.csv')
data3 = pd.read_csv('./validation.csv')

X_test_test = data2[['lights', 'T1', 'RH_1', 'T2', 'RH_2', 'T3', 'RH_3', 'T4', 'RH_4', 'T5',
                'RH_5', 'T6', 'RH_6', 'T7', 'RH_7', 'T8', 'RH_8', 'T9', 'RH_9', 'T_out',
                'Press_mm_hg', 'RH_out', 'Windspeed', 'Visibility', 'Tdewpoint']]
X_test_validation = data3[['lights', 'T1', 'RH_1', 'T2', 'RH_2', 'T3', 'RH_3', 'T4', 'RH_4', 'T5',
                'RH_5', 'T6', 'RH_6', 'T7', 'RH_7', 'T8', 'RH_8', 'T9', 'RH_9', 'T_out',
                'Press_mm_hg', 'RH_out', 'Windspeed', 'Visibility', 'Tdewpoint']]

y_test_test = data2['Appliances']
y_test_validation = data3['Appliances']

print "TRAIN"
results(reg.predict(X), Y)
print "VALIDATION"
results(reg.predict(X_test_validation), y_test_validation)
print "TEST"
results(reg.predict(X_test_test), y_test_test)

plt.show()

exit()
