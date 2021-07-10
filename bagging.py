from sklearn.base import clone
import numpy as np

class OOBaggingClassifier:
    def __init__(self, base_estimator, n_estimators=200):
        '''
        Parameters
        ----------
        base_estimator: a probabilistic classifier that implements the predict_proba function, such as DecisionTreeClassifier
        n_estimators: the maximum number of estimators allowed.
        '''
        self.base_estimator_ = base_estimator
        self.n_estimators = n_estimators
        self.estimators_ = []
        self.oob_errors_ = []

    def fit(self, X, y, random_state=None):
        if random_state:
            np.random.seed(random_state)

        self.best_n = 0

        probs_oob = None
        for i in range(self.n_estimators):
            estimator = clone(self.base_estimator_)

            # construct a bootstrap sample
            
            # train on bootstrap sample
         
            # compute OOB error
            oob_error = ... # replace ... with your code

            # save the OOB error and the new model
            self.oob_errors_.append(oob_error)
            self.estimators_.append(estimator)

            # stop early if smoothed OOB error increases (for the purpose of
            # this problem, we don't stop training when the criterion is
            # fulfilled, but simply set self.best_n to (i+1)).
            if (self.best_n == 0) and (OOB criterion);  # replace OOB criterion with your code
                self.best_n = (i+1)

    def errors(self, X, y):
        '''
        Parameters
        ----------
        X: an input array of shape (n_sample, n_features)
        y: an array of shape (n_sample,) containing the classes for the input examples

        Returns
        ------
        error_rates: an array of shape (n_estimators,), with the error_rates[i]
        being the error rate of the ensemble consisting of the first (i+1)
        models.
        '''
        error_rates = []
        # compute all the required error rates
        return error_rates

    def predict(self, X):
        '''
        Parameters
        ----------
        X: an input array of shape (n_sample, n_features)

        Returns
        ------
        y: an array of shape (n_samples,) containig the predicted classes
        '''
        probs = None
        for estimator in self.estimators_:
            p = estimator.predict_proba(X)
            if probs is None:
                probs = p
            else:
                probs += p
        return np.argmax(probs, axis=1)
