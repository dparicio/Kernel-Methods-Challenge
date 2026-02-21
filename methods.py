import numpy as np
from scipy.optimize import minimize
from utils import one_hot_encode


def softmax(S):
    '''
    This method computes the softmax of a matrix S normalizing along the row axis.
    Input: S (n_samples, n_classes)
    Output: softmax(S) (n_samples, n_classes) 
    '''
    S_exp = np.exp(S - np.max(S, axis=1, keepdims=True))  
    return S_exp / np.sum(S_exp, axis=1, keepdims=True)


def logsumexp(S):
    '''
    This method computes the log of the sum of exponentials of the input array S using
    log sum_c exp^{s_c} = m + log sum_c exp^{s_c - m} with m = max_c s_c
    which is numerically stable as it avoids overflow when S contains large values.
    Input: S (n_samples, n_classes)
    Output: logsumexp(S) (n_samples, 1)
    '''
    m = np.max(S, axis=1, keepdims=True)
    return m + np.log(np.sum(np.exp(S - m), axis=1, keepdims=True))


def cross_entropy_loss(Y_one_hot, S):
    '''
    This method computes the cross-entropy loss given the one-hot encoded labels and the scores S.
    Input: Y_one_hot (n_samples, n_classes)
           S (n_samples, n_classes)
    Output: cross_entropy_loss (scalar)
    '''
    logsoftmax = S - logsumexp(S) 
    return -np.mean(np.sum(Y_one_hot * logsoftmax, axis=1))


class RidgeRegression:
    def __init__(self, kernel, lmbda):
        self.kernel = kernel
        self.lmbda = lmbda

    def fit(self, X, Y, num_classes=None, epsilon=1e-10):
        # Save data
        self.X_train = X
        self.Y_train = Y

        # Create one-hot encoding of labels
        if num_classes is None: # ! assuming that Y contains all classes
            num_classes = len(np.unique(Y)) 
        Y_one_hot = one_hot_encode(Y, num_classes)

        # Compute the kernel matrix
        K = self.kernel(X)

        # Solve for alpha using the closed-form solution
        A = K + (self.lmbda * X.shape[0] + epsilon) * np.eye(K.shape[0])
        # Since A is p.d we decompose A using Cholesky into L @ L.T
        L = np.linalg.cholesky(A)
        # First solve L @ T = Y_one_hot for T, then L.T @ alpha = T
        T = np.linalg.solve(L, Y_one_hot)
        self.alpha = np.linalg.solve(L.T, T) # (n_samples, n_classes)


    def predict(self, X_test):
        # Compute the kernel matrix between test and train data
        K_test = self.kernel(X_test, self.X_train)

        # Compute predictions and get greatest 
        Y_pred_one_hot = K_test @ self.alpha
        Y_pred = np.argmax(Y_pred_one_hot, axis=1)
        return Y_pred
    

class LogisticRegression:
    def __init__(self, kernel, lmbda, max_iter=1000, tol=1e-6):
        self.kernel = kernel
        self.lmbda = lmbda
        self.max_iter = max_iter
        self.tol = tol


    def fit(self, X, Y, num_classes=None):
        def _loss_and_grad(alpha_flat):
            n_samples = K.shape[0]
            alpha = alpha_flat.reshape(-1, num_classes)
            K_alpha = K @ alpha
            S = K_alpha

            cel = cross_entropy_loss(Y_one_hot, S) * n_samples
            reg = 0.5 * self.lmbda * np.sum(alpha * K_alpha)
            loss = cel + reg

            P = softmax(S)
            grad = K @ (P - Y_one_hot) + self.lmbda * K_alpha
            
            return loss, grad.flatten()

        # Save data
        self.X_train = X
        self.Y_train = Y

        # Create one-hot encoding of labels
        if num_classes is None: # ! assuming that Y contains all classes
            num_classes = len(np.unique(Y))
        Y_one_hot = one_hot_encode(Y, num_classes)

        # Compute the kernel matrix
        K = self.kernel(X)
        K = (K + K.T) / 2.0 # to ensure symmetry

        alpha_init = np.zeros((K.shape[0], num_classes), dtype=np.float64).flatten()
        res = minimize(
            fun=_loss_and_grad,
            x0=alpha_init,
            jac=True,
            method='L-BFGS-B',
            options={'maxiter': self.max_iter}
            )

        self.alpha = res.x.reshape(-1, num_classes)


    def predict(self, X_test):
        # Compute the kernel matrix between test and train data
        K_test = self.kernel(X_test, self.X_train)

        # Compute predictions and get greatest 
        Y_pred_one_hot = K_test @ self.alpha
        Y_pred = np.argmax(Y_pred_one_hot, axis=1)
        return Y_pred
        