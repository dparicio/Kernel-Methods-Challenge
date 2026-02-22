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

class CSVM:
    def __init__(self, kernel, C, eta=1e-4, max_iter=2000, tol=1e-3):
        self.kernel = kernel
        self.C = C
        self.eta = eta
        self.max_iter = max_iter
        self.tol = tol

    def fit(self, X, Y, num_classes=None):
        # Save training data
        self.X_train = X

        # Create one-hot encoding of labels in {-1, 1}
        if num_classes is None:
            num_classes = len(np.unique(Y))
        Y_labels = 2 * one_hot_encode(Y, num_classes) - 1  # shape (n_samples, num_classes)
        self.Y_labels = Y_labels

        n_samples = X.shape[0]
        K = self.kernel(X)                          # Kernel matrix
        K = (K + K.T) / 2.0                         # Ensure symmetry

        alpha = np.zeros((n_samples, num_classes), dtype=np.float64)

        for _ in range(self.max_iter):
            # Gradient step
            grad = 2 * (Y_labels - K @ alpha)
            alpha += self.eta * grad

            # Box projection via beta
            beta = np.clip(alpha * Y_labels, 0, self.C)
            alpha = Y_labels * beta

            # Equality projection: sum_i alpha_i * y_i = 0
            alpha -= Y_labels * (np.sum(Y_labels * alpha, axis=0) / np.sum(Y_labels**2, axis=0))

            # Convergence check
            if np.linalg.norm(grad) <= self.tol:
                break

        self.alpha = alpha

        # Extract bias b for each class
        self.b = np.zeros(num_classes)
        for c in range(num_classes):
            beta_c = self.alpha[:,c] * Y_labels[:,c]
            sv_mask = (beta_c > 1e-12) & (beta_c < self.C - 1e-12)
            if np.any(sv_mask):
                self.b[c] = np.mean(Y_labels[sv_mask, c] - (K[sv_mask,:] @ (self.alpha[:,c] * Y_labels[:,c])))
            else:
                self.b[c] = 0.0

    def predict(self, X_test):
        # Kernel between test and training data
        K_test = self.kernel(X_test, self.X_train)  # shape (n_test, n_train)

        # Compute decision function for each class
        scores = K_test @ self.alpha + self.b  # shape (n_test, num_classes)

        # Predict class with highest score
        Y_pred = np.argmax(scores, axis=1)
        return Y_pred
        