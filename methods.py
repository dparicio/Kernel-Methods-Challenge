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

class SVM:
    def __init__(self, kernel, C=4, max_iter=200, tol=1e-5, eps=1e-8):
        self.C = C

        self.kernel = kernel 

        self.max_iter=max_iter
        self.tol = tol
        self.eps = eps
    
    def predict(self, K_test, Y=None, alpha=None, b=None):
        if alpha is not None:
            self.alphas=alpha
        if b is not None:
            self.b=b
        if Y is not None:
            self.Y_train=Y
        result = K_test @ (self.alphas * self.Y_train)  + self.b
        Y_pred = np.argmax(result, axis=1)

        return Y_pred

    def get_error(self, K, i, k):
        result = K[i,:] @ (self.alphas[:,k] * self.Y_train[:,k])  + self.b[k]
        return result - self.Y_train[i, k]

    def take_step(self, i1, i2, i):
        if (i1 == i2):
            return 0

        y1 = self.Y_train[i1, i]
        y2 = self.Y_train[i2, i]

        alpha1 = self.alphas[i1, i]
        alpha2 = self.alphas[i2, i]

        b = self.b

        E1 = self.get_error(self.kernel, i1, i)
        E2 = self.get_error(self.kernel, i2, i)

        s = y1 * y2

        if y1 != y2:
            L = max(0, alpha2 - alpha1)
            H = min(self.C, self.C + alpha2 - alpha1)
        else:
            L = max(0, alpha2 + alpha1 - self.C)
            H = min(self.C, alpha2 + alpha1)

        if L == H:
            return 0

        k11 = self.kernel[i1, i1]
        k12 = self.kernel[i1, i2]
        k22 = self.kernel[i2, i2]

        eta = k11 + k22 - 2 * k12

        if eta > 0:
            alpha2_new = alpha2 + y2 * (E1 - E2) / eta
            if alpha2_new >= H:
                alpha2_new = H
            elif alpha2_new <= L:
                alpha2_new = L
        else:
            # Abnormal case for eta <= 0, treat this scenario as no progress
            return 0

        # Numerical tolerance
        # if abs(alpha2_new - alpha2) < self.eps:   # this is slower
        # below is faster, not degrade the SVM performance
        if abs(alpha2_new - alpha2) < self.eps * (alpha2 + alpha2_new + self.eps):
            return 0

        alpha1_new = alpha1 + s * (alpha2 - alpha2_new)

        # Numerical tolerance
        if alpha1_new < self.eps:
            alpha1_new = 0
        elif alpha1_new > (self.C - self.eps):
            alpha1_new = self.C

        # Update threshold
        b1 = b - E1 - y1 * (alpha1_new - alpha1) * k11 - y2 * (alpha2_new - alpha2) * k12
        b2 = b - E2 - y1 * (alpha1_new - alpha1) * k12 - y2 * (alpha2_new - alpha2) * k22
        if 0 < alpha1_new < self.C:
            self.b = b1
        elif 0 < alpha2_new < self.C:
            self.b = b2
        else:
            self.b = 0.5 * (b1 + b2)


        self.alphas[i1, i] = alpha1_new
        self.alphas[i2, i] = alpha2_new

        # Error cache update
        ## if alpha1 & alpha2 are not at bounds, the error will be 0
        self.error[i1, i] = 0
        self.error[i2, i] = 0

        i_list = [idx for idx, alpha in enumerate(self.alphas[:,i]) \
                      if 0 < alpha and alpha < self.C]
        for k in i_list:
            self.error[k, i] += \
                  y1 * (alpha1_new - alpha1) * self.kernel[i1, k] \
                + y2 * (alpha2_new - alpha2) * self.kernel[i2, k] \
                + (self.b[i] - b)

        return 1


    def examine_example(self, i2, i):
        y2 = self.Y_train[i2, i]
        alpha2 = self.alphas[i2, i]
        E2 = self.get_error(self.kernel, i2, i)
        r2 = E2 * y2

        if ((r2 < -self.tol and alpha2 < self.C) or (r2 > self.tol and alpha2 > 0)):
            if len(self.alphas[(0 < self.alphas[:,i]) & (self.alphas[:,i] < self.C)]) > 1:
                if E2 > 0:
                    i1 = np.argmin(self.error[:,i])
                else:
                    i1 = np.argmax(self.error[:,i])

                if self.take_step(i1, i2, i):
                    return 1

            # loop over all non-zero and non-C alpha, starting at a random point
            i1_list = [idx for idx, alpha in enumerate(self.alphas) \
                           if 0 < alpha and alpha < self.C]
            i1_list = np.roll(i1_list, np.random.choice(np.arange(self.n)))
            for i1 in i1_list:
                if self.take_step(i1, i2):
                    return 1

            # loop over all possible i1, starting at a random point
            i1_list = np.roll(np.arange(self.n), np.random.choice(np.arange(self.n)))
            for i1 in i1_list:
                if self.take_step(i1, i2):
                    return 1

        return 0
    
    def fit(self, Y, alpha=None, b=None):
        self.Y_train = Y
        self.n = self.kernel.shape[0]
        k = self.Y_train.shape[1]
        if alpha is not None:
            self.alphas = alpha
        else:
            self.alphas = np.zeros((self.n,k))
        if b is not None:
            self.b = b
        else:
            self.b = np.zeros(k)
        self.error = np.zeros((self.n,k))
        for i in range(k):
            it = 0
            numChanged = 0
            examineAll = True
            while numChanged > 0 or examineAll:
                if it >= self.max_iter:
                    break

                numChanged = 0
                if examineAll:
                    for i2 in range(self.n):
                        numChanged += self.examine_example(i2, i)
                else:
                    i2_list = [idx for idx, alpha in enumerate(self.alphas[:,i]) if 0 < alpha and alpha < self.C]
                    for i2 in i2_list:
                        numChanged += self.examine_example(i2, i)

                if examineAll:
                    examineAll = False
                elif numChanged == 0:
                    examineAll = True

                it += 1
