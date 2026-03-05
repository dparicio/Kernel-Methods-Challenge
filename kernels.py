import numpy as np

class Kernel:
    def __call__(self, X1, X2=None):
        X1 = np.asarray(X1, dtype=np.float64)
        # If X2 not provided, compute kernel between X1 and itself
        X2 = X1 if X2 is None else np.asarray(X2, dtype=np.float64)
        # 
        return self._kernel(X1, X2)

    def _kernel(self, X1, X2):
        raise NotImplementedError("This is an abstract class! Implement the method")
    

class LinearKernel(Kernel):
    def _kernel(self, X1, X2):
        return X1 @ X2.T


class PolynomialKernel(Kernel):
    def __init__(self, c=0, d=3):
        self.c = float(c)
        self.d = int(d)

    def _kernel(self, X1, X2):
        return (X1 @ X2.T + self.c)**self.d


class RBFKernel(Kernel):
    def __init__(self, gamma):
        self.gamma = float(gamma)

    def _kernel(self, X1, X2):
        # Compute squared distance by developing ||x-y||^2 
        X1_sq = np.sum(X1**2, axis=1).reshape(-1, 1)  
        X2_sq = np.sum(X2**2, axis=1).reshape(1, -1) 
        dists_sq = X1_sq + X2_sq - 2 * (X1 @ X2.T)
        dists_sq = np.maximum(dists_sq, 0)

        # Compute RBF kernel
        K = np.exp(-dists_sq * self.gamma)
        return K


class ChiSquareKernel(Kernel):
    def __init__(self, gamma=1.0, eps=1e-12):
        self.gamma = float(gamma)
        self.eps = float(eps)

    def _kernel(self, X1, X2):
        # Compute chi-square distance:
        # d(x, y) = sum_i (x_i - y_i)^2 / (x_i + y_i + eps)
        n1, d = X1.shape
        n2 = X2.shape[0]
        dists = np.zeros((n1, n2), dtype=np.float64)

        for i in range(d):
            x1_i = X1[:, i].reshape(-1, 1)
            x2_i = X2[:, i].reshape(1, -1)
            denom = x1_i + x2_i + self.eps
            diff = x1_i - x2_i
            dists += (diff * diff) / denom

        # Exponential chi-square kernel
        K = np.exp(-self.gamma * dists)
        return K
