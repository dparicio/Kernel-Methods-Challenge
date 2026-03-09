import numpy as np

class Kernel:
    def __call__(self, X1, X2=None, **kwargs):
        X1 = np.asarray(X1, dtype=np.float64)

        # If X2 not provided, compute kernel between X1 and itself
        X2 = X1 if X2 is None else np.asarray(X2, dtype=np.float64)

        return self._kernel(X1, X2, **kwargs)

    def _kernel(self, X1, X2, **kwargs):
        raise NotImplementedError("This is an abstract class! Implement the method")

class LinearKernel(Kernel):
    def _kernel(self, X1, X2):
        X1 = np.atleast_2d(X1)
        X2 = np.atleast_2d(X2)

        return X1 @ X2.T


class PolynomialKernel(Kernel):
    def __init__(self, c=0, d=3):
        self.c = float(c)
        self.d = int(d)

    def _kernel(self, X1, X2):
        X1 = np.atleast_2d(X1)
        X2 = np.atleast_2d(X2)

        return (X1 @ X2.T + self.c)**self.d


class RBFKernel(Kernel):
    def __init__(self, gamma):
        self.gamma = float(gamma)

    def _kernel(self, X1, X2):
        X1 = np.atleast_2d(X1)
        X2 = np.atleast_2d(X2)

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

    def _kernel(self, X1, X2, block_size=256):
        X1 = np.atleast_2d(np.asarray(X1, dtype=np.float64))
        X2 = np.atleast_2d(np.asarray(X2, dtype=np.float64))

        n1, d = X1.shape
        n2 = X2.shape[0]

        K = np.empty((n1, n2), dtype=np.float64)

        for j in range(0, n2, block_size):
            j_end = min(j + block_size, n2)
            X2_block = X2[j:j_end]

            dists = np.zeros((n1, j_end - j), dtype=np.float64)

            for i in range(d):
                x1 = X1[:, i][:, None]        # (n1,1)
                x2 = X2_block[:, i][None, :]  # (1,b)

                diff = x1 - x2
                denom = x1 + x2 + self.eps

                dists += (diff * diff) / denom

            K[:, j:j_end] = np.exp(-self.gamma * dists)

        return K