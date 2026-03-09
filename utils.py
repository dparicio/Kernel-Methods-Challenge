import numpy as np

def one_hot_encode(Y, num_classes):
    n_samples = Y.shape[0]
    Y_one_hot = np.zeros((n_samples, num_classes))
    Y_one_hot[np.arange(n_samples), Y] = 1
    return Y_one_hot


def accuracy(y_true, y_pred, percentage=False):
    acc = np.mean(y_true == y_pred)
    if percentage:
        acc *= 100
    return acc


def split_train_test(X, Y, test_size=0.2, random_seed=42):
    # Set random for reproducibility
    np.random.seed(random_seed)
    
    # Shuffle indices and split
    n_samples = X.shape[0]
    indices = np.random.permutation(n_samples)
    
    test_size = int(n_samples * test_size)
    test_indices = indices[:test_size]
    train_indices = indices[test_size:]
    
    X_train, Y_train = X[train_indices], Y[train_indices]
    X_test, Y_test = X[test_indices], Y[test_indices]
    
    return X_train, Y_train, X_test, Y_test