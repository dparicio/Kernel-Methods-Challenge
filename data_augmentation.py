import numpy as np

# Flip image horizontally
def horizontal_flip(X):
    N, P = X.shape
    n = P // 3
    H = W = int(np.sqrt(n)) 

    Xc = X.reshape(N, 3, H, W)         
    X_flipped = Xc[:, :, :, ::-1].reshape(N, P)
    return X_flipped


# Add random Gaussian noise to the image.
def add_random_noise(X, noise_level=0.1):
    noise = np.random.normal(0, noise_level, X.shape)
    X_noisy = X + noise
    return X_noisy


# Combine transformations for data augmentation
def data_transformations(
    X,
    Y,
    noise_level=0.1,
    only_flip=False,
    only_noise=False,
    noise_clip_range=None,
):
    # Apply horizontal flip
    X_flipped = horizontal_flip(X)

    # Apply random noise
    X_noisy = add_random_noise(X, noise_level=noise_level)

    # Combine original, flipped and noisy data
    if only_flip:
        X_augmented = np.vstack((X, X_flipped))
        return X_augmented, np.tile(Y, 2)
    elif only_noise:
        X_augmented = np.vstack((X, X_noisy))
        return X_augmented, np.tile(Y, 2)
    else:   
        X_augmented = np.vstack((X, X_flipped, X_noisy))
    return X_augmented, np.tile(Y, 3)
