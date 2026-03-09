import numpy as np

def rgb2greyscale(X):
    N, P = X.shape
    n = P // 3
    H = W = int(np.sqrt(n)) 

    Xc = X.reshape(N, 3, H, W)         
    X_gray = 0.2989 * Xc[:, 0] + 0.5870 * Xc[:, 1] + 0.1140 * Xc[:, 2]
    return X_gray   


def compute_gradients(X_gray):
    gx = np.zeros_like(X_gray)
    gy = np.zeros_like(X_gray)
    gx[:, :, 1:-1] = X_gray[:, :, 2:] - X_gray[:, :, :-2]
    gy[:, 1:-1, :] = X_gray[:, 2:, :] - X_gray[:, :-2, :]
    return gx, gy


def block_normalize(cell_hist, block_size=2, eps=1e-6, clip=0.2):
    ncy, ncx, bins = cell_hist.shape
    blocks = []

    for by in range(ncy - block_size + 1):
        for bx in range(ncx - block_size + 1):
            block = cell_hist[by:by+block_size, bx:bx+block_size].ravel()

            # L2-Hys normalization
            block = block / np.sqrt(np.sum(block**2) + eps**2)
            block = np.minimum(block, clip)
            block = block / np.sqrt(np.sum(block**2) + eps**2)

            blocks.append(block)

    return np.concatenate(blocks)


def extract_hog_features(X, cell_size=8, bins=9, block_size=2, eps=1e-6, clip=0.2):
    # Convert to grayscale
    X_grey = rgb2greyscale(X)

    # Recover image shape and dimensions
    N, P = X.shape
    n_pixels = P // 3
    H = W = int(np.sqrt(n_pixels))

    # Compute gradient, magnitude and angle
    gx, gy = compute_gradients(X_grey)
    mag = np.sqrt(gx**2 + gy**2)
    ang = (np.arctan2(gy, gx) * 180.0 / np.pi) % 180.0

    # Build cell histograms
    ncy, ncx = H // cell_size, W // cell_size
    mag = mag[:, :ncy*cell_size, :ncx*cell_size]
    ang = ang[:, :ncy*cell_size, :ncx*cell_size]

    features = []
    for i in range(N):
        # build cell histograms grid
        cell_hist = np.zeros((ncy, ncx, bins), dtype=np.float32)
        for cy in range(ncy):
            for cx in range(ncx):
                # Select appropriate cell region
                x0 = cx * cell_size
                y0 = cy*cell_size
                cell_mag = mag[i, y0:y0+cell_size, x0:x0+cell_size].ravel()
                cell_ang = ang[i, y0:y0+cell_size, x0:x0+cell_size].ravel()
                # Compute histogram for the cell and save
                hist, _ = np.histogram(cell_ang, bins=bins, range=(0, 180), weights=cell_mag)
                cell_hist[cy, cx] = hist

        # Normalize blocks and flatten
        hog_vec = block_normalize(cell_hist, block_size, eps, clip) 
        features.append(hog_vec)

    return np.vstack(features)