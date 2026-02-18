import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt


def load_data(data_dir="../"):
    Xtr = np.array(pd.read_csv(os.path.join(data_dir, 'Xtr.csv'), header=None,sep=',', usecols=range(3072)))
    Xte = np.array(pd.read_csv(os.path.join(data_dir, 'Xte.csv'), header=None,sep=',', usecols=range(3072)))
    Ytr = np.array(pd.read_csv(os.path.join(data_dir, 'Ytr.csv'), sep=',', usecols=[1])).squeeze()

    print("Data loaded successfully")
    print("Xtr.shape:", Xtr.shape)
    print("Xte.shape:", Xte.shape)
    print("Ytr.shape:", Ytr.shape)

    return Xtr, Xte, Ytr

def normalize_data(Xtr, Xte):
    # Compute training std
    std_Xtr = Xtr.std()
    mean_Xtr = Xtr.mean()  # just for info

    print("Before scaling:")
    print("Overall mean:", mean_Xtr)
    print("Overall std:", std_Xtr)

    # Scale both datasets by training std
    Xtr /= std_Xtr
    Xte /= std_Xtr

    # Compute new stats
    print("\nAfter scaling:")
    print("Xtr mean:", Xtr.mean())
    print("Xtr std:", Xtr.std())
    print("Xte mean:", Xte.mean())
    print("Xte std:", Xte.std())

    # Save the values for later
    print("\nValues to save for later use:")
    print("mean_Xtr =", mean_Xtr)
    print("std_Xtr =", std_Xtr)

    return std_Xtr, Xtr, Xte

def show_images_per_label(X, Y, n_per_label=9):
    """
    Display n_per_label images for each class label 0..9
    in a 3x3 grid.
    """
    num_classes = 10
    
    for label in range(num_classes):
        # Find indices where the label matches
        idxs = [i for i, y in enumerate(Y) if y == label][:n_per_label]
        
        if len(idxs) == 0:
            continue
        
        plt.figure(figsize=(6,6))
        plt.suptitle(f"Label {label}", fontsize=16)
        
        for i, idx in enumerate(idxs):
            img_flat = X[idx]
            R = img_flat[:1024].reshape(32,32)
            G = img_flat[1024:2048].reshape(32,32)
            B = img_flat[2048:].reshape(32,32)
            img = np.stack([R,G,B], axis=2)
            
            # Scale for visualization
            img_vis = (img - img.min()) / (img.max() - img.min())
            
            plt.subplot(3,3,i+1)
            plt.imshow(img_vis)
            plt.axis("off")
        
        plt.show()
    print("""
    Based on the visualization, the labels correspond to:

    - Label 0: plane
    - Label 1: car
    - Label 2: bird
    - Label 3: cat
    - Label 4: deer
    - Label 5: dog
    - Label 6: frog
    - Label 7: horse
    - Label 8: boat
    - Label 9: truck
    """)


def create_submission(predictions, filename="submission.csv"):
    """
    Converts a vector of integer labels into a Kaggle submission file.
    
    Args:
        predictions (list or np.ndarray): Vector of integer labels (0-9).
        filename (str): Name of the CSV file to save.
    
    Returns:
        pd.DataFrame: The submission DataFrame.
    """
    # Ensure predictions are a NumPy array
    predictions = np.array(predictions, dtype=int)
    
    # Create Id column starting from 1
    submission_df = pd.DataFrame({
        "Id": np.arange(1, len(predictions) + 1),
        "Prediction": predictions
    })
    
    # Save to CSV with header, no index
    submission_df.to_csv(filename, index=False)
    
    return submission_df
