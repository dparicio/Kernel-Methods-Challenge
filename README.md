# Kernel-Methods-Challenge
--------------------------------

This challenge is part of the Master MVA Kernel Methods course. We are tasked with an image classification problem involving 10 classes, and, without using external machine learning libraries, we must implement a classifier from scratch using only kernel methods.

We have implemented the following components:

- Data preprocessing: Loading matrices (CSV → NumPy), normalization, visualization (Matplotlib), and submission formatting.

- Data augmentation: Image flipping and additive Gaussian noise.

- Kernels: A Kernel abstract class with several options, including linear, polynomial, Gaussian, and chi-square kernels.

- Features: Hand-crafted HOG features extracted from images.

- Methods: Ridge Regression, Logistic Regression, and SVM using SMO, all implemented from scratch. The SVM SMO implementation is adapted from: https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/tr-98-14.pdf

- Fine-tuning: Linear combinations of different kernels for kernel finetuning.