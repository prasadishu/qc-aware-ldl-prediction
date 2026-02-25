import numpy as np

def apply_qc(predictions, reference):

    mean = reference.mean()
    std = reference.std()

    upper = mean + 2 * std
    lower = mean - 2 * std

    return np.clip(predictions, lower, upper)
