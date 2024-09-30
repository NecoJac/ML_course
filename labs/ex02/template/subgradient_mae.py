import numpy as np


def compute_subgradient_mae(y, tx, w):
    """Compute a subgradient of the MAE at w.

    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,2)
        w: numpy array of shape=(2, ). The vector of model parameters.

    Returns:
        A numpy array of shape (2, ) (same shape as w), containing the subgradient of the MAE at w.
    """
    subgras=np.array([0,0])
    for i in range(len(y)):
        yi=y[i]
        xi=tx[i,:]
        e=yi-xi.dot(w)
        if e!=0:

            subgras = subgras+e/abs(e)*(-xi)
    return subgras/len(y)
