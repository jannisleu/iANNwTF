import numpy as np

class CCE():
    """Implementation of the categorical cross entropy as a loss function"""

    def __call__(self, y_pred, y_true):
        return -np.sum(y_true * np.log(y_pred), axis=1)
    
    def backwards(self, y_pred, y_true):
        """_summary_

        Args:
            y_pred (array): shape (batchsize,10)
            loss (array): shape (batchsize,1)
        """
        return y_pred - y_true