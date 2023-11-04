import numpy as np

class CCE():
    """Implementation of the categorical cross entropy as a loss function"""

    def __call__(self, y_true, y_pred):
        return -np.sum(y_true * np.log(y_pred), axis=1)