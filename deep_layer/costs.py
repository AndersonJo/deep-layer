def mean_squared_error(y_true, y_pred):
    """
    Mean Squared Error function
    """
    N = len(y_true)
    return ((y_pred - y_true) ** 2) / N


def dmean_squared_error(y_true, y_pred):
    """
    Derivative of the mean squared error function
    """
    N = len(y_true)
    return 2 * (y_pred - y_true) / N


losses = dict(mean_squared_error=mean_squared_error,
              dmean_squared_error=dmean_squared_error,
              mse=mean_squared_error,
              dmse=dmean_squared_error)
