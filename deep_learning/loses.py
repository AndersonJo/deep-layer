def mean_squared_error(y_true, y_pred):
    """
    Mean Squared Error function
    """
    N = len(y_true)
    return ((y_true - y_pred) ** 2) / N


def dmean_squared_error(y_true, y_pred):
    """
    Derivative of the mean squared error function
    """
    N = len(y_true)
    return 2 / N * (y_true - y_pred)


losses = dict(mean_squared_error=mean_squared_error,
              dmean_squared_error=dmean_squared_error)
