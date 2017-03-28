def mean_squared_error(y_true, y_pred):
    N = len(y_true)
    return ((y_true - y_pred) ** 2) / N


loses = dict(mean_squared_error=mean_squared_error)
