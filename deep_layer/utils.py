def _get_function(functions, name, func=None):
    if func is None:
        func = name

    f = None
    if callable(func):
        f = func

    if type(name) == str:
        f = functions.get(name, f)

    return f
