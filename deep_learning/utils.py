def _get_function(functions, name, func):
    f = None
    if callable(func):
        f = func

    if type(name) == str:
        f = functions.get(name, f)

    return f
