def _get_function(functions, name, f):
    if callable(f):
        return f
    elif type(name) == str:
        return functions.get(name, None)
    return None
