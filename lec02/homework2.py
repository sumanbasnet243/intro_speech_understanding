'''
This homework defines one method, called "arithmetic".
that method, type `help homework2.arithmetic`.
'''

def arithmetic(x, y):
    """
    Performs different operations based on types of x and y.
    """
    if isinstance(x, str) and isinstance(y, str):
        return x + y
    elif isinstance(x, float) and isinstance(y, str):
        return str(x) + y
    elif isinstance(x, str) and isinstance(y, float):
        return x * int(y)
    elif isinstance(x, float) and isinstance(y, float):
        return x * y

