def foo(a: int, b: str) -> int:
    """
    A small pure python function of some function we might want to
    ship and call elsewhere.

    Parameters
    ----------
    a:
        Anything.
    b:
        Anything.

    Returns
    -------
    a:
        Returns the first argument.

    """
    print(f"The input values are: {a = } and {b = }")
    return a
