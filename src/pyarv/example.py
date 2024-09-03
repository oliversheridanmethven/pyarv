"""
A small pure python function of some function we might want to
ship and call elsewhere.
"""

def foo(a: int, b: str) -> int:
    print(f"The input values are: {a = } and {b = }")
    return a