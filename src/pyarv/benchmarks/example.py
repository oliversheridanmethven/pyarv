#!/usr/bin/env python3

"""
An example script to benchmark the performance of doing something in pure
Python versus using a C binding.
"""

from pyarv.gaussian import foo as foo_pure_c
from pyarv.example import foo as foo_pure_python
from common.variables import variable_names_and_objects
from common.timing import time_function


def main():
    a = 1
    foo_pure_c(a, b='c')
    foo_pure_python(a, b='python')
    for name, function in variable_names_and_objects(foo_pure_c, foo_pure_python):
        time_function(a, b=name, name=name, function=function, suppress_output=True)


if __name__ == '__main__':
    main()
