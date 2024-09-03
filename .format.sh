#!/bin/sh

find -E src/ -regex '.*\.(cpp|c|hpp|h)' -exec clang-format -style=file -i {} \;
