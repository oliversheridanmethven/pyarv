#!/bin/bash
# CMake does not support an "make uninstall", so this is the next best thing:
# Code taken from: https://stackoverflow.com/a/44649542/5134817
xargs rm < install_manifest.txt