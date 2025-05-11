#!/usr/bin/env bash
find . -name "*.ipe" -execdir ipetoipe -pdf {} \;
