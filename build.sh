#!/bin/bash
OS=$(uname)
python setup.py build_ext --inplace
if [ "$OS" = "Darwin" ]; then
    mv binding.cpython-312-darwin.so src
else
    mv binding.cpython-312-x86_64-linux-gnu.so src
fi
rm -rf build