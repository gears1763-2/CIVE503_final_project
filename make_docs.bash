#! /usr/bin/bash

pdoc -o python/docs/ -d numpy --math python/
rm -rf python/__pycache__
rm -rf data/checkpoint.torch
rm python/requirements.txt
pipreqs python/
