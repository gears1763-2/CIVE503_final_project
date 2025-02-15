#! /usr/bin/bash

pdoc -o python/docs/ -d numpy --math python/
rm -rf python/__pycache__
rm python/requirements.txt
pipreqs python/
