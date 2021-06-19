#!/usr/bin/env bash

export PYTHONPATH=`dirname $PWD`

# ./posematch.py kf -k xieshan -i xieshan/reference.MP4 --debug True -g 9
# ./posematch.py kf -k xieshan --debug True -g 20
./posematch.py kf -k xieshan --debug False -g 14
