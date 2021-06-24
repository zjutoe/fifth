#!/usr/bin/env bash

export PYTHONPATH=`dirname $PWD`

# ./posematch.py kf -k xieshan -i xieshan/reference.MP4 --debug True -g 9
# ./posematch.py kf -k xieshan --debug True -g 20
# ./posematch.py kf -k xieshan --debug False -g 14
# ./posematch.py kf -r xieshan/reference.MP4 -k xieshan --debug False -g 20
# ./posematch.py kf -r xieshan/reference.MP4 -k xieshan -p kick/reference.MP4 --debug False -g 20
# ./posematch.py kf -r xieshan/reference.MP4 -k xieshan -i xieshan/reference.MP4 -p kick/reference.MP4 --debug False -g 20

# ./posematch.py kf -r motions/m0/reference.MP4 -k motions/m0 -p 1.MP4 --debug True -g 9
# ./posematch.py kf -r motions/m0/reference.MP4 -k motions/m0 -p 1.MP4 --debug True -t 9


# ./posematch.py kf -r motions/m0/reference.MP4 -k motions/m0 -p 1.MP4 --debug True -i 0 -t 28
# ./posematch.py kf -k motions/m0 --debug True -i 0 -t 1

./posematch.py kf  -r motions/m3/reference.MP4 -R motions/m3/reference_again.MP4 -k motions/m3 --debug True -i 0 -t 9
# ./posematch.py kf  -r motions/m3/reference.MP4 -k motions/m3 --debug True -i 0 -t 9 -e True
