#!/usr/bin/env bash

export PYTHONPATH=`dirname $PWD`

./posematch.py play -C config.py --debug True

