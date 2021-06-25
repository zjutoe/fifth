#!/usr/bin/env bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

POSITIONAL=()
while [[ $# -gt 0 ]]; do
    key="$1"

    case $key in
        -i)
            Input="$2"
            shift 2
            ;;
        -t)
            Threshold="$2"
            shift 2
            ;;
        -T)
            Timeout="$2"
            shift 2
            ;;
        -D)
            Debug="$2"
            shift 2
            ;;
        -C)
            Cfg="$2"
            shift 2
            ;;
        *)
            # save it in an array for later
            POSITIONAL+=("$1")
            # past argument
            shift
            ;;
    esac
done
# restore positional parameters
set -- "${POSITIONAL[@]}"

export PYTHONPATH=`dirname $PWD`

[ -z $Input ] && Input=0
[ -z $Threshold ] && Threshold=3
[ -z $Debug ] && Debug=False
[ -z $Cfg ] && Cfg=config.py


# ./posematch.py play -C config.py --debug True
./posematch.py play -C $Cfg --debug $Debug -i $Input -t $Threshold
