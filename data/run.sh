#!/bin/bash

if [ "$1" != "" ]; then
    echo "Running file $1 in octave"
else
    echo "Need to supply file to run in octave"
    exit
fi

docker run --rm \
  -v "$PWD":/work \
  -it koopman-octave \
  octave $1
