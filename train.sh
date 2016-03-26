#!/usr/bin/env bash

SOLVER='solver.prototxt'

if [ ! -z $1 ]; then
    SOLVER="${1}_${SOLVER}"
fi

SOLVER="./models/Ria_Gurtow/${SOLVER}"

echo $SOLVER    

caffe train -solver $SOLVER -weights ./models/VGG_S_rgb/EmotiW_VGG_S.caffemodel