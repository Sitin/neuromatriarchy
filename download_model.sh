#!/usr/bin/env bash

pushd models/VGG_S_rgb/
    wget https://dl.dropboxusercontent.com/u/38822310/demodir/VGG_S_rgb/EmotiW_VGG_S.caffemodel
    wget https://dl.dropboxusercontent.com/u/38822310/demodir/VGG_S_rgb/mean.binaryproto
popd