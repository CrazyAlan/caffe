#!/usr/bin/env sh
# Compute the mean image from the imagenet training lmdb
# N.B. this is available in data/ilsvrc12

EXAMPLE=examples/my_img_preprocess
DATA=examples/my_img_preprocess
TOOLS=build/tools

$TOOLS/compute_image_mean $EXAMPLE/shi_funt_train_lmdb \
  $DATA/shi_funt_mean.binaryproto

echo "Done."
