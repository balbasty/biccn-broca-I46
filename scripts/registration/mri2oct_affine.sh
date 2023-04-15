#!/bin/bash

ROOT="."

MRIVOL="$ROOT/mri/crop/mri.crop.mgz"
MRIGM="$ROOT/mri/crop2/gray.crop.nii"
MRIWM="$ROOT/mri/crop2/white.crop.nii"
MRIVSL="$ROOT/mri/crop/vessels.manual.crop.mgz"

OCTVOL="$ROOT/oct/pool3x3x3/oct.win33.nii"
OCTMSK="$ROOT/oct/pool3x3x3/mask.nii"
OCTGM="$ROOT/oct/gray.lowres.mgz"
OCTWM="$ROOT/oct/white.lowres.mgz"
OCTVSL="$ROOT/oct/pool4x4x4/vessels.manual.float.nii.gz"
OCTDST="$ROOT/oct/pool4x4x4/vessels.manual.distance.nii.gz"


LOSS_IMAGE=(\
@loss lcc --patch 5 % \
	@@mov $MRIVOL \
	@@fix $OCTVOL --mask $OCTMSK \
)
LOSS_VESSELS=(\
@loss ndot 0.1 \
	@@mov $MRIVSL --label 255 --fwhm 1 \
	@@fix $OCTDST --rescale 0 \
)
LOSS_SEGMENT=( \
@loss dice 0.5 \
	@@mov $MRIGM --label --fwhm 1 \
	@@fix $OCTGM --label \
@loss dice 0.5 \
	@@mov $MRIWM --label --fwhm 1 \
	@@fix $OCTWM --label \
)

OUT_DIR="$ROOT/mri2oct/affine"

set -x
nitorch register2 --verbose 4 --gpu --output-dir $OUT_DIR \
	"${LOSS_IMAGE[@]}" "${LOSS_SEGMENT[@]}" "${LOSS_VESSELS[@]}" \
	@affine affine --progressive \
	@@optim gn -t 1e-8 -n 256 \
	@pyramid --levels 1:5 \




