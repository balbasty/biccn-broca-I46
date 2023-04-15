#!/bin/bash
set +x

ROOT="."

MRIVOL="$ROOT/mri/crop/mri.crop.mgz"
OCTVOL="$ROOT/oct/pool3x3x3/oct.win33.nii"
OCTMSK="$ROOT/oct/pool3x3x3/mask.nii"

MODEL=affine

do_register() 
{
KERNEL="$1"

LOSS_IMAGE=(\
@loss lcc --patch $KERNEL % \
	@@mov $MRIVOL \
	@@fix $OCTVOL --mask $OCTMSK \
)

OUT_DIR="$ROOT/mri2oct/${MODEL}_lncc-${KERNEL}"
nitorch register2 --verbose 4 --gpu --output-dir $OUT_DIR \
	"${LOSS_IMAGE[@]}"  \
	@affine $MODEL --progressive \
	@@optim gn --tolerance 1e-8 -n 256 \
	@pyramid --levels 1:5
}

do_register 5
do_register 10
do_register 20
do_register 40
do_register 60

# NCC (non local)

LOSS_IMAGE=(\
@loss cc \
	@@mov $MRIVOL \
	@@fix $OCTVOL --mask $OCTMSK \
)

OUT_DIR="$ROOT/mri2oct/${MODEL}_ncc"
nitorch register2 --verbose 4 --gpu --output-dir $OUT_DIR \
	"${LOSS_IMAGE[@]}"  \
	@affine $MODEL --progressive \
	@@optim gn --tolerance 1e-8 -n 256 \
	@pyramid --levels 1:5

