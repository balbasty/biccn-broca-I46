#!/bin/bash


ROOT="."

for AFFINE in $ROOT/mri2oct/affine*/affine.lta
do

AFFDIR=$(dirname $AFFINE)
echo $AFFDIR

nitorch reslice \
	"$ROOT/mri/crop/mri.crop.mgz" \
	-l $AFFINE \
	-t "$ROOT/oct/oct.nii" \
	-o "$AFFDIR/mri.crop.reslice.12um.mgz"

nitorch reslice \
	"$ROOT/mri/crop/mri.crop.mgz" \
	-l $AFFINE \
	-t "$ROOT/oct/pool3x3x3/oct.padwin33.nii" \
	-o "$AFFDIR/mri.crop.reslice.36um.mgz" 

nitorch reslice \
	"$ROOT/mri/crop/vessels.manual.crop.mgz" \
	-l $AFFINE \
	-t "$ROOT/oct/oct.nii" \
	-o "$AFFDIR/vessels.manual.crop.reslice.12um.mgz" \
	-i 0 

nitorch reslice \
	"$ROOT/mri/crop/vessels.manual.crop.mgz" \
	-l $AFFINE \
	-t "$ROOT/oct/pool3x3x3/oct.padwin33.nii" \
	-o "$AFFDIR/vessels.manual.crop.reslice.36um.mgz" \
	-i 0 

done
