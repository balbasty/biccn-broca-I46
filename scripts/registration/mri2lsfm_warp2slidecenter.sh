#!/bin/bash

ROOT="."
MRI="$ROOT/mri/crop/mri.crop.mgz"
INFRASUPRA="$ROOT/mri/crop/infrasupra.smooth.crop.mgz"
MRI2OCT="$ROOT/mri2oct/affine/affine.lta"

for i in {08..23}
do

AFFINE="$ROOT/oct2lsfm/nonlin/$i/affine.lta"
NONLIN="$ROOT/oct2lsfm/nonlin/$i/svf.nii.gz"
TARGET="$ROOT/lsfm/slide_center_oct/neun.${i}.nii.gz"

OUTPUT="$ROOT/warp2lsfm/slide_center_oct/mri.crop.${i}.nii.gz"
nitorch reslice \
	$MRI \
	-l $MRI2OCT \
	-l2 $AFFINE \
	-v $NONLIN \
	-l2 $AFFINE \
	-t $TARGET \
	-o $OUTPUT

OUTPUT="$ROOT/warp2lsfm/slide_center_oct/infrasupra.smooth.crop.${i}.nii.gz"
nitorch reslice \
	$INFRASUPRA \
	-l $MRI2OCT \
	-l2 $AFFINE \
	-v $NONLIN \
	-l2 $AFFINE \
	-t $TARGET \
	-o $OUTPUT \
	-i l

done

