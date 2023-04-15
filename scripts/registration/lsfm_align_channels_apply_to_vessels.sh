#!/bin/bash

ROOT="."
ROOT+="/lsfm"

for i in {08..23}
do

nitorch reslice \
	$ROOT/init_oct/vessels/vessels.${i}*.nii.gz \
	-l calb2neun/$i/affine.lta \
	-o $ROOT/align_channels/{base}.moved{ext}

done
