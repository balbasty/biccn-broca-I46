#!/bin/bash

ROOT="."

nitorch reslice \
	$ROOT/oct/oct.nii \
	-il $ROOT/mri2oct/affine/affine.lta \
	-il $ROOT/mri/transforms/recon2samseg.lta \
	-o $ROOT/warp2surf/oct.2samseg.nii.gz

nitorch reslice \
	$ROOT/oct/vessels.manual.nii.gz \
	-il $ROOT/mri2oct/affine/affine.lta \
	-il $ROOT/mri/transforms/recon2samseg.lta \
	-o $ROOT/warp2surf/oct.vessels.2samseg.nii.gz

nitorch reslice \
	$ROOT/oct/pool3x3x3/oct.padwin33.nii \
	-il $ROOT/mri2oct/affine/affine.lta \
	-il $ROOT/mri/transforms/recon2samseg.lta \
	-o $ROOT/warp2surf/oct.padwin33.2samseg.nii.gz


