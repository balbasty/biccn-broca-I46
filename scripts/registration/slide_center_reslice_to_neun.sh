#!/bin/bash

ROOT="."
ROOT+="/lsfm/slide_center_oct"
pushd $ROOT

for i in {08..23}
do
nitorch reslice \
	../slide/vessels.${i}.distance.moved.slidecrop.nii.gz \
	-il neun.${i}.to_center.lta \
	-t neun.${i}.nii.gz \
	-o vessels.${i}.distance.nii.gz \
	--bound replicate --extrapolate

nitorch reslice \
	../slide/vessels.${i}.smooth.moved.slidecrop.nii.gz \
	-il neun.${i}.to_center.lta \
	-t neun.${i}.nii.gz \
	-o vessels.${i}.smooth.nii.gz

nitorch reslice \
	../slide/vessels.${i}.float.moved.slidecrop.nii.gz \
	-il neun.${i}.to_center.lta \
	-t neun.${i}.nii.gz \
	-o vessels.${i}.float.nii.gz

nitorch reslice \
	../slide/calb.${i}.moved.slidecrop.nii.gz \
	-il neun.${i}.to_center.lta \
	-t neun.${i}.nii.gz \
	-o calb.${i}.nii.gz

nitorch reslice \
	../slide/sst.${i}.moved.slidecrop.nii.gz \
	-il neun.${i}.to_center.lta \
	-t neun.${i}.nii.gz \
	-o sst.${i}.nii.gz

[ -f "../layers/interpolated_9.9um/layers.${i}.nii.gz" ] && \
	nitorch reslice \
	../layers/interpolated_9.9um/layers.${i}.nii.gz \
	-il neun.${i}.to_center.lta \
	-t neun.${i}.nii.gz \
	-o layers.${i}.interpol.nii.gz \
	-i l

[ -f "../layers/extracted_9.9um/layers.${i}.nii.gz" ] && \
	nitorch reslice \
	../layers/extracted_9.9um/layers.${i}.nii.gz \
	-il neun.${i}.to_center.lta \
	-t neun.${i}.nii.gz \
	-o layers.${i}.nii.gz \
	-i 0

done

popd
