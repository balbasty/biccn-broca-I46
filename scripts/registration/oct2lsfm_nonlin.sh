#!/bin/bash


ROOT="."
VERBOSE=4

for i in {08..23}; do

OCT="${ROOT}/oct/slices/unstack/oct.${i}.noisify.nii.gz"
OCTVSL="${ROOT}/oct/slices/unstack/vessels.${i}.nii.gz"

LSFM="${ROOT}/lsfm/slide_center_oct/neun.${i}.nii.gz"
LSFMVSL="${ROOT}/lsfm/slide_center_oct/vessels.${i}.float.nii.gz"
LSFMDST="${ROOT}/lsfm/slide_center_oct/vessels.${i}.distance.squeeze.nii.gz"

LOSS_IMAGE=(\
@loss lcc --patch  5 % --slicewise -1 \
	@@fix $LSFM --pad 64 64 0 --bound replicate --name lsfm \
	@@mov $OCT --bound replicate --no-missing \
)
if [ -f "$LSFMDST" ]; then
#LOSS_VESSELS=(
#@loss ndot 100 \
#	@@fix $LSFMDST --pad 64 64 0 --bound replicate --rescale 0 \
#	@@mov $OCTVSL --bound replicate --rescale 0 --fwhm 1 \
#)
LOSS_VESSELS=(
@loss dice 1 \
	@@fix $LSFMVSL --pad 64 64 0 --bound replicate --rescale 0 \
	@@mov $OCTVSL --bound replicate --rescale 0 --fwhm 1 \
)
else
LOSS_VESSELS=()
fi

nitorch register2 \
	-v $VERBOSE --gpu -o "${ROOT}/oct2lsfm/nonlin/${i}" \
	"${LOSS_IMAGE[@]}" \
	"${LOSS_VESSELS[@]}" \
	@affine affine --init "${ROOT}/oct2lsfm/affine/${i}/affine.lta" \
	@@optim none \
	@nonlin svf 1 -2d -a 1 -m 0 -b 50 -l 10 10 --voxel-size 0.1 mm --pad 10 % --fov lsfm \
	@@optim gn --tolerance 1e-8 -n 32 \
	@pyramid --levels 2:9

done
