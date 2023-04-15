#!/bin/bash


ROOT="."
VERBOSE=4

for i in {08..23}; do

OCT="${ROOT}/oct/slices/unstack/oct.${i}.masked.nii.gz"
OCTVSL="${ROOT}/oct/slices/unstack/vessels.${i}.nii.gz"

LSFM="${ROOT}/lsfm/slide_center_oct/neun.${i}.nii.gz"
LSFMVSL="${ROOT}/lsfm/slide_center_oct/vessels.${i}.float.nii.gz"
LSFMDST="${ROOT}/lsfm/slide_center_oct/vessels.${i}.distance.squeeze.nii.gz"

run_register ()
{
PATCH=$1
INIT=$2

LOSS_IMAGE=(\
@loss lcc --patch $PATCH % --slicewise -1 \
	@@fix $LSFM -z 64 64 0 -b replicate \
	@@mov $OCT -b zero \
)
if [ -f "$LSFMDST" ]; then
#LOSS_VESSELS=(
#@loss ndot 1 \
#	@@fix $LSFMDST -z 64 64 0 -b replicate --rescale 0 \
#	@@mov $OCTVSL -b zero --rescale 0 --fwhm 1 \
#)
LOSS_VESSELS=(
@loss dice 1 \
	@@fix $LSFMVSL -z 64 64 0 -b replicate --rescale 0 \
	@@mov $OCTVSL -b zero --rescale 0 --fwhm 1 \
)
else
LOSS_VESSELS=()
fi

if [ "$INIT" ]; then
INIT=(--init "${ROOT}/oct2lsfm/affine/${i}/affine.lta")
else
INIT=(--progessive)
fi

nitorch register2 \
	-v $VERBOSE --gpu -o "${ROOT}/oct2lsfm/affine/${i}" \
	"${LOSS_IMAGE[@]}" \
	"${LOSS_VESSELS[@]}" \
	@affine affine -2d ${INIT[@]} \
	@@optim gn --tolerance 1e-8 -n 1024 \
	@pyramid --levels 2:9

}

run_register 20
run_register 10 init
run_register 5 init


done
