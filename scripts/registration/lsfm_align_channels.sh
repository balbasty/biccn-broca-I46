
ROOT="."
ROOT+="/lsfm"
COLORS=(calb sst)
WAVELENGTHS=(488 561)

for c in 0
do

COLOR="${COLORS[$c]}"
WAVELENGTH="${WAVELENGTHS[$c]}"

for i in {08..23}
do

MOV="$ROOT/init_oct/${COLOR}_${WAVELENGTH}/down_9.9um/${COLOR}.${i}.nii.gz"
FIX="$ROOT/init_oct/neun_638/down_9.9um/neun.${i}.nii.gz"
OUT="$ROOT/align_channels/${COLOR}2neun/${i}"

nitorch register2 \
	-o "$OUT" --gpu -v 4  \
	@loss lncc --patch 5 % \
		@@fix "$FIX" \
		@@mov "$MOV" \
	@affine affine --progressive \
	@@optim gn --tolerance 1e-8 -n 256 \
	@pyramid --levels 2:6

done
done
