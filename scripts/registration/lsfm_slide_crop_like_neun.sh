ROOT="."
ROOT+="/lsfm"

for i in {08..23}
do

nitorch crop $ROOT/align_channels/*${i}*.nii* --like $ROOT/slide/neun.${i}.slidecrop.nii.gz -o $ROOT/slide/{base}.slidecrop.nii.gz

done
