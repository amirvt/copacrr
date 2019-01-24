#!/usr/bin/env bash
CUR_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source $CUR_DIR/set_env.sh

START=$(date +%s.%N)

expname=pacrrpub
#train_years=wt09_10_12_13
numneg=6
batch=32
winlen=3
kmaxpool=3 
binmat=False
context=False
combine=16
iterations=10
shuffle=False
#parentdir=~/playground/copacrr/out
outdir=$parentdir

python3 -m train_model with\
	expname=$expname \
	train_years=$train_years \
	numneg=$numneg \
	batch=$batch \
	winlen=$winlen \
	kmaxpool=$kmaxpool \
	binmat=$binmat \
	context=$context \
	combine=$combine \
	shuffle=$shuffle \
	parentdir=$parentdir \
	epochs=$iterations \
	outdir=$outdir \
	ud=False \
	nomfeat=2 \
	feat_names=sims_mis
END=$(date +%s.%N)
DIFF=$(echo "$END - $START" | bc)
echo $id finished within $DIFF


