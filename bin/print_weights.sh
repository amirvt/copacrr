#!/usr/bin/env bash
CUR_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

source $CUR_DIR/set_env.sh

START=$(date +%s.%N)

expname=pacrrpub
#train_years=fold01_02_03
#test_year=fold_04
numneg=6
batch=32
winlen=3
kmaxpool=3
binmat=False
context=False
combine=16
iterations=10
shuffle=False
parentdir="/media/vandermonde/HDD/PycharmProjects/copacrr/out"
outdir="$parentdir"
nfilter=32


python3 -m print_weights with\
	expname=$expname \
	train_years=$train_years \
	test_year=$test_year \
	numneg=$numneg \
	batch=$batch \
	winlen=$winlen \
	kmaxpool=$kmaxpool \
	binmat=$binmat \
	context=$context \
	combine=$combine \
	shuffle=$shuffle \
	parentdir="$parentdir" \
	epochs=$iterations \
	outdir="$outdir"\
	nomfeat=2 \
	featnames='sims*idfs' \
	nfilter=$nfilter

END=$(date +%s.%N)
DIFF=$(echo "$END - $START" | bc)
echo $id finished within $DIFF


