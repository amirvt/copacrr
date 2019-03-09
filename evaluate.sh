#!/bin/bash

base_dir="/media/vandermonde/HDD/PycharmProjects/copacrr/out"
folds=`ls "$base_dir/*/"`

for fold in $folds
do
	valid_test=`ls "$base_dir/$fold/pacrrpub/predict_per_epoch/*/"`
	
	train=${valid_test[0]}
	test=${valid_test[1]}
	
	train_files=`ls 

