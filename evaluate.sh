#!/bin/bash

exp_id="PACRR_simdim-800_epochs-10_nsamples-2048_maxqlen-16_binmat-False_numneg-6_batch-32_ud-False_distill-firstk_winlen-3_nfilter-32_kmaxpool-3_combine-16_qproximity-0_context-False_shuffle-False_xfilters-_cascade-_nomfeat-1_featnames-sims"


base_dir="/media/vandermonde/HDD/PycharmProjects/copacrr/out"
folds=`ls "$base_dir/*/"`
epochs=(0.run 1.run 2.run 3.run 4.run 5.run 6.run 7.run 8.run 9.run)

for fold in $folds
do
	valid_test=`ls "$base_dir/$fold/pacrrpub/predict_per_epoch/*/"`
	
	eval=${valid_test[0]}
	test=${valid_test[1]}
	
	eval_dir="$base_dir/$fold/pacrrpub/predict_per_epoch/$eval/$exp_id/"
	test_dir="$base_dir/$fold/pacrrpub/predict_per_epoch/$test/$exp_id/"
	
	best_res=-1
	for epoch in ${epochs[@]}
	do
		res=`./trec_eval/trec_eval ./data/qrels.adhoc.6y "$eval_dir/$epoch" | grep  "^map" | grep all |tr '\t' ' ' | tr -s ' ' | cut -d ' ' -f3`
		if [ $res>$best_res ]; then
			best_res=$res
			best_epoch=$epoch
		fi	
	done
	
	test_res=`./trec_eval/trec_eval ./data/qrels.adhoc.6y "$test_dir/$best_epoch" | grep  "^map" | grep all |tr '\t' ' ' | tr -s ' ' | cut -d ' ' -f3`
	
	echo "$fold  eval: $eval best_ecpoch: $best_epoch  best_res:$best_res, test: $test   test_res=$test_res"
		
	

	eval=${valid_test[1]}
        test=${valid_test[0]}

        eval_dir="$base_dir/$fold/pacrrpub/predict_per_epoch/$eval/$exp_id/"
        test_dir="$base_dir/$fold/pacrrpub/predict_per_epoch/$test/$exp_id/"

        best_res=-1
        for epoch in ${epochs[@]}
        do
                res=`./trec_eval/trec_eval ./data/qrels.adhoc.6y "$eval_dir/$epoch" | grep  "^map" | grep all |tr '\t' ' ' | tr -s ' ' | cut -d ' ' -f3`
                if [ $res>$best_res ]; then
                        best_res=$res
                        best_epoch=$epoch
                fi     
        done

        test_res=`./trec_eval/trec_eval ./data/qrels.adhoc.6y "$test_dir/$best_epoch" | grep  "^map" | grep all |tr '\t' ' ' | tr -s ' ' | cut -d ' ' -f3`

        echo "$fold  eval: $eval best_ecpoch: $best_epoch  best_res:$best_res, test: $test   test_res=$test_res"

done	
