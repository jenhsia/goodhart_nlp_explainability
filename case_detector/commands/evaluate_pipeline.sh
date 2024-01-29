source $data_dir/$dataset/case_detector_num_labels.txt


CUDA_VISIBLE_DEVICES=0,1,2,3 python glue_models/run_glue.py \
--model_name_or_path glue_models/$dataset/original_input \
--train_file $data_dir/$dataset/$explainer/train_"$num_cases"_way_case_detector.json \
--validation_file $data_dir/$dataset/$explainer/val_"$num_cases"_way_case_detector.json \
--test_file $data_dir/$dataset/$explainer/test_"$num_cases"_way_case_detector.json \
--num_labels $original_num_labels \
--do_predict \
--max_seq_length 512 \
--per_device_eval_batch_size 8 \
--output_dir glue_models/$dataset/cased_input/$explainer \
--ignore_data_skip \
--ignore_mismatched_sizes \
--overwrite_output_dir \
--overwrite_cache \
--run_name $dataset/cased_input/$explainer

conda deactivate
