source $data_dir/$dataset/case_detector_num_labels.txt

export num_cases=$num_cases

conda activate glue

CUDA_VISIBLE_DEVICES=$cuda_devices python glue_models/run_glue.py \
--model_name_or_path bert-base-uncased \
--train_file data/$dataset/$explainer/train_"$num_cases"_way_case_detector.json \
--validation_file data/$data/$explainer/val_"$num_cases"_way_case_detector.json \
--test_file data/$dataset/$explainer/test_"$num_cases"_way_case_detector.json \
--do_predict \
--num_labels $num_cases \
--max_seq_length 512 \
--per_device_train_batch_size 8 \
--learning_rate 2e-5 \
--num_train_epochs 3 \
--output_dir case_detector/$dataset/$explainer/"$num_cases"_way \
--ignore_data_skip \
--ignore_mismatched_sizes \
--overwrite_output_dir \
--evaluation_strategy steps \
--eval_steps 5000 \
--save_ste 5000 \
--metric_for_best_model loss \
--save_total_limit 5 \
--load_best_model_at_end \
--run_name case_detector/$dataset/$explainer/"$num_cases"_way

conda deactivate