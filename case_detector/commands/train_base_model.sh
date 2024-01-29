conda activate glue

python glue_models/run_glue.py \
--model_name_or_path bert-base-uncased \
--train_file $data_dir/$dataset/train.json \
--validation_file $data_dir/$dataset/val.json \
--test_file $data_dir/$dataset/test.json \
--do_train \
--do_eval \
--do_predict \
--num_labels -1 \
--max_seq_length 512 \
--per_device_train_batch_size 8 \
--learning_rate 2e-5 \
--num_train_epochs 10 \
--output_dir glue_models/$dataset/original_input \
--ignore_data_skip \
--ignore_mismatched_sizes \
--overwrite_output_dir \
--evaluation_strategy steps \
--eval_steps 5000 \
--save_steps 5000 \
--save_total_limit 5 \
--load_best_model_at_end

conda deactivate