conda activate glue

CUDA_VISIBLE_DEVICES=$cuda_device python glue_models/run_glue.py \
--model_name_or_path bert-base-uncased \
--train_file data/$data/train.json \
--validation_file data/$data/val.json \
--test_file data/$data/test.json \
--do_train \
--do_eval \
--do_predict \
--num_labels -1 \
--max_seq_length 512 \
--per_device_train_batch_size 4 \
--learning_rate 1e-5 \
--num_train_epochs 40 \
--output_dir glue_models/$data/eval-x-base \
--ignore_data_skip \
--ignore_mismatched_sizes \
--overwrite_output_dir \
--evaluation_strategy steps \
--eval_steps 500 \
--save_total_limit 5 \
--metric_for_best_model eval_loss \
--load_best_model_at_end \
--use_bernoulli_trainer \
--seed $random_seed

conda deactivate