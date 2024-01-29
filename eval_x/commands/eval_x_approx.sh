PYTHONPATH=./:$PYTHONPATH python eval_x/create_eval_x_approx_data.py --dataset $dataset --data_dir data --model_path glue_models --eval_model $eval_model --approx_model $approx_model

conda activate glue

CUDA_VISIBLE_DEVICES=$cuda_device python glue_models/run_glue.py \
--model_name_or_path glue_models/$dataset/eval-x-approx \
--train_file $data_dir/$dataset/train.json \
--validation_file $data_dir/$dataset/val.json \
--test_file $data_dir/$dataset/test.json \
--do_train \
--do_eval \
--do_predict \
--num_labels -1 \
--max_seq_length 512 \
--per_device_train_batch_size 8 \
--learning_rate 1e-5 \
--num_train_epochs 10 \
--output_dir glue_models/$dataset/eval-x-approx \
--ignore_data_skip \
--ignore_mismatched_sizes \
--overwrite_output_dir \
--evaluation_strategy steps \
--eval_steps 500 \
--save_total_limit 5 \
--load_best_model_at_end \
--use_bernoulli_trainer \
--seed 42

conda deactivate

PYTHONPATH=./:$PYTHONPATH python eval_x/get_eval_x_results.py --data_dir $data_dir --model_path glue_models --dataset $dataset --explainer $explainer --eval_model $eval_model