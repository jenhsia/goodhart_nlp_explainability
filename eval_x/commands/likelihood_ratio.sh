ratio_source=true

PYTHONPATH=./:$PYTHONPATH python eval_x/create_likelihood_ratio_data.py --dataset $dataset --data_dir data --eval_model_dir glue_models --eval_model_name $eval_model_name --ratio_source $ratio_source --truncate_ratio 

for file in data/$data/$explainer/*_tokens
do 
    echo $file
    sub_dir=$(basename $file)
    PYTHONPATH=./:$PYTHONPATH python eval_x/get_eval_x_results.py --data_dir data --model_path glue_models --dataset $dataset --explainer $explainer/$sub_dir --eval_model eval-x-$eval_seed
done

