ratio_source=true
PYTHONPATH=./:$PYTHONPATH python eval_x/create_naive_bayes_data.py --data $data --data_dir data --model_path glue_models --eval_model $eval_model --ratio_source $ratio_source --truncate_ratio 

for file in data/$data/$explainer/*_tokens
do 
    echo $file
    sub_dir=$(basename $file)
    PYTHONPATH=./:$PYTHONPATH python eval_x/get_eval_x_results.py --data_dir data --model_path glue_models --data $data --explainer $explainer/$sub_dir --eval_model eval-x-$eval_seed
done

