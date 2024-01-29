1. create data for base model
python case_detector/create_base_data.py --dataset $dataset --data_dir $data_dir

2. train base model
train_base_model.sh

3. create training data for case detector
CUDA_VISIBLE_DEVICES=$cuda_device PYTHONPATH=./:$PYTHONPATH python case_detector/create_cased_data.py --dataset $dataset --data_dir $data_dir --explainer $explainer --base_model_path glue_models 

4. train case detector
train_case_detector.sh

5. run pipelined model on data
evaluate_pipeline.sh

6. get downstream task and explainability scores
get_scores.sh 

