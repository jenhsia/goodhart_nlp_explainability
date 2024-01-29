source $data_dir/$dataset/case_detector_num_labels.txt

PYTHONPATH=./:$PYTHONPATH python case_detector/get_predictions.py --data_dir data --case_dir case_detector --model_dir glue_models --dataset $dataset --explainer $explainer --num_cases $num_cases 

PYTHONPATH=./:$PYTHONPATH python rationale_benchmark/metrics.py --split test --data_dir $data_dir/$dataset --results case_detector/$dataset/$explainer/test_decoded.jsonl --score_file case_detector/$dataset/$explainer/test_scores.json

PYTHONPATH=./:$PYTHONPATH python rationale_benchmark/metrics.py --split test --data_dir $data_dir/$dataset --results case_detector/$dataset/$explainer/"$num_cases"_way/wrapped_test_decoded.jsonl --score_file case_detector/$dataset/$explainer/"$num_cases"_way/wrapped_test_scores.json



