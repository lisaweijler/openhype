### train
python run_batch_exp.py --exp_batch_name exp_5scenes --config_template_path configs/multiple_scenes/config_template_train.yaml --eval_config_template_path configs/multiple_scenes/config_template_test.yaml --scene_id 45b0dac5e3 --output_dpath openhype_output/scannetpp --n_repeats 5
python run_batch_exp.py --exp_batch_name exp_5scenes --config_template_path configs/multiple_scenes/config_template_train.yaml --eval_config_template_path configs/multiple_scenes/config_template_test.yaml --scene_id 5ee7c22ba0 --output_dpath openhype_output/scannetpp --n_repeats 5
python run_batch_exp.py --exp_batch_name exp_5scenes --config_template_path configs/multiple_scenes/config_template_train.yaml --eval_config_template_path configs/multiple_scenes/config_template_test.yaml --scene_id 7b6477cb95 --output_dpath openhype_output/scannetpp --n_repeats 5
python run_batch_exp.py --exp_batch_name exp_5scenes --config_template_path configs/multiple_scenes/config_template_train.yaml --eval_config_template_path configs/multiple_scenes/config_template_test.yaml --scene_id 25f3b7a318 --output_dpath openhype_output/scannetpp --n_repeats 5
python run_batch_exp.py --exp_batch_name exp_5scenes --config_template_path configs/multiple_scenes/config_template_train.yaml --eval_config_template_path configs/multiple_scenes/config_template_test.yaml --scene_id 578511c8a9 --output_dpath openhype_output/scannetpp --n_repeats 5

### test
python eval.py --config_path openhype_output/scannetpp/45b0dac5e3/configs/exp_5scenes/eval_45b0dac5e3.yaml --evaluator scannetpp_evaluator
python eval.py --config_path openhype_output/scannetpp/7b6477cb95/configs/exp_5scenes/eval_7b6477cb95.yaml --evaluator scannetpp_evaluator
python eval.py --config_path openhype_output/scannetpp/25f3b7a318/configs/exp_5scenes/eval_25f3b7a318.yaml --evaluator scannetpp_evaluator
python eval.py --config_path openhype_output/scannetpp/578511c8a9/configs/exp_5scenes/eval_578511c8a9.yaml --evaluator scannetpp_evaluator
python eval.py --config_path openhype_output/scannetpp/5ee7c22ba0/configs/exp_5scenes/eval_5ee7c22ba0.yaml --evaluator scannetpp_evaluator

### aggregate results over runs and all scenes
python create_eval_stats.py --exp_batch_name exp_5scenes --input_dpath openhype_output/scannetpp --output_dpath openhype_output/scannetpp/ --scene_id_list 45b0dac5e3 5ee7c22ba0 7b6477cb95 25f3b7a318 578511c8a9



