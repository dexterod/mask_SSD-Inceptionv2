#!/bin/bash

project_root_dir=/project/train/src_repo
log_file=/project/train/log/log.txt

pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r /project/train/src_repo/requirements.txt \
&& echo "Preparing..." \
&& cd ${project_root_dir}/tf_models/research/ && protoc object_detection/protos/*.proto --python_out=. \
&& echo "Converting dataset..." \
&& python3 -u ${project_root_dir}/convert_dataset.py | tee -a ${log_file} \
&& echo "Start training..." \
&& cd ${project_root_dir} && python3 -u train.py --logtostderr --train_dir=training/ --pipeline_config_path=pre-trained-model/ssd_inception_v2_coco.config 2>&1 | tee -a ${log_file} \
&& echo "Done" \
&& echo "Exporting and saving models to /project/train/models..." \
&& python3 -u ${project_root_dir}/export_models.py | tee -a ${log_file} \
&& echo "Saving plots..." \
&& python3 -u ${project_root_dir}/save_plots.py | tee -a ${log_file}