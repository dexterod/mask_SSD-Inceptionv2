FROM 10.9.0.187/algo_team/cuda10.0-cudnn7.4.2-dev-ubuntu16.04-opencv4.1.1-tensorflow1.13-openvino2020r1-workspace

# 创建默认目录，训练过程中生成的模型文件、日志、图必须保存在这些固定目录下，训练完成后这些文件将被保存
RUN mkdir -p /project/train/src_repo && mkdir -p /project/train/result-graphs && mkdir -p /project/train/log && mkdir -p /project/train/models

# 安装训练环境依赖端软件，请根据实际情况编写自己的代码
COPY . /project/train/src_repo/
RUN python3.6 -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r /project/train/src_repo/requirements.txt

RUN wget http://10.9.0.103:8888/group1/M00/00/02/CgkAZ15ibP2EQwGkAAAAAPNBqdc5432.gz -O /project/train/src_repo/pre-trained-model/ssd_inception_v2_coco_2018_01_28.tar.gz \
    && cd /project/train/src_repo/pre-trained-model/ \
    && tar zxf ssd_inception_v2_coco_2018_01_28.tar.gz -C ./ \
    && mv ssd_inception_v2_coco_2018_01_28/* ./
