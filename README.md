# mask_SSD-Inceptionv2
## Introduction
这是我前段时间参加的一个口罩检测比赛使用的代码。使用的是谷歌公司推出的object detection API中的SSD-Inceptionv2模型，现记录于此。
注：这次比赛是在云服务器上跑的，其中Dockerfile里的内容是用于构建镜像的。如果在本地服务器上跑，可以把其中的内容在Terminal或者shell文件里重新写一下。
关于这次比赛数据集的情况、完成过程、评价标准以及遇到的一些问题都记录在了个人博客里，仅供参考。

[口罩、安全帽识别比赛踩坑记（一） 经验漫谈及随想](https://blog.csdn.net/dexterod/article/details/105370351)

[口罩、安全帽识别比赛踩坑记（二） 比赛流程及 SSD / YOLO V3 两版本实现](https://blog.csdn.net/dexterod/article/details/105438526)

[SSD 论文详解](https://blog.csdn.net/dexterod/article/details/104825742)

## Requirements
 - Ubuntu 16.04
 - python 3.6.8
 - Tensorflow-gpu 1.13.2
 - CUDA 10.0 / cudnn 7.4.2 
 - OpenCV 4.2
 - pandas
 - matplotlib
 - pillow
 - seaborn
 - tensorboard
 
如果要用openvino加速，则还需要去[OpenVINO官方文档](https://docs.openvinotoolkit.org.html)上下载相应的包。
 - OpenVINO2020R1

## Download Model
这里先下载本模型的预训练模型[SSD-Inceptionv2](https://pan.baidu.com/s/1v4mndRbgNrb5hkZM5ONqMA)，提取码：9nw4<br>
下载后放在 你的项目地址/src_repo/pre-trained-model 里解压就好。

当然你也可以直接在Terminal里wget [http://10.9.0.103:8888/group1/M00/00/02/CgkAZ15ibP2EQwGkAAAAAPNBqdc5432.gz](http://10.9.0.103:8888/group1/M00/00/02/CgkAZ15ibP2EQwGkAAAAAPNBqdc5432.gz)<br>
但最好还是在外面用自己的下载软件下载，不然你就会体验到什么叫绝望。

## 程序说明
因为是在云服务器上跑的比赛，听的是赛组委的宣，所以跟本地跑的程序有几点不同，主要是Dockerfile镜像构建部分可以不用管它，还有测试程序没有写，这部分是在后期封装成SDK以后预留一个测试接口，然后使用这个测试接口直接与他们内网的测试数据集关联，进行测试后直接输出结果的。

### 1.整个程序结构：

├── project/<br>
│&nbsp;&nbsp;&nbsp;├── ev_sdk&nbsp;&nbsp;&nbsp;# ev_sdk封装代码仓库，软链接到/usr/local/ev_sdk<br>
│&nbsp;&nbsp;&nbsp;├── train&nbsp;&nbsp;&nbsp;# 训练相关的代码、模型、日志文件、图等<br>
│&nbsp;&nbsp;&nbsp;│&nbsp;&nbsp;&nbsp;├──models&nbsp;&nbsp;&nbsp;# 模型和openvino文件存放地址<br>
│&nbsp;&nbsp;&nbsp;│&nbsp;&nbsp;&nbsp;├──src_repo&nbsp;&nbsp;&nbsp;# 训练代码仓库<br>
│&nbsp;&nbsp;&nbsp;│&nbsp;&nbsp;&nbsp;├──result-graphs&nbsp;&nbsp;&nbsp;# 训练相关图<br>
│&nbsp;&nbsp;&nbsp;│&nbsp;&nbsp;&nbsp;├──log&nbsp;&nbsp;&nbsp;# 日志文件<br>

### 2.训练相关程序结构
其中/project/train/src_repo存放的是数据处理、训练、生成openvino模型等代码。程序结构如下：

├── project/train/src_repo/<br>
│&nbsp;&nbsp;&nbsp;├── openvino_config&nbsp;&nbsp;&nbsp;# openvino加速推理时需要的config<br>
│&nbsp;&nbsp;&nbsp;├── pre-trained-model&nbsp;&nbsp;&nbsp;# 预训练模型存储的地方，包括config<br>
│&nbsp;&nbsp;&nbsp;├── tf_models&nbsp;&nbsp;&nbsp;# ODA模型<br>
│&nbsp;&nbsp;&nbsp;├── convert_dataset.py&nbsp;&nbsp;&nbsp;# 数据预处理模块<br>
│&nbsp;&nbsp;&nbsp;├── Dockerfile&nbsp;&nbsp;&nbsp;# 构建镜像部分，没有用云的话就可以不管<br>
│&nbsp;&nbsp;&nbsp;├── export_models.py&nbsp;&nbsp;&nbsp;# .ckpt转换为.pb，最后.pb再转换为openvino模型<br>
│&nbsp;&nbsp;&nbsp;├── global_config.py<br>
│&nbsp;&nbsp;&nbsp;├── requirements.txt&nbsp;&nbsp;&nbsp;# 程序运行所需的包<br>
│&nbsp;&nbsp;&nbsp;├── save_plots.py&nbsp;&nbsp;&nbsp;# 训练过程可视化<br>
│&nbsp;&nbsp;&nbsp;├── start_train.sh&nbsp;&nbsp;&nbsp;# 训练命令集成<br>
│&nbsp;&nbsp;&nbsp;├── train.py<br>

### 3.SDK封装相关程序结构
/project/ev_sdk存放的是EV_SDK封装代码，该目录在比赛中是/usr/local/ev_sdk的软链接，内部的程序结构如下。当然，我们日常用，不用考虑落地的程序不需要理会ev_sdk里面的内容。训练中生成的openvino模型.xml和.bin文件会存储在/project/train/models里，与/project/ev_sdk/model是会同步的，这里需要我们自己手动mv一哈~ 

具体的测试和各部分的功用参考/project/ev_sdk/README.md。

/usr/local/ev_sdk/<br>
├── CMakeLists.txt    # cmake构建文件<br>
├── Dockerfile            # 构建镜像的文件<br>
├── include<br>
│&nbsp;&nbsp;&nbsp;├── ji.h    # 接口头文件<br>
│&nbsp;&nbsp;&nbsp;└── SampleDetector.hpp<br>
├── model<br>
├── README.md<br>
├── src<br>
│&nbsp;&nbsp;&nbsp;├── ji.cpp    # ji.h的接口实现代码<br>
│&nbsp;&nbsp;&nbsp;└── SampleDetector.cpp    # 检测算法加载与推理的实现代码<br>
└── test<br>

## 训练自己的数据集
### 1.更改项目存放地址
本程序中默认的项目存放地址为/project/train/src_repo，如要更改为自己存放项目的地址，需要改动以下几处地方：
 - start_train.sh第3/4/6行
 - global_config.py第4行
 - ssd_inception_v2_coco.config第171/173/185/187行
 - ssd_inception_v2_coco.config第152行（预训练模型存储地址）
 - export_models.py第17行
 - save_plots.py第52行
 
### 2.建立数据集
我们使用的数据集标注为VOC2007格式，可使用labelImg进行xml标注。采用bounding box四点标注的方式。标注好后的图片和xml都存储在同一个文件夹里，

本程序中默认的该文件夹地址为/home/data/12。如需修改为自己的地址，可以更改convert_dataset.py第21行；

还需替换自己的label_names，更改位置在convert_dataset.py第28/212行；ssd_inception_v2_coco.config第9行。

### 3.使用openvino套件加速推理
如果要使用，需要去[OpenVINO官网](https://docs.openvinotoolkit.org.html)上下载相应的套件，本程序默认的安装地址是/opt/intel/openvino_2020.1.023/，如要更改地址，需更改export_models.py第14/55行；

如果不使用openvino加速，请注释掉export_models.py第14/15,53~63行。

### 3.训练
训练时，我们先要在ssd_inception_v2_coco.config中设定训练的一些相关参数，其中第142、158行是关于训练次数的，需要更改。

设定好后，我们在Terminal中输入指令：bash /project/train/src_repo/start_train.sh (其中/project/train/src_repo/需更改为你的项目地址）即可

### 4.SDK封装
如果你最终需要封装成SDK，可以更改:
 - /project/ev_sdk/src/SampleDetector.cpp第39~42行
 - /project/ev_sdk/src/ji.cpp第100行
 - 相应的测试程序
