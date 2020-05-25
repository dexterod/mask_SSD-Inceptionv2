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
这里先下载本模型的预训练模型[SSD-Inceptionv2](https://pan.baidu.com/s/1v4mndRbgNrb5hkZM5ONqMA)，提取码：9nw4
下载后放在 ./src_repo/pre-trained-model 里解压就好。

当然你也可以直接在terminal里wget [http://10.9.0.103:8888/group1/M00/00/02/CgkAZ15ibP2EQwGkAAAAAPNBqdc5432.gz](http://10.9.0.103:8888/group1/M00/00/02/CgkAZ15ibP2EQwGkAAAAAPNBqdc5432.gz)
但最好还是在外面用自己的下载软件下载，不然你就会体验到什么叫绝望。

## 程序说明
因为是在云服务器上跑的比赛，听的是赛组委的宣，所以跟本地跑的程序有几点不同，主要是Dockerfile镜像构建部分可以不用管它，还有测试程序没有写，这部分是在后期封装成
SDK以后预留一个测试接口，然后使用这个测试接口直接与他们内网的测试数据集关联，进行测试后直接输出结果的。

下面来看程序结构：

├── project/<br>
│&nbsp;&nbsp;&nbsp;├── ev_sdk&nbsp;&nbsp;&nbsp;# ev_sdk封装代码仓库，软链接到/usr/local/ev_sdk<br>
│&nbsp;&nbsp;&nbsp;├── train&nbsp;&nbsp;&nbsp;# 训练相关的代码、模型、日志文件、图等<br>
│&nbsp;&nbsp;&nbsp;│&nbsp;&nbsp;&nbsp;├──models&nbsp;&nbsp;&nbsp;# 模型和openvino文件存放地址<br>
│&nbsp;&nbsp;&nbsp;│&nbsp;&nbsp;&nbsp;├──src_repo&nbsp;&nbsp;&nbsp;# 训练代码仓库<br>
│&nbsp;&nbsp;&nbsp;│&nbsp;&nbsp;&nbsp;├──result-graphs&nbsp;&nbsp;&nbsp;# 训练相关图<br>
│&nbsp;&nbsp;&nbsp;│&nbsp;&nbsp;&nbsp;├──log&nbsp;&nbsp;&nbsp;# 日志文件<br>

/project/ev_sdk存放EV_SDK封装代码，该目录是/usr/local/ev_sdk的软链接，参与一个新项目并首次进入开发环境时，会在此处初始化一个git仓库，

## 训练自己的数据集
### 数据集
需要变动的地方

测试部分
