# 简易版EV_SDK

## 说明
### EV_SDK的目标
开发者专注于算法开发及优化，最小化业务层编码，即可快速部署到生产环境，共同打造商用级高质量算法。
### 简易版EV_SDK

简易版EV_SDK用于开发者算法模型完成后，降低其在封装过程中的难度。与标准版EV_SDK相比，简易版SDK移除了授权和模型加密两大模块，并且需要实现的接口减小为3个。

### 开发者需要做什么
1. 模型的训练和调优；
2. 实现部分`ji.h`中定义的接口；
3. 实现约定的输入输出；

## 目录

### 代码目录结构

```
ev_sdk
|-- 3rd             # 第三方源码或库目录，发布时请删除
|   |-- wkt_parser          # 针对使用WKT格式编写的字符串的解析器
|   |-- cJSON               # c版json库，简单易用
|   `-- darknet             # 示例项目依赖的库
|-- CMakeLists.txt          # 本项目的cmake构建文件
|-- README.md       # 本说明文件
|-- model           # 模型数据存放文件夹
|-- config          # 程序配置目录
|   |-- README.md   # algo_config.json文件各个参数的说明和配置方法
|   `-- algo_config.json    # 程序配置文件
|-- include         # 库头文件目录
|   `-- ji.h        # libji.so的头文件，理论上仅有唯一一个头文件
|-- lib             # 本项目编译并安装之后，默认会将依赖的库放在该目录，包括libji.so
|-- src             # 实现ji.cpp的代码
`-- test            # 针对ji.h中所定义接口的测试代码，请勿修改！！！
```
## 使用示例
作为示例，我们提供了一个使用`darknet`实现的图像检测器，并将其使用简易版EV_SDK进行封装，需要实现的业务逻辑是：当检测到**狗**时，需要返回相关的报警信息和检测到的目标。使用如下步骤尝试编译和测试该项目：

#### 编译

编译和安装`libji.so`：

```shell
mkdir -p /usr/local/ev_sdk/build
cd /usr/local/ev_sdk/build
cmake ..
make install
```

#### 测试示例程序和接口规范

执行完成之后，`/usr/local/ev_sdk/lib`下将生成`libji.so`和相关的依赖库，以及`/usr/local/ev_sdk/bin/`下的测试程序`test-ji-api`。

2. 使用`test-ji-api`测试`ji_calc_frame`接口，测试添加了一个`ROI`参数（ROI参数使用标准的WKT格式表示，并且坐标为归一化值），以下示例传入一个多边形的ROI参数

   ```shell
   /usr/local/ev_sdk/bin/test-ji-api -f ji_calc_frame -i /usr/local/ev_sdk/data/dog.jpg -o /tmp/output.jpg -a "{\"roi\":[\"POLYGON((0.21666666666666667 0.255,0.6924242424242424 0.1375,0.8833333333333333 0.72,0.4106060606060606 0.965,0.048484848484848485 0.82,0.2196969696969697 0.2575))\"]}"
   ```

   输出内容样例：

   ```shell
    code: 0
    json: {
    "alert_flag":   1,
    "dogs": [{
            "xmin": 129,
            "ymin": 186,
            "xmax": 369,
            "ymax": 516,
            "confidence":   0.566474,
            "name": "dog"
        }]
    }
   ```

## 使用`EV_SDK`快速封装算法

假设项目需要检测输入图像中是否有**狗**，如果检测到**狗**，就需要输出报警信息，以下示例开发算法与使用`EV_SDK`进行封装的流程

#### 实现自己的模型

假设我们使用`darknet`开发了针对**狗**的检测算法，程序需要在检测到狗时输出报警信息。

#### 实现`ji.h`中的接口

`ji.h`中定义了所有`EV_SDK`规范的接口，详细的接口定义和实现示例，请参考接口文件[ji.h](include/ji.h)和示例代码[ji.cpp](src/ji.cpp)。

简易版EV_SDK需要实现的接口包括：

- `ji_create_predictor`：创建算法实例
- `ji_destroy_predictor`：释放算法实例和其他资源
- `ji_calc_frame`：使用算法实例处理数据，并填充结果

将代码编译成`libji.so`

```shell
mkdir -p /usr/local/ev_sdk/build
cd /usr/local/ev_sdk/build
cmake ..
make install
```

编译完成后，将在`/usr/local/ev_sdk/lib`下生成`libji.so`和其他依赖的库。

#### 测试接口功能

测试`libji.so`的授权功能是否正常工作以及`ji.h`的接口规范

2. 检查授权功能和`ji.h`的接口规范性

   `EV_SDK`代码中提供了测试所有接口的测试程序，**编译并安装**`libji.so`之后，会在`/usr/local/ev_sdk/bin`下生成`test-ji-api`可执行文件，`test-ji-api`用于测试`ji.h`的接口实现是否正常，例如，测试`ji_calc_frame`接口：

   ```shell
   /usr/local/ev_sdk/bin/test-ji-api -f ji_calc_frame \
   -i /usr/local/ev_sdk/data/dog.jpg \
   -o /tmp/output.jpg \
   -a "{\"roi\":[\"POLYGON((0.21666666666666667 0.255,0.6924242424242424 0.1375,0.8833333333333333 0.72,0.4106060606060606 0.965,0.048484848484848485 0.82,0.2196969696969697 0.2575))\"]}"
   ```
   

## 哪些内容必须完成才能通过简易版EV_SDK测试？
#### 接口功能要求
1. `ji_calc_frame` ，用于实时视频流分析；
2. `ji_create_predictor`，`ji_destroy_predictor`，算法实例需要正常创建和释放；

#### 规范要求

规范测试大部分内容依赖于内置的`/usr/local/ev_sdk/test`下面的代码，这个测试程序会链接`/usr/local/ev_sdk/lib/libji.so`库，`EV_SDK`封装完成提交后，极市方会使用`test-ji-api`程序测试`ji.h`中的所有接口。测试程序与`EV_SDK`的实现没有关系，所以**请勿修改`/usr/local/ev_sdk/test`目录下的代码！！！**

1. 接口功能要求
  
   - 确定`test-ji-api`能够正常编译，并且将`test-ji-api`移动到任意目录，都能够正常运行（确保动态库链接正确）；
   
   - 在提交算法之前，请自行通过`/usr/local/ev_sdk/bin/test-ji-api`测试接口功能是否正常；
   
   - 对于接口中传入的参数`args`（如，`ji_calc_frame(void *, const JI_CV_FRAME *, const char *args, JI_CV_FRAME *, JI_EVENT *)`中中`args`），根据项目需求，算法实现需要支持`args`实际传入的参数。
   
     例如，如果项目需要支持在`args`中传入`roi`参数，使得算法只对`roi`区域进行分析，那么**算法内部必须实现只针对`roi`区域进行分析的功能**；
   
2. 业务逻辑要求

   针对需要报警的需求，算法必须按照以下规范输出结果：

   - 当需要输出报警信息时
     - 报警时输出：`JI_EVENT.code=JISDK_CODE_ALARM`，`JI_EVENT.json`内部填充`"alert_flag"=1`；
     - 未报警时输出：`JI_EVENT.code=JISDK_CODE_NORMAL`，`JI_EVENT.json`内部填充`"alert_flag"=0`；
   - 处理失败的接口返回`JI_EVENT.code=JISDK_CODE_FAILED`

   - 算法输出的`json`数据格式必须与**项目需求**保持一致；

3. 文件结构规范要求

   * 与模型相关的文件必须存放在`/usr/local/ev_sdk/model`目录下，例如权重文件、目标检测通常需要的名称文件`coco.names`等。
   * 最终编译生成的`libji.so`必须自行链接必要的库，`test-ji-api`不会链接除`/usr/local/ev_sdk/lib/libji.so`以外的算法依赖库；


## FAQ

### 如何使用接口中的`args`？

通常，在实际项目中，外部需要将多种参数（例如`ROI`）传入到算法，使得算法可以根据这些参数来改变处理逻辑。`EV_SDK`接口（如`int ji_calc_frame(void *, const JI_CV_FRAME *, const char *args, JI_CV_FRAME *, JI_EVENT *)`中的`args`参数通常由开发者自行定义和解析，但只能使用[JSON](https://www.json.cn/wiki.html)格式。格式样例：

```shell
{
    "cid": "1000",
    "roi": [
        "POLYGON((0.0480.357,0.1660.0725,0.3930.0075,0.3920.202,0.2420.375))",
        "POLYGON((0.5130.232,0.790.1075,0.9280.102,0.9530.64,0.7590.89,0.510.245))",
        "POLYGON((0.1150.497,0.5920.82,0.5810.917,0.140.932))"
    ]
}
```

例如当算法支持输入`ROI`参数时，那么开发者需要在`EV_SDK`的接口实现中解析上面示例中`roi`这一值，提取其中的`ROI`参数，并使用`WKTParser`对其进行解析，应用到自己的算法逻辑中。
### 为什么不能且不需要修改`/usr/local/ev_sdk/test`下的代码？

1. `/usr/local/ev_sdk/test`下的代码是用于测试`ji.h`接口在`libji.so`中是否被正确实现，这一测试程序与`EV_SDK`的实现无关，且是极市方的测试标准，不能变动；
2. 编译后`test-ji-api`程序只会依赖`libji.so`，如果`test-ji-api`无法正常运行，很可能是`libji.so`没有正确链接依赖库；

### 为什么运行`test-ji-api`时，会提示找不到链接库？

由于`test-ji-api`对于算法而言，只链接了`/usr/local/ev_sdk/lib/libji.so`库，如果`test-ji-api`运行过程中，找不到某些库，那么很可能是`libji.so`依赖的某些库找不到了。此时

1. 可以使用`ldd /usr/local/ev_sdk/lib/libji.so`检查是否所有链接库都可以找到；
2. 请按照规范将系统动态库搜索路径以外的库放在`/usr/local/ev_sdk/lib`目录下；
3. 检查是否正确设置动态库的`RPATH`、`RUNPATH`
