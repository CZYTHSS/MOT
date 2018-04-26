## 代码运行环境

Python 2.7.6

NumPy 1.13.3

SciPy 0.19.1

argparse 1.1

Opencv 3.3.0

## 多目标追踪方法简介

![WX20171117-194945](/Users/lichen/Desktop/华为多目标追踪/WX20171117-194945.png)

- 方法对每个tracklet维护一个卡尔曼滤波作为单目标追踪器，状态空间为（center x, center y, ratio, height, delta center x, delta center y, delta ratio, delta height），并对每个tracklet维护一个budget，存储此对象在各个时刻的良好appearance特征，为之后度量与detections的特征之间的距离做准备
- 对于多目标追踪过程，首先通过度量tracklets与detections之间的appearance特征的距离，并使用卡尔曼滤波对预测位置状态与实际位置状态距离度量进行过滤，得到tracklets与detections之间的距离矩阵；之后将多目标追踪过程的每一步转化为最小代价分配问题，并使用匈牙利算法进行解决；最后通过每一步的匹配对原来tracklets的状态进行更新，对未匹配上的每一个detection生成新的tracklet
- appearance特征由深度模型训练得到

## 输入

视频数据集：整理为MOTTChanllenge格式；

检测结果：存储到2维numpy.array中，每行表示一个检测结果，格式为（1～10列：MOTChanllenge格式检测结果，10～最后列：检测结果的feature），并将numpy.array持久化保存到本地。

## 输出

MOTTChanllenge标准格式的追踪输出结果。

##代码结构

####`mottracker_app.py`

- 可执行脚本，对`MOTTracker.py`定义的多目标追追踪方法的执行

- 包括将数据转化为MOTTracker.Detection类（检测结果读入），以及对MOTTracker.MOTTracker的调用（多目标追踪执行），以及追踪结果的存储

- 包括模型参数的描述，以及入口

  `sequence_dir`：为MOTChallenge格式的视频数据的存储路径

  `detection_mat`：检测结果矩阵（numpy.array）的存储路径

  `output`：追踪结果的输出路径

  `min_confidence`：阈值，对低于此阈值的检测结果进行过滤，默认值为0.8

  `max_cosine_distance`：appearance association阈值，若tracklet与detection的关联距离大于此阈值（关联效果不佳），则进行过滤，默认值为0.2

  `nn_budget`：每个单目标追踪器存储的appearance特征的最大个数

  `budget_association_threshold`：budget的准入阈值，若tracklet与detection的关联距离小于此阈值（appearance特征匹配较好，视为良好关联），则将appearance存入追踪器的budget中

  `budget_detection_threshold`：budget的准入阈值，confidence大于此阈值的检测（可信的检测）的appearance特征才可被存入追踪器的budget中

  `matching_time_depth`：进行关联的时间窗口大小，当tracklet超过窗口时间没有被更新时，视为对象离开场景，删除tracklet

#### `MOTTracker.py`

- 多目标追踪方法的接口，包括`Detection`、`Tracker`、`MOTTracker`几个类

- `Detection`：定义检测框的属性

  ​	`tlwh: numpy.array` 检测框的位置，左上坐标以及宽高

  ​	`confidence: float` 检测框的得分

  ​	`feature: numpy.array` 检测框的appearance特征

- `Tracker`：定义单目标追踪器的属性

  ​	`track_id: int` 追踪id

  ​	`time_since_update: int` 距上次追踪成功的时间

  ​	`features: list` tracklet的budget存储的appearance特征

  ​	`state: int` tracklet的目前的状态，0表示初始状态（未确认存在，可能是误检，只能进行iou匹配），1表示确认状态（已确认，进行级联最小代价匹配，以及iou匹配），-1表示tracklet生命周期结束（超过matching_time_depth的时间未追踪到）

  ​	`hits: int` 计数追踪到目标的次数

  ​	`kf: kalman_filter.KalmanFilter` 单目标追踪器

- `MOTTracker`：多目标追踪类，包括级联匈牙利匹配算法，iou匹配算法，以及对追踪过程的维护

  ​	`budget: int` 每个单目标追踪器存储的appearance特征的最大个数

  ​	`max_iou_distance: float` iou association阈值

  ​	`matching_time_depth: int` 进行关联的时间窗口大小

  ​	`n_init: int` 连续n_init的时间追踪到对象，则确认对象

  ​	`tracks: list` 存储此多目标追踪器的各个单目标追踪器

  ​	`_next_id: int` 内部计数属性，记录下一个被分配的track id值

  ​	`distance: numpy.array` 记录当前被用于匹配过程的距离矩阵

  ​	`max_cosine_distance: float` appearance association阈值

#### `kalman_filter.py`

- 卡尔曼滤波器

## 可执行脚本执行命令示例

#### 格式

```
python mottracker_app.py \
    --sequence_dir=/path/to/motchanllenge/dataset \
    --detection_mat=/path/to/detection/numpy/file \
    --output=/path/to/output \
    --min_confidence=0.3 \
    --max_cosine_distance=0.2 \
    --nn_budget=100 \
    --budget_association_threshold=1.0 \
    --budget_detection_threshold=0.3 \
    --matching_time_depth=30
```

#### 在我电脑上的示例

```
python mottracker_app.py \
    --sequence_dir=/home/lichen/MOT/MOT17/train/MOT17-13-FRCNN \
    --detection_mat=/home/lichen/MOT/MOT_detections/train/MOT17-13-FRCNN.npy \
    --output=/home/lichen/ds-output/MOT17-13-FRCNN.txt \
    --min_confidence=0.3 \
    --max_cosine_distance=0.2 \
    --nn_budget=100 \
    --budget_association_threshold=1.0 \
    --budget_detection_threshold=0.3 \
    --matching_time_depth=30
```