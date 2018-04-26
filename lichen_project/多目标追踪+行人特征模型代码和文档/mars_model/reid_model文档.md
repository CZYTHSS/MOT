##inference部分

#### generate_detections.py

入口脚本，可在其中设置model的路径（变量model），以及提取特征的图片大小（变量image_shape，需与训练模型的image_shape保持一致）

#### generate_detections.py

Inference模型结构

|            层名称             | 卷积核大小/步长 | 输出feature map尺寸 |
| :------------------------: | :------: | :-------------: |
|           Conv 1           |  3x3/1   |  32 x 128 x 64  |
|           Conv 2           |  3x3/1   |  32 x 128 x 64  |
|         Max pool 3         |  3x3/2   |  32 x 64 x 32   |
|         Residual 4         |  3x3/1   |  32 x 64 x 32   |
|         Residual 5         |  3x3/1   |  32 x 64 x 32   |
|         Residual 6         |  3x3/2   |  64 x 32 x 16   |
|         Residual 7         |  3x3/1   |  64 x 32 x 16   |
|         Residual 8         |  3x3/2   |  128 x 16 x 8   |
|         Residual 9         |  3x3/1   |  128 x 16 x 8   |
|          Dense 10          |          |       128       |
| Batch and l2 Normalization |          |       128       |

#### 脚本使用示例

```
python generate_detections.py \
	--sequence_dir=/home/lichen/MOT/MOT17/train/MOT17-13-FRCNN \
	--detection_txt=/home/lichen/MOT/MOT17/train/MOT17-13-FRCNN/det/det.txt \
	--detection_npy==/home/lichen/MOT/MOT17/train/MOT17-13-FRCNN/det/det.npy
```

## training部分

训练集：

L. Zheng, Z. Bie, Y. Sun, J. Wang, C. Su, S. Wang, and Q. Tian, “MARS: A video benchmark for large-scale person re-identification,” in ECCV, 2016.

#### train.py

模型训练脚本，具体参数可在脚本中设置

![WX20171127-235152](/Users/lichen/Desktop/mars_model/WX20171127-235152.png)

#### train_features.py

training模型结构，相比于inference模型多了softmax loss用于进行分类训练

####脚本使用示例

```
python train.py --sequence_dir=/path/to/training/dataset
```