# Tianchi_Jinnan
天池算法津南数字制造算法挑战赛初赛代码
### This repository is based on [mmdetection](https://github.com/open-mmlab/mmdetection)
### 配置代码库
    cd code/
    bash install.sh

### 对数据集进行预处理
请将 jinnan2_round1_20190305.zip, jinnan2_round1_test_a_20190306.zip, jinnan2_round1_test_b_20190326.zip 放置于 data/First_round_data/ 文件夹下，运行以下命令，数据集预处理需要大约10分钟

    cd data/First_round_data/ 
    bash data_preprocessing.sh

#### 预处理介绍
该模型所用的训练集及验证集在 data 文件下，data/train_val/文件下为训练集+验证集的所有图片，共1381张图片。
data/annotations/train.json 和 data/annotations/val.json 分别是训练集和验证集的json文件。
训练集共有1282张图片，其中882张图片是从比赛提供的restricted图片里随机选择的限制品图片，剩余400张是利用代码生成的图片，
具体生成方式为：从限制品图片中随机选择1200个bounding box信息，求得该限制品的bounding box与minAreaRect的重叠区域，
将该区域随机覆盖到一张随机normal图片的随机位置上，以此来合成新的限制品图片。验证集为99张restricted图片。
（注：我们随机选择了400张normal图片作为可以被覆盖的图片，但400张图片中可能有少量图片未被随机选中并覆盖，则此类图片不会被训练，因此训练集中合成的新图片可能小于400张，训练集总图片数可能小于1282张。

#### 数据集结构：
    |--First_round_data
        |--annotations
            |--train.json  #未增强的训练集
            |--train_aug.json  #增强的训练集
            |--val.json  #验证集
            |--test_a.json  #test_a的json文件，由于推理部分代码读入的是json文件，因此将test_a的图片转换为json文件，该json文件只有图片信息，没有标注信息。
            |--test_b.json  #test_b的json文件
        |--data_preprocessing.sh  #数据集预处理脚本
        |--train_val  #训练集和验证集图片
        |--split_datasets.py  #随机分配训练集与验证集，分配比例9:1
        |--select_normal_images.py  #随机选择400张normal图片作为可能被覆盖的图片
        |--data_augmentation.py  #随机将1200个限制品的boungding box与minAreaRect重叠区域覆盖到随机一张被选择的normal图片的随机位置
        |--merge.py  #将被覆盖的图片与原有限制品图片放置在一个文件夹下
        |--test_json.py  #测试集文件下的图片转换成/annotations/test_b.json
        


### 训练代码
    cd code/
    bash train.sh


### 生成submit文件
code/work_dirs/submit_weights/ 文件夹下有两个训练好的权重，其中submit_a.pth为A榜最好得分0.5685的权重，submit_b.pth为B榜最好得分0.5272的权重
#### A榜结果
    cd code/ 
    bash submit_a.sh
#### B榜结果 
    cd code/
    bash submit_b.sh
推理结果以当前时间命名，存放于 submit 文件下。

