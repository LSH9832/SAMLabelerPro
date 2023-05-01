# SAMLabeler Pro: 使用 [Segment Anyting Model](https://github.com/facebookresearch/segment-anything) 辅助的图像标注工具，支持远程多人同时标注

![image](https://user-images.githubusercontent.com/69880398/235317010-2ec560cf-1de9-436d-81a4-79654e533de1.png)

## 注意
- 本工具非原创，**魔改自[yatengLG](https://github.com/yatengLG) 大佬的[ISAT_with_segment_anything](https://github.com/yatengLG/ISAT_with_segment_anything) ，包含其所有功能**（截至2023年4月26日23:59），界面也相同，其功能使用方法请阅读该项目的[README.md](https://github.com/yatengLG/ISAT_with_segment_anything/blob/master/README.md) 。再次感谢大佬的分享！

![image](https://github.com/yatengLG/ISAT_with_segment_anything/raw/master/display/%E6%A0%87%E6%B3%A8%20-big-original.gif)

- 使用本工具时为避免导入冲突，请务必不要在运行环境中安装SAM源码，本项目中的segment_anything文件夹便是作了一定改动的SAM源码。
- 如果有QT报错，大概率是opencv的原因，请把requirements.txt文件中的opencv_python换为opencv_python_headless

## 1 即将更新（祝大家劳动节快乐）

- 暂无，发现bug请在Issues中留言

## 2 相对于原版的新特性

### 2.1 支持局域网内多人同时远程标注

当所有需要标注的图片在某一台电脑上时，可以通过启动服务端程序，使其他电脑能够通过本工具远程访问并进行标注，标注数据将保存至服务端配置文件中指定的文件夹。

![image](https://user-images.githubusercontent.com/69880398/235212348-79245f9d-907a-481c-ad7f-52b79339592b.png)

当准备编辑的图像有其他用户正在编辑时弹出提示，防止覆盖他人标注结果

![image](https://user-images.githubusercontent.com/69880398/235287612-7f9e2e40-b81d-4d9a-a872-b35b6e6dce14.png)

### 2.2 半精度推理
可以选择使用半精度推理（默认打开），大幅减少显存消耗（最大的模型貌似是过拟合了还是怎么的，使用半精度效果奇差，本工具会强制使用单精度，但另外两个模型的半精度推理能够正常使用）

如果需要关闭半精度，只需要在第一次运行后自动生成的cache/last_edit.yaml中设置
```yaml
half: false  # 
```
下次打开就使用单精度FP32了。

在RTX3060 12G上进行测试，显存占用变化如下

- [vit_b模型](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth) ：0.9G（打开本工具前）→ 2.6G（打开工具并开始标注后）
- [vit_l模型](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth) ：0.9G（打开本工具前）→ 3.1G（打开工具并开始标注后）
- 由于[vit_h模型](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth) 半精度效果差，在此不列出。

### 2.3 可手动选择模型大小

原工具会从大到小优先的原则使用，有时候同时下载了三个模型，但标注时想要使用较小的模型，可以手动选择加载的模型，同样在cache/last_edit.yaml中设置
```yaml
force_model_type: l   # 或 b
```
再打开本工具即可

### 2.4 增加COCO格式转换工具

对于本工具标注的标签格式，可以通过 **“工具-Convert to COCO”** 打开标签转换工具转换为COCO格式，并配有划分训练集、验证集、测试集的功能。

![image](https://user-images.githubusercontent.com/69880398/235314366-ac94f210-06d4-48c7-9e6d-7a4728cb3272.png)

![image](https://user-images.githubusercontent.com/69880398/235314404-fc57e176-c8e5-4112-bed7-62e59e3693d4.png)


### 2.5 多种数据集生成segment标签

支持YOLO/VOC/COCO/VisDrone格式的bbox标签生成可在本工具中编辑的segment标签文件。

首先在cfg/dataset/****.yaml配置文件中编辑相应格式数据集的相关信息（数据集目录等信息），注意，VOC格式必须有两个txt文件分别包含训练集和测试集所有图像的名称（不带后缀），如原VOC2012数据集可写成
```yaml
# 见 cfg/dataset/voc.yaml
type: "voc"

dataset_path: "F:/dataset/VOC/VOC2012"

kwargs:
  suffix: "jpg"
  use_cache: true                      # 如果非首次加载，是否从生成的缓存文件中加载（可提高加载速度）

train:
  image_dir: "JPEGImages"
  anno_dir: "Annotations"
  label: "ImageSets/Main/train.txt"    # 描述训练集的txt文件

val:
  image_dir: "JPEGImages"
  anno_dir: "Annotations"
  label: "ImageSets/Main/val.txt"      # 描述验证集的txt文件

test:
  test_dir: "test"

segmentaion_enabled: false

names: ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog',
        'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train',  'tvmonitor']

```

然后使用如下命令即可转换
```shell
python demo/box2segment.py --cfg cfg/dataset/yolo.yaml
                           --size h     # SAM模型大小， b, l
                           --half       # 半精度，h模型无效
                           --dist datasets/myYOLODataset/annotations   # 生成的标签的保存路径，训练集在子文件夹train中，验证集在val中
                           --val        # 是否是验证集，否则为训练集
                           --overwrite  # 覆盖保存路径中已有的同名标签
```

### 2.6 使用体验优化

#### 2.6.1 快速恢复
重新打开本工具会记住上一次关闭时加载的图片以及相应的图像和标签的文件夹，以及所有的用于标注的类别，从而快速继续标注工作

#### 2.6.2 图片切换速度优化

通过创建子线程，在子线程中让SAM加载图片，大幅优化图片间的切换速度，使用最大模型时尤其明显

#### 2.6.3 快速标注

标注时会记住上一次标注的类别，若一直标注同一类别无需重新选择，同时增设快捷键，当进入类别选择菜单时：
- 按下“E”键：确认（不带group id）
- 按下“Q”键：确认（沿用上次标注时的group id，即同一目标的不同部分）
- 按下“W”键：确认（在上次标注时的group id的基础上+1，即下一个目标）
- 按下“C”键：取消

因此可以通过以下连招：
```
1. Q → 鼠标点点点 → E → E → Q → 鼠标点点点 → E → E → Q → ...      # 不带标签标注目标
2. Q → 鼠标点点点 → E → Q → Q → 鼠标点点点 → E → Q → Q → ...      # 带标签标注同一目标
3. Q → 鼠标点点点 → E → W → Q → 鼠标点点点 → E → W → Q → ...      # 带标签标注不同目标
```
进行快速标注。

#### 2.5.4 增大图像缩放倍数

原来只能放大3倍，拿4K屏标注小图片的小目标可就太难受了，现在能放大100倍

#### 2.5.5 新增预设标签

增加了coco数据集、voc数据集、visdrone数据集、dota数据集的类别预设标签，均在settings文件夹下，可以通过设置导入

## 3 安装与运行

### 3.1 客户端
- 首先把本工具和[模型](#22-半精度推理)下载下来
```shell
git clone https://github.com/LSH9832/SAMLabelerPro.git
cd SAMLabelerPro

# （for linux）根据需求和设备性能选择性下载
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
```
- 然后torch和torchvision不用多说，按照pytorch官网[给出的步骤](https://pytorch.org/get-started/previous-versions/) 安装， 然后
```shell
pip install -r requirements.txt
python main.py
```
即可。如果报错，见[注意](#注意)最后一条

- 远程标注功能在“文件→远程模式设置”中，在打开远程模式前，需要先运行服务端。

### 3.2服务端

- 首先在服务端设备上下载本工具代码，然后打开“settings/server_settings.yaml”并进行修改，内容如下
```yaml
users:            # 所有可登录的用户信息，
  admin:          # 用户名
    pwd: admin    # 密码
    image_path: "./example/images"         # 该用户登陆后需要标注的图片所在文件夹
    label_path: "./example/images"         # 该用户标注的结果保存的文件夹
    category_file: "./settings/coco.yaml"  # 该用户标注时使用的预设标签类别
  user:
    pwd: password
    image_path: "./example/images"
    label_path: "./example/images"
    category_file: "./settings/coco.yaml"
average: false                            # 是否平均分配，同一个文件夹分配多个人进行标注时，将图片平均分配，每个人只能访问自己分配到的图片，否则这些人能够访问所有图片
```
- 不同于客户端，服务端只需要安装两个第三方库即可运行，即
```shell
pip install flask pyyaml

# host：可访问服务端的IP网段，0.0.0.0代表所有网段均可访问
# port：该服务的端口
python server.py --host 0.0.0.0 --port 12345
```






