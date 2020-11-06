---
thumbnail: https://image.zhangxiann.com/andrea-hagenhoff-dIXCt2zUzV0-unsplash.jpg
toc: true
date: 2020/4/9 19:11:20
disqusId: zhangxian
categories:
- PyTorch

tags:
- AI
- Deep Learning
---

> 本章代码：[https://github.com/zhangxiann/PyTorch_Practice/blob/master/lesson7/finetune_resnet18.py](https://github.com/zhangxiann/PyTorch_Practice/blob/master/lesson7/finetune_resnet18.py)

这篇文章主要介绍了模型的 Finetune。

迁移学习：把在 source domain 任务上的学习到的模型应用到 target domain 的任务。

Finetune 就是一种迁移学习的方法。比如做人脸识别，可以把 ImageNet 看作 source domain，人脸数据集看作 target domain。通常来说 source domain 要比 target domain 大得多。可以利用  ImageNet 训练好的网络应用到人脸识别中。

对于一个模型，通常可以分为前面的 feature extractor (卷积层)和后面的 classifier，在 Finetune  时，通常不改变  feature extractor 的权值，也就是冻结卷积层；并且改变最后一个全连接层的输出来适应目标任务，训练后面 classifier 的权值，这就是 Finetune。<!--more-->通常  target domain 的数据比较小，不足以训练全部参数，容易导致过拟合，因此不改变  feature extractor 的权值。

Finetune 步骤如下：

1. 获取预训练模型的参数
2. 使用`load_state_dict()`把参数加载到模型中
3. 修改输出层
4. 固定 feature extractor 的参数。这部分通常有 2 种做法：
   1. 固定卷积层的预训练参数。可以设置`requires_grad=False`或者`lr=0`
   2. 可以通过`params_group`给 feature extractor 设置一个较小的学习率



下面微调 ResNet18，用于蜜蜂和蚂蚁图片的二分类。训练集每类数据各 120 张，验证集每类数据各 70 张图片。

数据下载地址：http://download.pytorch.org/tutorial/hymenoptera_data.zip

预训练好的模型参数下载地址：http://download.pytorch.org/models/resnet18-5c106cde.pth



## 不使用 Finetune

第一次我们首先不使用 Finetune，而是从零开始训练模型，这时只需要修改全连接层即可：

```
# 首先拿到 fc 层的输入个数
num_ftrs = resnet18_ft.fc.in_features
# 然后构造新的 fc 层替换原来的 fc 层
resnet18_ft.fc = nn.Linear(num_ftrs, classes)
```

输出如下：

```
use device :cpu
Training:Epoch[000/025] Iteration[010/016] Loss: 0.7192 Acc:47.50%
Valid:	 Epoch[000/025] Iteration[010/010] Loss: 0.6885 Acc:51.63%
Training:Epoch[001/025] Iteration[010/016] Loss: 0.6568 Acc:60.62%
Valid:	 Epoch[001/025] Iteration[010/010] Loss: 0.6360 Acc:59.48%
Training:Epoch[002/025] Iteration[010/016] Loss: 0.6411 Acc:60.62%
Valid:	 Epoch[002/025] Iteration[010/010] Loss: 0.6191 Acc:66.01%
Training:Epoch[003/025] Iteration[010/016] Loss: 0.5765 Acc:71.25%
Valid:	 Epoch[003/025] Iteration[010/010] Loss: 0.6179 Acc:67.32%
Training:Epoch[004/025] Iteration[010/016] Loss: 0.6074 Acc:67.50%
Valid:	 Epoch[004/025] Iteration[010/010] Loss: 0.6251 Acc:62.75%
Training:Epoch[005/025] Iteration[010/016] Loss: 0.6177 Acc:58.75%
Valid:	 Epoch[005/025] Iteration[010/010] Loss: 0.6541 Acc:64.71%
Training:Epoch[006/025] Iteration[010/016] Loss: 0.6103 Acc:65.62%
Valid:	 Epoch[006/025] Iteration[010/010] Loss: 0.7100 Acc:60.78%
Training:Epoch[007/025] Iteration[010/016] Loss: 0.6560 Acc:60.62%
Valid:	 Epoch[007/025] Iteration[010/010] Loss: 0.6019 Acc:67.32%
Training:Epoch[008/025] Iteration[010/016] Loss: 0.5454 Acc:70.62%
Valid:	 Epoch[008/025] Iteration[010/010] Loss: 0.5761 Acc:71.90%
Training:Epoch[009/025] Iteration[010/016] Loss: 0.5499 Acc:71.25%
Valid:	 Epoch[009/025] Iteration[010/010] Loss: 0.5598 Acc:71.90%
Training:Epoch[010/025] Iteration[010/016] Loss: 0.5466 Acc:69.38%
Valid:	 Epoch[010/025] Iteration[010/010] Loss: 0.5535 Acc:70.59%
Training:Epoch[011/025] Iteration[010/016] Loss: 0.5310 Acc:68.12%
Valid:	 Epoch[011/025] Iteration[010/010] Loss: 0.5700 Acc:70.59%
Training:Epoch[012/025] Iteration[010/016] Loss: 0.5024 Acc:72.50%
Valid:	 Epoch[012/025] Iteration[010/010] Loss: 0.5537 Acc:71.90%
Training:Epoch[013/025] Iteration[010/016] Loss: 0.5542 Acc:71.25%
Valid:	 Epoch[013/025] Iteration[010/010] Loss: 0.5836 Acc:71.90%
Training:Epoch[014/025] Iteration[010/016] Loss: 0.5458 Acc:71.88%
Valid:	 Epoch[014/025] Iteration[010/010] Loss: 0.5714 Acc:71.24%
Training:Epoch[015/025] Iteration[010/016] Loss: 0.5331 Acc:72.50%
Valid:	 Epoch[015/025] Iteration[010/010] Loss: 0.5613 Acc:73.20%
Training:Epoch[016/025] Iteration[010/016] Loss: 0.5296 Acc:71.25%
Valid:	 Epoch[016/025] Iteration[010/010] Loss: 0.5646 Acc:71.24%
Training:Epoch[017/025] Iteration[010/016] Loss: 0.5039 Acc:75.00%
Valid:	 Epoch[017/025] Iteration[010/010] Loss: 0.5643 Acc:71.24%
Training:Epoch[018/025] Iteration[010/016] Loss: 0.5351 Acc:73.75%
Valid:	 Epoch[018/025] Iteration[010/010] Loss: 0.5745 Acc:71.24%
Training:Epoch[019/025] Iteration[010/016] Loss: 0.5441 Acc:69.38%
Valid:	 Epoch[019/025] Iteration[010/010] Loss: 0.5703 Acc:71.90%
Training:Epoch[020/025] Iteration[010/016] Loss: 0.5582 Acc:69.38%
Valid:	 Epoch[020/025] Iteration[010/010] Loss: 0.5759 Acc:71.90%
Training:Epoch[021/025] Iteration[010/016] Loss: 0.5219 Acc:73.75%
Valid:	 Epoch[021/025] Iteration[010/010] Loss: 0.5689 Acc:72.55%
Training:Epoch[022/025] Iteration[010/016] Loss: 0.5670 Acc:70.62%
Valid:	 Epoch[022/025] Iteration[010/010] Loss: 0.6052 Acc:69.28%
Training:Epoch[023/025] Iteration[010/016] Loss: 0.5725 Acc:65.62%
Valid:	 Epoch[023/025] Iteration[010/010] Loss: 0.6047 Acc:68.63%
Training:Epoch[024/025] Iteration[010/016] Loss: 0.5761 Acc:66.25%
Valid:	 Epoch[024/025] Iteration[010/010] Loss: 0.5923 Acc:70.59%
```



训练了 25 个 epoch 后的准确率为：70.59%。

训练的 loss 曲线如下：

<div align="center"><img src="https://image.zhangxiann.com/20200707194544.png"/></div><br>

## 使用 Finetune

然后我们把下载的模型参数加载到模型中：

```
path_pretrained_model = enviroments.resnet18_path
state_dict_load = torch.load(path_pretrained_model)
resnet18_ft.load_state_dict(state_dict_load)
```



### 不冻结卷积层

这时我们不冻结卷积层，所有层都是用相同的学习率，输出如下：

```
use device :cpu
Training:Epoch[000/025] Iteration[010/016] Loss: 0.6299 Acc:65.62%
Valid:	 Epoch[000/025] Iteration[010/010] Loss: 0.3387 Acc:90.20%
Training:Epoch[001/025] Iteration[010/016] Loss: 0.3122 Acc:90.00%
Valid:	 Epoch[001/025] Iteration[010/010] Loss: 0.2150 Acc:94.12%
Training:Epoch[002/025] Iteration[010/016] Loss: 0.2748 Acc:85.62%
Valid:	 Epoch[002/025] Iteration[010/010] Loss: 0.2423 Acc:91.50%
Training:Epoch[003/025] Iteration[010/016] Loss: 0.1440 Acc:94.38%
Valid:	 Epoch[003/025] Iteration[010/010] Loss: 0.1666 Acc:95.42%
Training:Epoch[004/025] Iteration[010/016] Loss: 0.1983 Acc:92.50%
Valid:	 Epoch[004/025] Iteration[010/010] Loss: 0.1809 Acc:94.77%
Training:Epoch[005/025] Iteration[010/016] Loss: 0.1840 Acc:92.50%
Valid:	 Epoch[005/025] Iteration[010/010] Loss: 0.2437 Acc:91.50%
Training:Epoch[006/025] Iteration[010/016] Loss: 0.1921 Acc:93.12%
Valid:	 Epoch[006/025] Iteration[010/010] Loss: 0.2014 Acc:95.42%
Training:Epoch[007/025] Iteration[010/016] Loss: 0.1311 Acc:93.12%
Valid:	 Epoch[007/025] Iteration[010/010] Loss: 0.1890 Acc:96.08%
Training:Epoch[008/025] Iteration[010/016] Loss: 0.1395 Acc:94.38%
Valid:	 Epoch[008/025] Iteration[010/010] Loss: 0.1907 Acc:95.42%
Training:Epoch[009/025] Iteration[010/016] Loss: 0.1390 Acc:93.75%
Valid:	 Epoch[009/025] Iteration[010/010] Loss: 0.1933 Acc:95.42%
Training:Epoch[010/025] Iteration[010/016] Loss: 0.1065 Acc:96.88%
Valid:	 Epoch[010/025] Iteration[010/010] Loss: 0.1865 Acc:95.42%
Training:Epoch[011/025] Iteration[010/016] Loss: 0.0845 Acc:98.12%
Valid:	 Epoch[011/025] Iteration[010/010] Loss: 0.1851 Acc:96.08%
Training:Epoch[012/025] Iteration[010/016] Loss: 0.1068 Acc:95.62%
Valid:	 Epoch[012/025] Iteration[010/010] Loss: 0.1862 Acc:95.42%
Training:Epoch[013/025] Iteration[010/016] Loss: 0.0986 Acc:96.25%
Valid:	 Epoch[013/025] Iteration[010/010] Loss: 0.1803 Acc:96.73%
Training:Epoch[014/025] Iteration[010/016] Loss: 0.1083 Acc:96.88%
Valid:	 Epoch[014/025] Iteration[010/010] Loss: 0.1867 Acc:96.08%
Training:Epoch[015/025] Iteration[010/016] Loss: 0.0683 Acc:98.12%
Valid:	 Epoch[015/025] Iteration[010/010] Loss: 0.1863 Acc:95.42%
Training:Epoch[016/025] Iteration[010/016] Loss: 0.1271 Acc:96.25%
Valid:	 Epoch[016/025] Iteration[010/010] Loss: 0.1842 Acc:94.77%
Training:Epoch[017/025] Iteration[010/016] Loss: 0.0857 Acc:97.50%
Valid:	 Epoch[017/025] Iteration[010/010] Loss: 0.1776 Acc:96.08%
Training:Epoch[018/025] Iteration[010/016] Loss: 0.1338 Acc:94.38%
Valid:	 Epoch[018/025] Iteration[010/010] Loss: 0.1736 Acc:96.08%
Training:Epoch[019/025] Iteration[010/016] Loss: 0.1381 Acc:95.62%
Valid:	 Epoch[019/025] Iteration[010/010] Loss: 0.1852 Acc:93.46%
Training:Epoch[020/025] Iteration[010/016] Loss: 0.0936 Acc:96.25%
Valid:	 Epoch[020/025] Iteration[010/010] Loss: 0.1820 Acc:95.42%
Training:Epoch[021/025] Iteration[010/016] Loss: 0.1818 Acc:93.75%
Valid:	 Epoch[021/025] Iteration[010/010] Loss: 0.1949 Acc:92.81%
Training:Epoch[022/025] Iteration[010/016] Loss: 0.1525 Acc:93.75%
Valid:	 Epoch[022/025] Iteration[010/010] Loss: 0.1816 Acc:95.42%
Training:Epoch[023/025] Iteration[010/016] Loss: 0.1942 Acc:93.12%
Valid:	 Epoch[023/025] Iteration[010/010] Loss: 0.1744 Acc:96.08%
Training:Epoch[024/025] Iteration[010/016] Loss: 0.1268 Acc:96.25%
Valid:	 Epoch[024/025] Iteration[010/010] Loss: 0.1808 Acc:96.08%
```



训练了 25 个 epoch 后的准确率为：96.08%。

训练的 loss 曲线如下：

<div align="center"><img src="https://image.zhangxiann.com/20200707210357.png"/></div><br>

### 冻结卷积层



#### 设置`requires_grad=False` 

这里先冻结所有参数，然后再替换全连接层，相当于冻结了卷积层的参数：

```
for param in resnet18_ft.parameters():
	param.requires_grad = False
	# 首先拿到 fc 层的输入个数
num_ftrs = resnet18_ft.fc.in_features
# 然后构造新的 fc 层替换原来的 fc 层
resnet18_ft.fc = nn.Linear(num_ftrs, classes)
```

这里不提供实验结果。



#### 设置学习率为 0

这里把卷积层的学习率设置为 0，需要在优化器里设置不同的学习率。首先获取全连接层参数的地址，然后使用 filter 过滤不属于全连接层的参数，也就是保留卷积层的参数；接着设置优化器的分组学习率，传入一个 list，包含 2 个元素，每个元素是字典，对应 2 个参数组。其中卷积层的学习率设置为 全连接层的 0.1 倍。

```
# 首先获取全连接层参数的地址
fc_params_id = list(map(id, resnet18_ft.fc.parameters()))     # 返回的是parameters的 内存地址
# 然后使用 filter 过滤不属于全连接层的参数，也就是保留卷积层的参数
base_params = filter(lambda p: id(p) not in fc_params_id, resnet18_ft.parameters())
# 设置优化器的分组学习率，传入一个 list，包含 2 个元素，每个元素是字典，对应 2 个参数组
optimizer = optim.SGD([{'params': base_params, 'lr': 0}, {'params': resnet18_ft.fc.parameters(), 'lr': LR}], momentum=0.9)
```

这里不提供实验结果。



### 使用分组学习率

这里不冻结卷积层，而是对卷积层使用较小的学习率，对全连接层使用较大的学习率，需要在优化器里设置不同的学习率。首先获取全连接层参数的地址，然后使用 filter 过滤不属于全连接层的参数，也就是保留卷积层的参数；接着设置优化器的分组学习率，传入一个 list，包含 2 个元素，每个元素是字典，对应 2 个参数组。其中卷积层的学习率设置为 全连接层的 0.1 倍。

```
# 首先获取全连接层参数的地址
fc_params_id = list(map(id, resnet18_ft.fc.parameters()))     # 返回的是parameters的 内存地址
# 然后使用 filter 过滤不属于全连接层的参数，也就是保留卷积层的参数
base_params = filter(lambda p: id(p) not in fc_params_id, resnet18_ft.parameters())
# 设置优化器的分组学习率，传入一个 list，包含 2 个元素，每个元素是字典，对应 2 个参数组
optimizer = optim.SGD([{'params': base_params, 'lr': LR*0}, {'params': resnet18_ft.fc.parameters(), 'lr': LR}], momentum=0.9)
```

这里不提供实验结果。





## 使用 GPU 的 tips

PyTorch 模型使用 GPU，可以分为 3 步：

1. 首先获取 device：`device = torch.device("cuda" if torch.cuda.is_available() else "cpu")`
2. 把模型加载到 device：`model.to(device)`
3. 在 data_loader 取数据的循环中，把每个 mini-batch 的数据和 label 加载到 device：`inputs, labels = inputs.to(device), labels.to(device)`



**参考资料**

- [深度之眼 PyTorch 框架班](https://ai.deepshare.net/detail/p_5df0ad9a09d37_qYqVmt85/6)

<br>

如果你觉得这篇文章对你有帮助，不妨点个赞，让我有更多动力写出好文章。
<br>

我的文章会首发在公众号上，欢迎扫码关注我的公众号**张贤同学**。

<div align="center"><img src="https://image.zhangxiann.com/QRcode_8cm.jpg"/></div><br>