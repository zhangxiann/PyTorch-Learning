---
thumbnail: https://image.zhangxiann.com/andrea-hagenhoff-dIXCt2zUzV0-unsplash.jpg
toc: true
date: 2020/4/1 19:19:20
disqusId: zhangxian
categories:
- PyTorch

tags:
- AI
- Deep Learning
---

> 本章代码：
>
> - [https://github.com/zhangxiann/PyTorch_Practice/blob/master/lesson6/bn_and_initialize.py](https://github.com/zhangxiann/PyTorch_Practice/blob/master/lesson6/bn_and_initialize.py)
> - [https://github.com/zhangxiann/PyTorch_Practice/blob/master/lesson6/bn_in_123_dim.py](https://github.com/zhangxiann/PyTorch_Practice/blob/master/lesson6/bn_in_123_dim.py)
> - [https://github.com/zhangxiann/PyTorch_Practice/blob/master/lesson6/normallization_layers.py](https://github.com/zhangxiann/PyTorch_Practice/blob/master/lesson6/normallization_layers.py)

这篇文章主要介绍了 Batch Normalization 的概念，以及 PyTorch 中的 1d/2d/3d Batch Normalization 实现。



# Batch Normalization

称为批标准化。批是指一批数据，通常为 mini-batch；标准化是处理后的数据服从$N(0,1)$的正态分布。

批标准化的优点有如下：

- 可以使用更大的学习率，加速模型收敛
- 可以不用精心设计权值初始化
- 可以不用 dropout 或者较小的 dropout
- 可以不用 L2 或者较小的 weight decay
- 可以不用 LRN (local response normalization)<!--more-->



假设输入的 mini-batch 数据是$\mathcal{B}=\left\{x_{1 \dots m}\right\}$，Batch Normalization 的可学习参数是$\gamma, \beta$，步骤如下：

- 求 mini-batch 的均值：$\mu_{\mathcal{B}} \leftarrow \frac{1}{m} \sum_{i=1}^{m} x_{i}$
- 求 mini-batch 的方差：$\sigma_{\mathcal{B}}^{2} \leftarrow \frac{1}{m} \sum_{i=1}\left(x_{i}-\mu_{\mathcal{B}}\right)^{2}$
- 标准化：$\widehat{x}_{i} \leftarrow \frac{x_{i}-\mu_{\mathcal{B}}}{\sqrt{\sigma_{B}^{2}+\epsilon}}$，其中$\epsilon$ 是放置分母为 0 的一个数
- affine transform(缩放和平移)：$y_{i} \leftarrow \gamma \widehat{x}_{i}+\beta \equiv \mathrm{B} \mathrm{N}_{\gamma, \beta}\left(x_{i}\right)$，这个操作可以增强模型的 capacity，也就是让模型自己判断是否要对数据进行标准化，进行多大程度的标准化。如果$\gamma= \sqrt{\sigma_{B}^{2}}$，$\beta=\mu_{\mathcal{B}}$，那么就实现了恒等映射。

Batch Normalization 的提出主要是为了解决 Internal Covariate Shift (ICS)。在训练过程中，数据需要经过多层的网络，如果数据在前向传播的过程中，尺度发生了变化，可能会导致梯度爆炸或者梯度消失，从而导致模型难以收敛。

Batch Normalization 层一般在激活函数前一层。

下面的代码打印一个网络的每个网络层的输出，在没有进行初始化时，数据尺度越来越小。

```
import torch
import numpy as np
import torch.nn as nn
from common_tools import set_seed

set_seed(1)  # 设置随机种子


class MLP(nn.Module):
    def __init__(self, neural_num, layers=100):
        super(MLP, self).__init__()
        self.linears = nn.ModuleList([nn.Linear(neural_num, neural_num, bias=False) for i in range(layers)])
        self.bns = nn.ModuleList([nn.BatchNorm1d(neural_num) for i in range(layers)])
        self.neural_num = neural_num

    def forward(self, x):

        for (i, linear), bn in zip(enumerate(self.linears), self.bns):
            x = linear(x)
            # x = bn(x)
            x = torch.relu(x)

            if torch.isnan(x.std()):
                print("output is nan in {} layers".format(i))
                break

            print("layers:{}, std:{}".format(i, x.std().item()))

        return x

    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):

                # method 1
                # nn.init.normal_(m.weight.data, std=1)    # normal: mean=0, std=1

                # method 2 kaiming
                nn.init.kaiming_normal_(m.weight.data)


neural_nums = 256
layer_nums = 100
batch_size = 16

net = MLP(neural_nums, layer_nums)
# net.initialize()

inputs = torch.randn((batch_size, neural_nums))  # normal: mean=0, std=1

output = net(inputs)
print(output)
```

当使用`nn.init.kaiming_normal_()`初始化后，数据的标准差尺度稳定在 [0.6, 0.9]。

当我们不对网络层进行权值初始化，而是在每个激活函数层之前使用 bn 层，查看数据的标准差尺度稳定在 [0.58, 0.59]。因此 Batch Normalization 可以不用精心设计权值初始化。

下面以人民币二分类实验中的 LeNet 为例，添加 bn 层，对比不带 bn 层的网络和带 bn 层的网络的训练过程。

不带 bn 层的网络，并且使用 kaiming 初始化权值，训练过程如下：

<div align="center"><img src="https://image.zhangxiann.com/20200706203137.png"/></div><br>
可以看到训练过程中，训练集的 loss 在中间激增到 1.4，不够稳定。

带有 bn 层的 LeNet 定义如下：

```
class LeNet_bn(nn.Module):
    def __init__(self, classes):
        super(LeNet_bn, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.bn1 = nn.BatchNorm2d(num_features=6)

        self.conv2 = nn.Conv2d(6, 16, 5)
        self.bn2 = nn.BatchNorm2d(num_features=16)

        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.bn3 = nn.BatchNorm1d(num_features=120)

        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, classes)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)

        out = F.max_pool2d(out, 2)

        out = self.conv2(out)
        out = self.bn2(out)
        out = F.relu(out)

        out = F.max_pool2d(out, 2)

        out = out.view(out.size(0), -1)

        out = self.fc1(out)
        out = self.bn3(out)
        out = F.relu(out)

        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out
```

带 bn 层的网络，并且不使用 kaiming 初始化权值，训练过程如下：

<div align="center"><img src="https://image.zhangxiann.com/20200706203417.png"/></div><br>
虽然训练过程中，训练集的 loss 也有激增，但只是增加到 0.4，非常稳定。



# Batch Normalization in PyTorch

在 PyTorch 中，有 3 个 Batch Normalization 类

- nn.BatchNorm1d()，输入数据的形状是 $B \times C \times 1D\_feature$
- nn.BatchNorm2d()，输入数据的形状是 $B \times C \times 2D\_feature$
- nn.BatchNorm3d()，输入数据的形状是 $B \times C \times 3D\_feature$

以`nn.BatchNorm1d()`为例，如下：

```
torch.nn.BatchNorm1d(num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
```

参数：

- num_features：一个样本的特征数量，这个参数最重要
- eps：在进行标准化操作时的分布修正项
- momentum：指数加权平均估计当前的均值和方差
- affine：是否需要 affine transform，默认为 True
- track_running_stats：True 为训练状态，此时均值和方差会根据每个 mini-batch 改变。False 为测试状态，此时均值和方差会固定

主要属性：

- runninng_mean：均值
- running_var：方差
- weight：affine transform 中的 $\gamma$
- bias：affine transform 中的 $\beta$

在训练时，均值和方差采用指数加权平均计算，也就是不仅考虑当前 mini-batch 的值均值和方差还考虑前面的 mini-batch 的均值和方差。

在训练时，均值方差固定为当前统计值。

所有的 bn 层都是根据**特征维度**计算上面 4 个属性，详情看下面例子。



## nn.BatchNorm1d()

输入数据的形状是 $B \times C \times 1D\_feature$。在下面的例子中，数据的维度是：(3, 5, 1)，表示一个 mini-batch 有 3 个样本，每个样本有 5 个特征，每个特征的维度是 1。那么就会计算 5 个均值和方差，分别对应每个特征维度。momentum 设置为 0.3，第一次的均值和方差默认为 0 和 1。输入两次 mini-batch 的数据。

数据如下图：

<div align="center"><img src="https://image.zhangxiann.com/20200706220302.png"/></div><br>
代码如下所示：

```
    batch_size = 3
    num_features = 5
    momentum = 0.3

    features_shape = (1)

    feature_map = torch.ones(features_shape)                                                    # 1D
    feature_maps = torch.stack([feature_map*(i+1) for i in range(num_features)], dim=0)         # 2D
    feature_maps_bs = torch.stack([feature_maps for i in range(batch_size)], dim=0)             # 3D
    print("input data:\n{} shape is {}".format(feature_maps_bs, feature_maps_bs.shape))

    bn = nn.BatchNorm1d(num_features=num_features, momentum=momentum)

    running_mean, running_var = 0, 1
    mean_t, var_t = 2, 0
    for i in range(2):
        outputs = bn(feature_maps_bs)

        print("\niteration:{}, running mean: {} ".format(i, bn.running_mean))
        print("iteration:{}, running var:{} ".format(i, bn.running_var))



        running_mean = (1 - momentum) * running_mean + momentum * mean_t
        running_var = (1 - momentum) * running_var + momentum * var_t

        print("iteration:{}, 第二个特征的running mean: {} ".format(i, running_mean))
        print("iteration:{}, 第二个特征的running var:{}".format(i, running_var))

```

输出为：

```
input data:
tensor([[[1.],
         [2.],
         [3.],
         [4.],
         [5.]],
        [[1.],
         [2.],
         [3.],
         [4.],
         [5.]],
        [[1.],
         [2.],
         [3.],
         [4.],
         [5.]]]) shape is torch.Size([3, 5, 1])
iteration:0, running mean: tensor([0.3000, 0.6000, 0.9000, 1.2000, 1.5000]) 
iteration:0, running var:tensor([0.7000, 0.7000, 0.7000, 0.7000, 0.7000]) 
iteration:0, 第二个特征的running mean: 0.6 
iteration:0, 第二个特征的running var:0.7
iteration:1, running mean: tensor([0.5100, 1.0200, 1.5300, 2.0400, 2.5500]) 
iteration:1, running var:tensor([0.4900, 0.4900, 0.4900, 0.4900, 0.4900]) 
iteration:1, 第二个特征的running mean: 1.02 
iteration:1, 第二个特征的running var:0.48999999999999994
```

虽然两个 mini-batch 的数据是一样的，但是 bn 层的均值和方差却不一样。以第二个特征的均值计算为例，值都是 2。

- 第一次 bn 层的均值计算：$running_mean=(1-momentum) \times pre_running_mean + momentum \times mean_t =(1-0.3) \times 0 + 0.3 \times 2 =0.6$
- 第二次 bn 层的均值计算：$running_mean=(1-momentum) \times pre_running_mean + momentum \times mean_t =(1-0.3) \times 0.6 + 0.3 \times 2 =1.02$



网络还没进行前向传播之前，断点查看 bn 层的属性如下：

<div align="center"><img src="https://image.zhangxiann.com/20200706220100.png"/></div><br>
## nn.BatchNorm2d()

输入数据的形状是 $B \times C \times 2D\_feature$。在下面的例子中，数据的维度是：(3, 3, 2, 2)，表示一个 mini-batch 有 3 个样本，每个样本有 3 个特征，每个特征的维度是 $1 \times 2$。那么就会计算 3 个均值和方差，分别对应每个特征维度。momentum 设置为 0.3，第一次的均值和方差默认为 0 和 1。输入两次 mini-batch 的数据。

数据如下图：

<div align="center"><img src="https://image.zhangxiann.com/20200706220726.png"/></div><br>
代码如下：

```
    batch_size = 3
    num_features = 3
    momentum = 0.3
    
    features_shape = (2, 2)

    feature_map = torch.ones(features_shape)                                                    # 2D
    feature_maps = torch.stack([feature_map*(i+1) for i in range(num_features)], dim=0)         # 3D
    feature_maps_bs = torch.stack([feature_maps for i in range(batch_size)], dim=0)             # 4D

    # print("input data:\n{} shape is {}".format(feature_maps_bs, feature_maps_bs.shape))

    bn = nn.BatchNorm2d(num_features=num_features, momentum=momentum)

    running_mean, running_var = 0, 1

    for i in range(2):
        outputs = bn(feature_maps_bs)

        print("\niter:{}, running_mean: {}".format(i, bn.running_mean))
        print("iter:{}, running_var: {}".format(i, bn.running_var))

        print("iter:{}, weight: {}".format(i, bn.weight.data.numpy()))
        print("iter:{}, bias: {}".format(i, bn.bias.data.numpy()))
```



输出如下：

```
iter:0, running_mean: tensor([0.3000, 0.6000, 0.9000])
iter:0, running_var: tensor([0.7000, 0.7000, 0.7000])
iter:0, weight: [1. 1. 1.]
iter:0, bias: [0. 0. 0.]
iter:1, running_mean: tensor([0.5100, 1.0200, 1.5300])
iter:1, running_var: tensor([0.4900, 0.4900, 0.4900])
iter:1, weight: [1. 1. 1.]
iter:1, bias: [0. 0. 0.]
```





## nn.BatchNorm3d()

输入数据的形状是 $B \times C \times 3D\_feature$。在下面的例子中，数据的维度是：(3, 2, 2, 2, 3)，表示一个 mini-batch 有 3 个样本，每个样本有 2 个特征，每个特征的维度是 $2 \times 2 \times 3$。那么就会计算 2 个均值和方差，分别对应每个特征维度。momentum 设置为 0.3，第一次的均值和方差默认为 0 和 1。输入两次 mini-batch 的数据。

数据如下图：

<div align="center"><img src="https://image.zhangxiann.com/20200706221801.png"/></div><br>
代码如下：

```
    batch_size = 3
    num_features = 3
    momentum = 0.3

    features_shape = (2, 2, 3)

    feature = torch.ones(features_shape)                                                # 3D
    feature_map = torch.stack([feature * (i + 1) for i in range(num_features)], dim=0)  # 4D
    feature_maps = torch.stack([feature_map for i in range(batch_size)], dim=0)         # 5D

    # print("input data:\n{} shape is {}".format(feature_maps, feature_maps.shape))

    bn = nn.BatchNorm3d(num_features=num_features, momentum=momentum)

    running_mean, running_var = 0, 1

    for i in range(2):
        outputs = bn(feature_maps)

        print("\niter:{}, running_mean.shape: {}".format(i, bn.running_mean.shape))
        print("iter:{}, running_var.shape: {}".format(i, bn.running_var.shape))

        print("iter:{}, weight.shape: {}".format(i, bn.weight.shape))
        print("iter:{}, bias.shape: {}".format(i, bn.bias.shape))

```

输出如下：

```
iter:0, running_mean.shape: torch.Size([3])
iter:0, running_var.shape: torch.Size([3])
iter:0, weight.shape: torch.Size([3])
iter:0, bias.shape: torch.Size([3])
iter:1, running_mean.shape: torch.Size([3])
iter:1, running_var.shape: torch.Size([3])
iter:1, weight.shape: torch.Size([3])
iter:1, bias.shape: torch.Size([3])
```



# Layer Normalization

提出的原因：Batch Normalization 不适用于变长的网络，如 RNN

思路：每个网络层计算均值和方差

注意事项：

- 不再有 running_mean 和 running_var
- $\gamma$ 和 $\beta$ 为逐样本的

<div align="center"><img src="https://image.zhangxiann.com/20200707095227.png"/></div><br>
```
torch.nn.LayerNorm(normalized_shape, eps=1e-05, elementwise_affine=True)
```

参数：

- normalized_shape：该层特征的形状，可以取$C \times H \times W$、$H \times W$、$W$
- eps：标准化时的分母修正项
- elementwise_affine：是否需要逐个样本 affine transform

下面代码中，输入数据的形状是 $B \times C \times feature$，(8, 2, 3, 4)，表示一个 mini-batch 有 8 个样本，每个样本有 2 个特征，每个特征的维度是 $3 \times 4$。那么就会计算 8 个均值和方差，分别对应每个样本。

```
    batch_size = 8
    num_features = 2

    features_shape = (3, 4)

    feature_map = torch.ones(features_shape)  # 2D
    feature_maps = torch.stack([feature_map * (i + 1) for i in range(num_features)], dim=0)  # 3D
    feature_maps_bs = torch.stack([feature_maps for i in range(batch_size)], dim=0)  # 4D

    # feature_maps_bs shape is [8, 6, 3, 4],  B * C * H * W
    # ln = nn.LayerNorm(feature_maps_bs.size()[1:], elementwise_affine=True)
    # ln = nn.LayerNorm(feature_maps_bs.size()[1:], elementwise_affine=False)
    # ln = nn.LayerNorm([6, 3, 4])
    ln = nn.LayerNorm([2, 3, 4])

    output = ln(feature_maps_bs)

    print("Layer Normalization")
    print(ln.weight.shape)
    print(feature_maps_bs[0, ...])
    print(output[0, ...])
```



```
Layer Normalization
torch.Size([2, 3, 4])
tensor([[[1., 1., 1., 1.],
         [1., 1., 1., 1.],
         [1., 1., 1., 1.]],
        [[2., 2., 2., 2.],
         [2., 2., 2., 2.],
         [2., 2., 2., 2.]]])
tensor([[[-1.0000, -1.0000, -1.0000, -1.0000],
         [-1.0000, -1.0000, -1.0000, -1.0000],
         [-1.0000, -1.0000, -1.0000, -1.0000]],
        [[ 1.0000,  1.0000,  1.0000,  1.0000],
         [ 1.0000,  1.0000,  1.0000,  1.0000],
         [ 1.0000,  1.0000,  1.0000,  1.0000]]], grad_fn=<SelectBackward>)
```

Layer Normalization 可以设置 normalized_shape 为 (3, 4) 或者 (4)。



# Instance Normalization

提出的原因：Batch Normalization 不适用于图像生成。因为在一个 mini-batch 中的图像有不同的风格，不能把这个 batch 里的数据都看作是同一类取标准化。

思路：逐个 instance 的 channel 计算均值和方差。也就是每个 feature map 计算一个均值和方差。

包括 InstanceNorm1d、InstanceNorm2d、InstanceNorm3d。

以`InstanceNorm1d`为例，定义如下：

```
torch.nn.InstanceNorm1d(num_features, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
```

参数：

- num_features：一个样本的特征数，这个参数最重要
- eps：分母修正项
- momentum：指数加权平均估计当前的的均值和方差
- affine：是否需要 affine transform
- track_running_stats：True 为训练状态，此时均值和方差会根据每个 mini-batch 改变。False 为测试状态，此时均值和方差会固定

下面代码中，输入数据的形状是 $B \times C \times 2D\_feature$，(3, 3, 2, 2)，表示一个 mini-batch 有 3 个样本，每个样本有 3 个特征，每个特征的维度是 $2 \times 2 $。那么就会计算  $3 \times 3 $ 个均值和方差，分别对应每个样本的每个特征。如下图所示：

<div align="center"><img src="https://image.zhangxiann.com/20200707103153.png"/></div><br>
下面是代码：

```
    batch_size = 3
    num_features = 3
    momentum = 0.3

    features_shape = (2, 2)

    feature_map = torch.ones(features_shape)    # 2D
    feature_maps = torch.stack([feature_map * (i + 1) for i in range(num_features)], dim=0)  # 3D
    feature_maps_bs = torch.stack([feature_maps for i in range(batch_size)], dim=0)  # 4D

    print("Instance Normalization")
    print("input data:\n{} shape is {}".format(feature_maps_bs, feature_maps_bs.shape))

    instance_n = nn.InstanceNorm2d(num_features=num_features, momentum=momentum)

    for i in range(1):
        outputs = instance_n(feature_maps_bs)

        print(outputs)
```

输出如下：

```
Instance Normalization
input data:
tensor([[[[1., 1.],
          [1., 1.]],
         [[2., 2.],
          [2., 2.]],
         [[3., 3.],
          [3., 3.]]],
        [[[1., 1.],
          [1., 1.]],
         [[2., 2.],
          [2., 2.]],
         [[3., 3.],
          [3., 3.]]],
        [[[1., 1.],
          [1., 1.]],
         [[2., 2.],
          [2., 2.]],
         [[3., 3.],
          [3., 3.]]]]) shape is torch.Size([3, 3, 2, 2])
tensor([[[[0., 0.],
          [0., 0.]],
         [[0., 0.],
          [0., 0.]],
         [[0., 0.],
          [0., 0.]]],
        [[[0., 0.],
          [0., 0.]],
         [[0., 0.],
          [0., 0.]],
         [[0., 0.],
          [0., 0.]]],
        [[[0., 0.],
          [0., 0.]],
         [[0., 0.],
          [0., 0.]],
         [[0., 0.],
          [0., 0.]]]])
```



# Group Normalization

提出的原因：在小 batch 的样本中，Batch Normalization 估计的值不准。一般用在很大的模型中，这时 batch size 就很小。

思路：数据不够，通道来凑。 每个样本的特征分为几组，每组特征分别计算均值和方差。可以看作是 Layer Normalization 的基础上添加了特征分组。

注意事项：

- 不再有 running_mean 和 running_var

- $\gamma$ 和 $\beta$ 为逐通道的

定义如下：

```
torch.nn.GroupNorm(num_groups, num_channels, eps=1e-05, affine=True)
```

参数：

- num_groups：特征的分组数量
- num_channels：特征数，通道数。注意 num_channels 要可以整除 num_groups
- eps：分母修正项
- affine：是否需要 affine transform

下面代码中，输入数据的形状是 $B \times C \times 2D\_feature$，(2, 4, 3, 3)，表示一个 mini-batch 有 2 个样本，每个样本有 4 个特征，每个特征的维度是 $3 \times 3 $。num_groups 设置为 2，那么就会计算  $2 \times (4 \div 2) $ 个均值和方差，分别对应每个样本的每个特征。

```
   batch_size = 2
    num_features = 4
    num_groups = 2   
    features_shape = (2, 2)

    feature_map = torch.ones(features_shape)    # 2D
    feature_maps = torch.stack([feature_map * (i + 1) for i in range(num_features)], dim=0)  # 3D
    feature_maps_bs = torch.stack([feature_maps * (i + 1) for i in range(batch_size)], dim=0)  # 4D

    gn = nn.GroupNorm(num_groups, num_features)
    outputs = gn(feature_maps_bs)

    print("Group Normalization")
    print(gn.weight.shape)
    print(outputs[0])
```

输出如下：

```
Group Normalization
torch.Size([4])
tensor([[[-1.0000, -1.0000],
         [-1.0000, -1.0000]],
        [[ 1.0000,  1.0000],
         [ 1.0000,  1.0000]],
        [[-1.0000, -1.0000],
         [-1.0000, -1.0000]],
        [[ 1.0000,  1.0000],
         [ 1.0000,  1.0000]]], grad_fn=<SelectBackward>)
```



**参考资料**

- [深度之眼 PyTorch 框架班](https://ai.deepshare.net/detail/p_5df0ad9a09d37_qYqVmt85/6)

<br>

如果你觉得这篇文章对你有帮助，不妨点个赞，让我有更多动力写出好文章。
<br>

我的文章会首发在公众号上，欢迎扫码关注我的公众号**张贤同学**。

<div align="center"><img src="https://image.zhangxiann.com/QRcode_8cm.jpg"/></div><br>










