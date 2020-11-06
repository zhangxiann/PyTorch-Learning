---
thumbnail: https://image.zhangxiann.com/andrea-hagenhoff-dIXCt2zUzV0-unsplash.jpg
toc: true
date: 2020/4/23 18:55:20
disqusId: zhangxian
categories:
- PyTorch

tags:
- AI
- Deep Learning
---

> 本章代码：
>
> - [https://github.com/zhangxiann/PyTorch_Practice/blob/master/lesson8/gan_inference.py](https://github.com/zhangxiann/PyTorch_Practice/blob/master/lesson8/gan_inference.py)
>
> - [https://github.com/zhangxiann/PyTorch_Practice/blob/master/lesson8/gan_demo.py](https://github.com/zhangxiann/PyTorch_Practice/blob/master/lesson8/gan_demo.py)

这篇文章主要介绍了生成对抗网络（Generative Adversarial Network），简称 GAN。

GAN 可以看作是一种可以生成特定分布数据的模型。

下面的代码是使用 Generator 来生成人脸图像，Generator 已经训练好保存在 pkl 文件中，只需要加载参数即可。由于模型是在多 GPU 的机器上训练的，因此加载参数后需要使用`remove_module()`函数来修改`state_dict`中的`key`。<!--more-->

```
def remove_module(state_dict_g):
    # remove module.
    from collections import OrderedDict

    new_state_dict = OrderedDict()
    for k, v in state_dict_g.items():
        namekey = k[7:] if k.startswith('module.') else k
        new_state_dict[namekey] = v

    return new_state_dict
```

把随机的高斯噪声输入到模型中，就可以得到人脸输出，最后进行可视化。全部代码如下：

```
import os
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
from common_tools import set_seed
from torch.utils.data import DataLoader
from my_dataset import CelebADataset
from dcgan import Discriminator, Generator
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def remove_module(state_dict_g):
    # remove module.
    from collections import OrderedDict

    new_state_dict = OrderedDict()
    for k, v in state_dict_g.items():
        namekey = k[7:] if k.startswith('module.') else k
        new_state_dict[namekey] = v

    return new_state_dict

set_seed(1)  # 设置随机种子

# config
path_checkpoint = os.path.join(BASE_DIR, "gan_checkpoint_14_epoch.pkl")
image_size = 64
num_img = 64
nc = 3
nz = 100
ngf = 128
ndf = 128

d_transforms = transforms.Compose([transforms.Resize(image_size),
                   transforms.CenterCrop(image_size),
                   transforms.ToTensor(),
                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
               ])

# step 1: data
fixed_noise = torch.randn(num_img, nz, 1, 1, device=device)

flag = 0
# flag = 1
if flag:
    z_idx = 0
    single_noise = torch.randn(1, nz, 1, 1, device=device)
    for i in range(num_img):
        add_noise = single_noise
        add_noise = add_noise[0, z_idx, 0, 0] + i*0.01
        fixed_noise[i, ...] = add_noise

# step 2: model
net_g = Generator(nz=nz, ngf=ngf, nc=nc)
# net_d = Discriminator(nc=nc, ndf=ndf)
checkpoint = torch.load(path_checkpoint, map_location="cpu")

state_dict_g = checkpoint["g_model_state_dict"]
state_dict_g = remove_module(state_dict_g)
net_g.load_state_dict(state_dict_g)
net_g.to(device)
# net_d.load_state_dict(checkpoint["d_model_state_dict"])
# net_d.to(device)

# step3: inference
with torch.no_grad():
    fake_data = net_g(fixed_noise).detach().cpu()
img_grid = vutils.make_grid(fake_data, padding=2, normalize=True).numpy()
img_grid = np.transpose(img_grid, (1, 2, 0))
plt.imshow(img_grid)
plt.show()
```

输出如下：

<div align="center"><img src="https://image.zhangxiann.com/20200709232923.png"/></div><br>
下面对 GAN 的网络结构进行讲解

<div align="center"><img src="https://image.zhangxiann.com/20200709233026.png"/></div><br>
Generator 接受随机噪声$z$作为输入，输出生成的数据$G(z)$。Generator 的目标是让生成数据和真实数据的分布越接近。Discriminator 接收$G(z)$和随机选取的真实数据$x$，目标是分类真实数据和生成数据，属于 2 分类问题。Discriminator 的目标是把它们二者之间分开。这里体现了对抗的思想，也就是 Generator 要欺骗 Discriminator，而 Discriminator 要识别 Generator。



# GAN 的训练和监督学习训练模式的差异

在监督学习的训练模式中，训练数经过模型得到输出值，然后使用损失函数计算输出值与标签之间的差异，根据差异值进行反向传播，更新模型的参数，如下图所示。

<div align="center"><img src="https://image.zhangxiann.com/20200710101149.png"/></div><br>
在 GAN 的训练模式中，Generator 接收随机数得到输出值，目标是让输出值的分布与训练数据的分布接近，但是这里不是使用人为定义的损失函数来计算输出值与训练数据分布之间的差异，而是使用 Discriminator 来计算这个差异。需要注意的是这个差异不是单个数字上的差异，而是分布上的差异。如下图所示。

<div align="center"><img src="https://image.zhangxiann.com/20200710102929.png"/></div><br>
# GAN 的训练

1. 首先固定 Generator，训练 Discriminator。

   - 输入：真实数据$x$，Generator 生成的数据$G(z)$
   - 输出：二分类概率

   从噪声分布中随机采样噪声$z$，经过 Generator 生成$G(z)$。$G(z)$和$x$输入到 Discriminator 得到$D(x)$和$D(G(z))$，损失函数为$\frac{1}{m} \sum_{i=1}^{m}\left[\log D\left(\boldsymbol{x}^{(i)}\right)+\log \left(1-D\left(G\left(\boldsymbol{z}^{(i)}\right)\right)\right)\right]$，这里是最大化损失函数，因此使用梯度上升法更新参数：$$\nabla_{\theta_{d}} \frac{1}{m} \sum_{i=1}^{m}\left[\log D\left(\boldsymbol{x}^{(i)}\right)+\log \left(1-D\left(G\left(\boldsymbol{z}^{(i)}\right)\right)\right)\right]$$。

2. 固定 Discriminator，训练 Generator。

   - 输入：随机噪声$z$
   - 输出：分类概率$D(G(z))$，目的是使$D(G(z))=1$

   从噪声分布中重新随机采样噪声$z$，经过 Generator 生成$G(z)$。$G(z)$输入到 Discriminator 得到$D(G(z))$，损失函数为$\frac{1}{m} \sum_{i=1}^{m} \log \left(1-D\left(G\left(z^{(i)}\right)\right)\right)$，这里是最小化损失函数，使用梯度下降法更新参数：$\nabla_{\theta_{g}} \frac{1}{m} \sum_{i=1}^{m} \log \left(1-D\left(G\left(z^{(i)}\right)\right)\right)$。



下面是 DCGAN 的例子，DC 的含义是 Deep Convolution，指 Generator 和 Discriminator 都是卷积神经网络。

Generator 的网络结构如下图左边，使用的是 transpose convolution，输入是 100 维的随机噪声$z$，形状是$(1,100,1,1)$，看作是 100 个 channel，每个特征图宽高是$1 \times 1$；输出是$(3,64,64)$的图片$G(z)$。

Generator 的网络结构如下图右边，使用的是 convolution，输入是$G(z)$或者真实图片$x$，输出是 2 分类概率。

<div align="center"><img src="https://image.zhangxiann.com/20200710104917.png"/></div><br>
使用数据集来源于 CelebA 人脸数据：http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html，有 22 万张人脸图片数据。

但是由于人脸在图片中的角度、位置、所占区域大小等都不一样。如下图所示。

<div align="center"><img src="https://image.zhangxiann.com/20200710105813.png"/></div><br>
需要对关键点检测算法对人脸在图片中的位置和大小等进行矫正。下图是矫正后的数据。

<div align="center"><img src="https://image.zhangxiann.com/20200710105836.png"/></div><br>
人脸矫正数据的下载地址：https://pan.baidu.com/s/1OhE_ITg3Je4ETECm74VfRA，提取码：yarv。



在对图片进行标准化时，经过`toTensor()`转换到$[0,1]$后，把`transforms.Normalize()`的均值和标准差均设置为 0.5，这样就把数据转换为到$[-1,1]$区间，因为$((0,1)-0.5)/0.5=(-1,1)$。

DCGAN 的定义如下：

```
from collections import OrderedDict
import torch
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, nz=100, ngf=128, nc=3):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        return self.main(input)

    def initialize_weights(self, w_mean=0., w_std=0.02, b_mean=1, b_std=0.02):
        for m in self.modules():
            classname = m.__class__.__name__
            if classname.find('Conv') != -1:
                nn.init.normal_(m.weight.data, w_mean, w_std)
            elif classname.find('BatchNorm') != -1:
                nn.init.normal_(m.weight.data, b_mean, b_std)
                nn.init.constant_(m.bias.data, 0)


class Discriminator(nn.Module):
    def __init__(self, nc=3, ndf=128):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)

    def initialize_weights(self, w_mean=0., w_std=0.02, b_mean=1, b_std=0.02):
        for m in self.modules():
            classname = m.__class__.__name__
            if classname.find('Conv') != -1:
                nn.init.normal_(m.weight.data, w_mean, w_std)
            elif classname.find('BatchNorm') != -1:
                nn.init.normal_(m.weight.data, b_mean, b_std)
                nn.init.constant_(m.bias.data, 0)

```

其中`nz`是输入的通道数 100，`ngf`表示最后输出的图片宽高，这里设置为 64，会有一个倍数关系, `nc`是最后的输出通道数 3。

在迭代训练时，首先根据 DataLoader 获得的真实数据的 batch size，构造真实的标签 1；

然后随机生成噪声，构造生成数据的标签 0。把噪声输入到 Generator 中，得到生成数据。

分别把生成数据和真实数据输入到 Discriminator，得到两个 Loss，分别求取梯度相加，然后使用 Discriminator  的优化器更新 Discriminator 的参数。

然后生成数据的标签改为 1，输入到 Generator，求取梯度，这次使用 Generator 的优化器更新 Generator 的参数。

这里只使用了 2000 张图片来训练 20 个 epoch，下图是每个 epoch 生成的数据的可视化。



效果不是很好，可以使用几个 trick 来提升生成图片的效果。

- 使用 22 万张图片
- `ngf`设置为 128
- 标签平滑：真实数据的标签设置为 0.9，生成数据的标签设置为 0.1





# GAN 的一些应用

- 生成人的不同姿态

  <div align="center"><img src="https://image.zhangxiann.com/20200710120508.png"/></div><br>

- CycleGAN：对一个风格的图片转换为另一个风格

  <div align="center"><img src="https://image.zhangxiann.com/20200710120632.png"/></div><br>

- PixelDTGAN：通过一件衣服生成相近的衣服

  <div align="center"><img src="https://image.zhangxiann.com/20200710131528.png"/></div><br>

- SRGAN：根据模糊图像生成超分辨率的图像

  <div align="center"><img src="https://image.zhangxiann.com/20200710131625.png"/></div><br>

- Progressive GAN：生成高分辨率的人脸图像

  <div align="center"><img src="https://image.zhangxiann.com/20200710131720.png"/></div><br>

- StackGAN：根据文字生成图片

  <div align="center"><img src="https://image.zhangxiann.com/20200710131800.png"/></div><br>

- Context Encoders：补全图片中缺失的部分

  <div align="center"><img src="https://image.zhangxiann.com/20200710131853.png"/></div><br>

- Pix2Pix：也属于图像风格迁移

  <div align="center"><img src="https://image.zhangxiann.com/20200710131959.png"/></div><br>

- IcGAN：控制生成人脸的条件，如生成的人脸的头发颜色、是否戴眼镜等

<div align="center"><img src="https://image.zhangxiann.com/20200710132118.png"/></div><br>

**参考资料**

- [深度之眼 PyTorch 框架班](https://ai.deepshare.net/detail/p_5df0ad9a09d37_qYqVmt85/6)


<br>

如果你觉得这篇文章对你有帮助，不妨点个赞，让我有更多动力写出好文章。
<br>

我的文章会首发在公众号上，欢迎扫码关注我的公众号**张贤同学**。

<div align="center"><img src="https://image.zhangxiann.com/QRcode_8cm.jpg"/></div><br>







