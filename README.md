---
date: 2020/6/1 21:26:20
disqusId: zhangxian
categories:
- PyTorch

tags:
- AI
- Deep Learning
---



# PyTorch 学习笔记

这篇文章是我学习 PyTorch 过程中所记录的学习笔记汇总，包括 **25** 篇文章，是我学习 **PyTorch** 框架版课程期间所记录的内容。

学习笔记的结构遵循课程的顺序，共分为 8 周，循序渐进，**力求通俗易懂**。



## 代码

配套代码：[https://github.com/zhangxiann/PyTorch_Practice](https://github.com/zhangxiann/PyTorch_Practice)

所有代码均在 PyCharm 中通过测试，建议通过 git 克隆到本地运行。

<!--more-->



## 数据

由于代码中会用到一些第三方的数据集，这里给出百度云的下载地址（如果有其他更好的数据托管方式，欢迎告诉我）。

数据下载地址：
链接：https://pan.baidu.com/s/1f9wQM7gvkMVx2x5z6xC9KQ 
提取码：w7xt



## 面向读者

本教程假定读你有一定的机器学习和深度学习基础。

如果你没有学习过机器学习或者深度学习，建议先观看 Andrew ng 的深度学习（Deep Learning）课程，课程地址： [https://mooc.study.163.com/university/deeplearning_ai#/c](https://mooc.study.163.com/university/deeplearning_ai#/c)。

然后再学习本教程，效果会更佳。



## 学习计划

这个学习笔记共 25 章，分为 8 周进行的，每周大概 3 章（当然你可以根据自己的进度调整），每章花费的时间约 30 分钟到 2 个小时之间。

目录大纲如下：

- **Week 1（基本概念）**
  - [1.1 PyTorch 简介与安装](https://blog.zhangxiann.com/202002022039/)
  - [1.2 Tensor(张量)介绍](https://blog.zhangxiann.com/202002052039/)
  - [1.3 张量操作与线性回归](https://blog.zhangxiann.com/202002082037/)
  - [1.4 计算图与动态图机制](https://blog.zhangxiann.com/202002112035/)
  - [1.5 autograd 与逻辑回归](https://blog.zhangxiann.com/202002152033/)
- **Week 2（图片处理与数据加载）**
  - [2.1 DataLoader 与 DataSet](https://blog.zhangxiann.com/202002192017/)
  - [2.2 图片预处理 transforms 模块机制](https://blog.zhangxiann.com/202002212045/)
  - [2.3 二十二种 transforms 图片数据预处理方法](https://blog.zhangxiann.com/202002272047/)
- **Week 3（模型构建）**
  - [3.1 模型创建步骤与 nn.Module](https://blog.zhangxiann.com/202003012001/)
  - [3.2 卷积层](https://blog.zhangxiann.com/202003032009/)
  - [3.3 池化层、线性层和激活函数层](https://blog.zhangxiann.com/202003072007/)
- **Week 4（模型训练）**
  - [4.1 权值初始化](https://blog.zhangxiann.com/202003092013/)
  - [4.2 损失函数](https://blog.zhangxiann.com/202003132033/)
  - [4.3 优化器](https://blog.zhangxiann.com/202003172017/)
- **Week 5（可视化与 Hook）**
  - [5.1 TensorBoard 介绍](https://blog.zhangxiann.com/202003192045/)
  - [5.2 Hook 函数与 CAM 算法](https://blog.zhangxiann.com/202003232051/)
- **Week 6（正则化）**
  - [6.1 weight decay 和 dropout](https://blog.zhangxiann.com/202003272049/)
  - [6.2 Normalization](https://blog.zhangxiann.com/202004011919/)
- **Week 7（模型其他操作）**
  - [7.1 模型保存与加载](https://blog.zhangxiann.com/202004051903/)
  - [7.2 模型 Finetune](https://blog.zhangxiann.com/202004091911/)
  - [7.3 使用 GPU 训练模型](https://blog.zhangxiann.com/202004151915/)
- **Week 8（实际应用）**
  - [8.1 图像分类简述与 ResNet 源码分析](https://blog.zhangxiann.com/202004171947/)
  - [8.2 目标检测简介](https://blog.zhangxiann.com/202004211903/)
  - [8.3 GAN（生成对抗网络）简介](https://blog.zhangxiann.com/202004231855/)
  - [8.4 手动实现 RNN](https://blog.zhangxiann.com/202004271841/)



如果这份 PyTorch 学习笔记对你有帮助，欢迎 star：

https://github.com/zhangxiann/PyTorch_Practice

<br>

如果你觉得这篇文章对你有帮助，不妨点个赞，让我有更多动力写出好文章。



<br>

欢迎扫码关注我的公众号**张贤同学**。

<div align="center"><img src="https://image.zhangxiann.com/QRcode_8cm.jpg"/></div><br>
