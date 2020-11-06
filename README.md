# \(开篇词\)PyTorch 学习笔记

这篇文章是我学习 PyTorch 过程中所记录的学习笔记汇总，包括 **25** 篇文章，是我学习 **PyTorch** 框架版课程期间所记录的内容。

学习笔记的结构遵循课程的顺序，共分为 8 周，循序渐进，**力求通俗易懂**。

## 代码

配套代码：[https://github.com/zhangxiann/PyTorch\_Practice](https://github.com/zhangxiann/PyTorch_Practice)

所有代码均在 PyCharm 中通过测试，建议通过 git 克隆到本地运行。

## 数据

由于代码中会用到一些第三方的数据集，这里给出百度云的下载地址（如果有其他更好的数据托管方式，欢迎告诉我）。

数据下载地址： 链接：[https://pan.baidu.com/s/1f9wQM7gvkMVx2x5z6xC9KQ](https://pan.baidu.com/s/1f9wQM7gvkMVx2x5z6xC9KQ) 提取码：w7xt

## 面向读者

本教程假定读你有一定的机器学习和深度学习基础。

如果你没有学习过机器学习或者深度学习，建议先观看 Andrew ng 的深度学习（Deep Learning）课程，课程地址： [https://mooc.study.163.com/university/deeplearning\_ai\#/c](https://mooc.study.163.com/university/deeplearning_ai#/c)。

然后再学习本教程，效果会更佳。

## 学习计划

这个学习笔记共 25 章，分为 8 周进行的，每周大概 3 章（当然你可以根据自己的进度调整），每章花费的时间约 30 分钟到 2 个小时之间。

目录大纲如下：

* [\(开篇词\)PyTorch 学习笔记](README.md)
* [1 基本概念](1-ji-ben-gai-nian/README.md)
  * [1.1 PyTorch 简介与安装](1-ji-ben-gai-nian/1.1-pytorch-jian-jie-yu-an-zhuang.md)
  * [1.2 Tensor\(张量\)介绍](1-ji-ben-gai-nian/1.2-tensor-zhang-liang-jie-shao.md)
  * [1.3 张量操作与线性回归](1-ji-ben-gai-nian/1.3-zhang-liang-cao-zuo-yu-xian-xing-hui-gui.md)
  * [1.4 计算图与动态图机制](1-ji-ben-gai-nian/1.4-ji-suan-tu-yu-dong-tai-tu-ji-zhi.md)
  * [1.5 autograd 与逻辑回归](1-ji-ben-gai-nian/1.5-autograd-yu-luo-ji-hui-gui.md)
* [2 图片处理与数据加载](2-tu-pian-chu-li-yu-shu-ju-jia-zai/README.md)
  * [2.1 DataLoader 与 DataSet](2-tu-pian-chu-li-yu-shu-ju-jia-zai/2.1-dataloader-yu-dataset.md)
  * [2.2 图片预处理 transforms 模块机制](2-tu-pian-chu-li-yu-shu-ju-jia-zai/2.2-tu-pian-yu-chu-li-transforms-mo-kuai-ji-zhi.md)
  * [2.3 二十二种 transforms 图片数据预处理方法](2-tu-pian-chu-li-yu-shu-ju-jia-zai/2.3-er-shi-er-zhong-transforms-tu-pian-shu-ju-yu-chu-li-fang-fa.md)
* [3 模型构建](3-mo-xing-gou-jian/README.md)
  * [3.1 模型创建步骤与 nn.Module](3-mo-xing-gou-jian/3.1-mo-xing-chuang-jian-bu-zhou-yu-nn.module.md)
  * [3.2 卷积层](3-mo-xing-gou-jian/3.2-juan-ji-ceng.md)
  * [3.3 池化层、线性层和激活函数层](3-mo-xing-gou-jian/3.3-chi-hua-ceng-xian-xing-ceng-he-ji-huo-han-shu-ceng.md)
* [4 模型训练](4-mo-xing-xun-lian/README.md)
  * [4.1 权值初始化](4-mo-xing-xun-lian/4.1-quan-zhi-chu-shi-hua.md)
  * [4.2 损失函数](4-mo-xing-xun-lian/4.2-sun-shi-han-shu.md)
  * [4.3 优化器](4-mo-xing-xun-lian/4.3-you-hua-qi.md)
* [5 可视化与 Hook](5-ke-shi-hua-yu-hook/README.md)
  * [5.1 TensorBoard 介绍](5-ke-shi-hua-yu-hook/5.1-tensorboard-jie-shao.md)
  * [5.2 Hook 函数与 CAM 算法](5-ke-shi-hua-yu-hook/5.2-hook-han-shu-yu-cam-suan-fa.md)
* [6 正则化](6-zheng-ze-hua/README.md)
  * [6.1 weight decay 和 dropout](6-zheng-ze-hua/6.1-weight-decay-he-dropout.md)
  * [6.2 Normalization](6-zheng-ze-hua/6.2-normalization.md)
* [7 模型其他操作](7-mo-xing-qi-ta-cao-zuo/README.md)
  * [7.1 模型保存与加载](7-mo-xing-qi-ta-cao-zuo/7.1-mo-xing-bao-cun-yu-jia-zai.md)
  * [7.2 模型 Finetune](7-mo-xing-qi-ta-cao-zuo/7.2-mo-xing-finetune.md)
  * [7.3 使用 GPU 训练模型](7-mo-xing-qi-ta-cao-zuo/7.3-shi-yong-gpu-xun-lian-mo-xing.md)
* [8 实际应用](8-shi-ji-ying-yong/README.md)
  * [8.1 图像分类简述与 ResNet 源码分析](8-shi-ji-ying-yong/8.1-tu-xiang-fen-lei-jian-shu-yu-resnet-yuan-ma-fen-xi.md)
  * [8.2 目标检测简介](8-shi-ji-ying-yong/8.2-mu-biao-jian-ce-jian-jie.md)
  * [8.3 GAN（生成对抗网络）简介](8-shi-ji-ying-yong/8.3-gan-sheng-cheng-dui-kang-wang-luo-jian-jie.md)
  * [8.4 手动实现 RNN](8-shi-ji-ying-yong/8.4-shou-dong-shi-xian-rnn.md)
* [9 其他](9-qi-ta/README.md)
  * [PyTorch 常见报错信息](9-qi-ta/pytorch-chang-jian-bao-cuo-xin-xi.md)
  * [图神经网络 PyTorch Geometric 入门教程](9-qi-ta/tu-shen-jing-wang-luo-pytorch-geometric-ru-men-jiao-cheng.md)

如果这份 PyTorch 学习笔记对你有帮助，欢迎 star：

[https://github.com/zhangxiann/PyTorch\_Practice](https://github.com/zhangxiann/PyTorch_Practice)

如果你觉得这篇文章对你有帮助，不妨点个赞，让我有更多动力写出好文章。

欢迎扫码关注我的公众号**张贤同学**。

![](https://image.zhangxiann.com/QRcode_8cm.jpg)

