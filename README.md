# GoogLeNet_V1V2_frame


# 1. 相关文献
- 《Going deeper with convolutions》
- 《Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift》
- 《Rethinking the Inception Architecture for Computer Vision》
- 《Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning》

## 2. GoogLeNet_V1 Model Frame
![GoogLeNet](https://static.oschina.net/uploads/space/2018/0317/141544_FfKB_876354.jpg)

# 3. Inception
## 3.1 Inception_Basic
### 3.1.1 Inception_basic structure
![Inception Basic](https://static.oschina.net/uploads/space/2018/0317/141510_fIWh_876354.png)

## 3.2 Inception_V1
### 3.2.1 Inception_v1 structure
![Inception V1](https://static.oschina.net/uploads/space/2018/0317/141520_31TH_876354.png)

### 3.2.2 优缺点：
* 在3x3、5x5前，maxpool后分别加上1x1的卷积核，起到了降低特征图厚度的作用。
* 简单的网络堆叠，虽然可以提高准确率，但是会导致计算效率明显的下降。

### 3.2.3 GoogLeNet_V1 Model Parameters
![GoogLeNet参数](https://static.oschina.net/uploads/space/2018/0317/141605_c1XW_876354.png)

## 3.3 Inception_V2
### 3.3.1 Inception_v2 structure (middle-figure)
![img](https://img-blog.csdn.net/20160228155230994)

### 3.3.2 较V1改进点：
* 一方面了加入了BN层（在卷积与激活层间），减少了Internal Covariate Shift（内部neuron的数据分布发生变化），使每一层的输出都规范化到一个N(0, 1)的高斯
* 另外一方面学习VGG用2个3x3的conv替代inception模块中的5x5，既降低了参数数量，也加速计算。

### 3.3.3 GoogLeNet_V2 Model Parameters
![5.png](https://github.com/ShaoQiBNU/GoogleNet/blob/master/images/5.png?raw=true)


## 3.4 Inception_V3
### 3.3.1 改进点：
* 最重要的改进就是分解（Factorization），将7×7分解成两个一维卷积（1×7,7×1），3×3也是一样（1×3，3×1）。这样的好处是，既可以加速计算，又可以将一个卷积层拆分为两个卷积，使得网络深度进一步加深，增加了网络的非线性（每一层均需要ReLU）。



## 4. To do list...
Inception_V3, Inception_V4






