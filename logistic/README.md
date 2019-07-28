# Logistic回归

## 公式推导
### Sigmoid函数
Sigmoid函数公式如下
$$
S(x) = \frac{1}{1+e^{-x}}
$$
该函数可以把数值映射到0~1的区间，函数的图像如下:
<img src="imgs/2019-07-27 18-32-08.png">

可以看到在x变化较大的情况下sigmoid函数像0-1函数一样。

&nbsp;
---
## 最大似然法推导（二分类）
设$P(y_i=1|x;w)=S(wx)$，则模型的输出有：

1. 当$y_i=1$时，模型的输出为$P(y_i=1|x;w)^{y_i}$
2. 当$y_i=0$时，模型的输出为$(1-P(y_i=1|x;w))^{(1-y_i)}$

所以对于$y_i$，模型的输出可以简化为
$$
P(y_i|x)=P(y_i=1|x;w)^{y_i}(1-P(y_i=1|x;w))^{(1-y_i)}
$$

对于最大似然法，需要使得下面公式达到最大
$$
P(y|x) = \prod_{i=1}^{N}P(y_i|x)
$$

但是因为概率都是0~1的，连乘之后的值可能会下溢，所以这里取概率的log来将概率连乘变为数值连加，所以最大似然公式变为：
$$
L(w) = logP(y|x)=\sum_{i=1}^{N}(y_ilogP(y_i|x;w)+(1-y_i)log(1-P(y_i|x;w)))
$$

为了求得概率似然的最大值，这里使用梯度上升法求解，求解L(w)关于模型参数w的梯度：
$$
\frac{\partial{L}}{\partial{w}} = \sum_{i=1}^{N}(\frac{y_i}{P(y_i=1|x)}\frac{\partial{P}}{\partial{w}}-\frac{1-y_i}{1-P(y_i=1|x)}\frac{\partial{P}}{\partial{w}})
$$

对于模型参数的更新公式：
$$
w = w + \alpha \delta{w}
$$

因为$P=S(wx)$，是sigmoid函数，对该函数求导可得：
$$
\frac{\partial{P}}{\partial{w}}=(1-P)Px
$$

带入到梯度公式中就可以得到更新量：
$$
\frac{\partial{L}}{\partial{w}} = \sum_{i=1}^{N}(y_i-s(wx_i))x_i
$$

&nbsp;
---
## logistic对权重的更新
在logistic的更新中，正确的分类和错误的分类都是会对决策平面起到修正作用，例如：

初始状态下的$w=(0, 0，0), x=(1, 2，1)$，此时模型的输出为$s(wx)=0.5$，模型预测为正例，下面考虑两种情况：
1. $y_i=1$, 此时模型预测的是正确的，则$w_{new}=(0,0)+\alpha*(1-0.5)*(1,2)=(1,2)$，更新之后模型的输出为：$s(wx)=\frac{1}{1+e^5} \approx1$
2. $y_i=0$, 此时模型预测的是错误的，则$w_{new}=(0,0)+\alpha*(0-0.5)*(1,2)=(-1,-2)$，更新之后模型的输出为：$s(wx)=\frac{1}{1+e^{-5}} \approx0$







