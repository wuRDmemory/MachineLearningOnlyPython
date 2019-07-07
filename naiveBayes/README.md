# 朴素贝叶斯估计

## 基本方法
对于训练数$T={(x_1, y_1), (x_2, y_2),..., (x_n, y_n)}$，其中x属于输入空间$x \in X$, 输出类别$y \in Y$, 朴素贝叶斯通过训练学习联合概率分布$P(X, Y)$.

具体的，由贝叶斯公式可知
$$
P(Y|X) = \frac{P(X|Y)P(Y)}{P(X)}
$$
若想通过输入变量$X$得到相应的类别$Y$，则需要知道输出变量$Y$的概率，条件概率以及输入类别$X$的概率。如果使用暴力的统计方法获得**条件概率**的话，公式如下：
$$
P(x=x|Y=c_k)=P(X^{(1)}=x^{(1)},.., X^{(n)}=x^{(n)}|Y=c_k)
$$
那么对于一个n维的输入变量x，其统计的情况会有$K\prod_{j=1}^{n}S_{j}$种，其中K为输出标签$y$的数量，$S_j$表示每一维特征能够取到多少种情况。


朴素贝叶斯方法做了一个比较强的假设：即对条件概率分布做了条件独立性假设，按照公式即：
$$
\begin{aligned}
P(x=x|Y=c_k)&=P(X^{(1)}=x^{(1)},.., X^{(n)}=x^{(n)}|Y=c_k) \\
&=\prod_{j=1}^{n}P(X^{(j)}=x^{(j)}|Y=c_k)
\end{aligned}
$$

可以看到使用这样的假设之后，整个通过过程变为了$K\sum_{j=1}^{n}S_{j}$，大大减少了运算量。

&nbsp;
## 公式推导
首先通过贝叶斯公式：
$$
\begin{align}
P(Y=c_k|X=x)&= \frac{P(X=x|Y=c_k)P(Y=c_k)}{\sum_{k}P(X=x|Y=c_k)P(Y=c_k)} \\
&= \frac{\prod_{j=1}^{n}{P(X^{(j)}=x^{(j)}|Y=c_k)}P(Y=c_k)}{\sum_{k}P(X^{(j)}=x^{(j)}|Y=c_k)P(Y=c_k)}
\end{align}
$$

可以通过训练集的数据得到k个概率，每个概率表示x属于第k个个类的概率，其中最大的类就是当前输入所属的类别，即：
$$
y=f(x)=\arg \max _{c_{k}} \frac{P\left(Y=c_{k}\right) \prod_{j} P\left(X^{(j)}=x^{(j)} | Y=c_{k}\right)}{\sum_{k} P\left(Y=c_{k}\right) \prod_{j} P\left(X^{(j)}=x^{(j)} | Y=c_{k}\right)}
$$

其中分母部分对于每个类而言都是一样的，因此这部分可以省去，因此可以得到：
$$
y=\arg \max _{q} P\left(Y=c_{k}\right) \prod_{j} P\left(X^{(j)}=x^{(j)} | Y=c_{k}\right)
$$

可以看到，本质而言，贝叶斯方法是生成方法，其学习的模型为$P(X, Y)$.

&nbsp;
## 最大后验的含义
朴素贝叶斯方法是将类别分在后验最大的类中（其中先验是类的概率分布），这等价于期望风险（也即泛化误差）最小化。

假设我们选择0-1损失函数
$$
L(Y, f(X))=\left\{\begin{array}{ll}{1,} & {Y \neq f(X)} \\ {0,} & {Y=f(X)}\end{array}\right. 
$$

作为损失函数的话，那么期望风险函数就是：
$$
R_{\exp }(f)=E_{p}[L(Y, f(X))]
$$

其中期望得对联合概率$P(x, y)$求取得，这里展开就得到：
$$
\begin{align}
R_{\exp }(f)&=E_{p}[L(Y, f(X))] \\
&=\sum_{x,y}L(Y, f(X))P(x, y) \\
&=\sum_{x,y}L(Y, f(X))P(y|x)p(x) \\
&=E_x[\sum_{y}L(Y, f(X))P(y|x)] \\
&=E_x[\sum_{k}L(c_k, f(X))P(c_k|x)]
\end{align}
$$

所以，为了期望风险最小化，我们可以使得每一项都达到最小，即：
$$
\begin{align}
f(x)&=\mathop{\arg\min}_{f(x)} \sum_{k}L(c_k, f(x))P(c_k|x) \\
&=\mathop{\arg\min}_{f(x)} \sum_{k}P(c_k \neq f(x)|x) \\
&=\mathop{\arg\min}_{f(x)} \sum_{k}(1-P(c_k = f(x)|x)) \\
&=\mathop{\arg\max}_{f(x)} \sum_{k}P(c_k = f(x)|x)
\end{align}
$$

所以可以看到，经过变换，期望风险最小化最终可以写作最大化$P(Y=y|X=x)$的形式，也就是最大化后验概率。

