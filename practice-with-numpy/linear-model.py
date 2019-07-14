import numpy as np
import sys

class LinearRegression:
    """
    原理：
    线性模型用向量形式表示为
        f(x) = w1x1 + w2x2 + …… + wdxd + b = (w)T·x +b
    w为参数向量w(w1;w2;……;wd)。
    下面从仅有一个变量，w是一个参数的情况开始讨论。
    线性回归(Linear Regression)试图从数据集
        D = {(x1,y1),……,(xm,ym)}
    中学得一个能尽可能准确预测实值输出的线性模型。
    确定w、b的过程需要衡量f(x)与y的差别，一般使用均方误差(square loss)并尝试使其最小化。
    均方误差与几何中的欧几里得距离对应，基于均方误差最小化的模型求解方法就是最小二乘法(least square loss)。
        (w*,b*) = argmin(w,b)Σ|i=1,m| (f(xi) - yi)^2 = argmin(w,b)Σ|i=1,m| (yi - wxi -b)^2
    求解w,b的过程称为最小二乘“参数估计”。
    考虑均方误差的期望E(w,b) = Σ|i=1,m| (yi - wxi -b)^2，该函数显然在O-xyz空间内是存在最小值的凹曲面，
    所以对E(w,b)求偏导，使∂E/∂w=0,∂E/∂b=0，即可求得使均方误差最小的(w,b):
        w = (Σ|i=1,m| yi(xi - avg(x)))/(Σ|i=1,m| xi^2 - 1/m*(Σ|i=1,m|xi)^2)
        b = 1/m*Σ|i=1,m| (yi - wxi)
    以上为一元线性回归的情况，对于多元线性回归，将w,b表示为一个向量<w>=(w;b)，将数据集D表示为
        X = [(D)T;1] = [[x1.T 1],[x2.T 1],……,[xm.T 1]], y = (y1;y2;……;ym)
    参考一元的情况，得<w>* = argmin R^2 = argmin (y - X·<w>).T·(y - X·<w>)
                    # 相当于对(y - X·<w>)每个元素求平方
    再对E<w>求关于<w>的偏导，使其为零即得最优闭式解：
        ∂E<w>/∂<w> = 2X.T · (X·<w> - y)
    当(X)T·X为满秩矩阵或正定矩阵时，利用矩阵的逆即可求出
        <w>* = (X.T · X)^(-1) · X.T · y
        代入<xi> = (xi,1)，即得
        f(xi) = <xi>.T · (X.T · X)^(-1) · X.T · y
    但是一般(X)T·X不是满秩矩阵，这时就会有多个<w>的解，一般由学习算法的归纳偏好决定。
    对线性回归模型进行非线性映射，可得到对数线性回归乃至广义线性回归。

    实现说明：
    矩阵运算,求<w>* = (X.T · X)^(-1) · X.T · y
    np.linalg :numpy的线性代数模块，其中的inv()方法可以求矩阵的逆
    """
    def __init__(self, fit_intercept=True):
        self.parameter = None
        self.fit_intercept = fit_intercept

    def fit_model(self, X, y):
        # 将X转换为设计矩阵(design matrix)
        if self.fit_intercept:
            X = np.c_[np.ones(X.shape[0]), X]
        mid = np.dot(np.linalg.inv(np.dot(X.T, X)), X.T)
        self.parameter = mid.dot(y)

    def predict(self, X):
        if self.fit_intercept:
            X = np.c_[np.ones(X.shape[0]), X]
        return np.dot(X, self.parameter)

class LogisticRegression:
    """
    原理：
    逻辑回归，即对数几率回归。
    """
