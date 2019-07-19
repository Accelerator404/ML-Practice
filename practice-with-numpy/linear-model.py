import numpy as np
import sys


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


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
    当我们想要使用线性模型进行分类任务时，观察广义线性模型
        y = g^(-1)(w.T · x + b)   # g^(-1)(·)为单调可微函数，称为联系函数(link function)
    可知，只要找一个合适的联系函数，将分类任务的真实标记y与线性回归模型的预测值联系起来。
    二分类任务可以考虑单位阶跃函数(unit-step function)，
    但是由于它在y=f(0)处不连续，为此使用近似的对数几率函数(logistic function)作为替代函数
        y = 1/(1 + e^(-z))
    它是一个Sigmoid函数，将它作为g^(-1)(·)代入广义线性模型，即得
        y = 1/(1 + e^(-(w.T · x + b)))      ## 式2-1
        --> 1 + e^(-(w.T · x + b)) = 1/y
        --> (y - 1)/y = e^(-(w.T · x + b))
        --> y/(y - 1) = e^(w.T · x + b)
        --> ln(y/(y - 1)) = w.T · x + b     ## 式2-2
    对于变形后的该式，如果将y视为样本x作为正例的可能性，则1-y就是反例可能性，则称y/(1-y)是几率(odds)，反映x是正例的相对可能性。
    称ln(y/(1-y))为对数几率(log odds)，便可看出上式的意义：用线性回归模型的预测结果去逼近真实标记的对数几率。
    因此其对应的模型称为对数几率回归(logistic regression)。
    下面求解w和b。
    将y视为类后验概率p(y=1|x),则将式2-1改写，得到
        p(y=1|x)/p(y=0|x) = e^(w.T · x + b)
    又p(y=1|x) + p(y=0|x) = 1，故
        p(y=1|x) = e^(w.T · x + b)/(1 + e^(w.T · x + b))
        p(y=0|x) = 1/(1 + e^(w.T · x + b))
    运用最大似然估计便可估计出w,b的值，为了便于计算，基于数据集{(xi,yi)}|m,i=1使用“对数似然”(log-likelihood)
        l(w,b) = Σ|m,i=1 ln(p(yi|xi;w,b))  ## 式2-3
    令β(w;b), <x> = (x;1)，则(w.T · x + b)可简写为β.T·<x>，
    再令p1(<x>;β) = p(y=1|<x>;β)，p0(<x>;β) = p(y=0|<x>;β) = 1 - p1(<x>;β)，则可重写式2-3中的似然项为
        p(yi|xi;w,b) = yi·p1(<x>;β) + (1-yi)p0(<x>;β)
    代入式2-3，得
        l(w,b) = Σ|m,i=1 ln(yi·p1(<x>;β) + (1-yi)p0(<x>;β))
    而，将(w.T · x + b)简写为β.T·<x>后
        p1(<x>;β) = e^(β.T·<x>)/(1 + e^(β.T·<x>))
        p0(<x>;β) = 1/(1 + e^(β.T·<x>))
    考虑每一个似然项，使2-3最大，就是使下式最小：
        l(β) = Σ|m,i=1 ln((1 + e^(β.T·<x>))/(yi·e^(β.T·<x>)))
             = Σ|m,i=1 (-yi·β.T·<x> + ln(1 + e^(β.T·<x>)))
    于是根据凸优化理论，梯度下降法或者牛顿法均可求出上式的最优解：
        β* = argmin|β l(β)

    实现：
    根据上面的分析，可知通过梯度下降(gradient descent)就可以求出逻辑回归的参数的最优解。
    根据梯度的定义，以负梯度向量方向进行移动时，f下降最快。为了确定移动的步长，定义一个正标量：学习率，用η表示。
    所以新的最速下降建议点可以表示为x' = x - η▽xf(x),▽xf(x)是f(x)对向量x的梯度。
    正则化与参数范数惩罚方面，此处考虑L1参数正则化与L2参数正则化两种方法（默认L2）。
    """
    def __init__(self, penalty_type="l2", regulation_weight=0, fit_intercept=True):
        '''
        :param penalty_type: 参数范数惩罚方式
        :param regulation_weight: 惩罚的力度，取值0~1的浮点数
        :param fit_intercept: 输入的系数矩阵是否多一个截距项b
        '''
        err_msg = "The penalty type must be 'l1' or 'l2'."
        assert penalty_type in ["l2", "l1"], err_msg
        self.regulation_weight = regulation_weight
        self.beta = None
        self.penalty_type = penalty_type
        self.fit_intercept = fit_intercept

    def fit(self, X, y, learning_rate=0.01, tol=1e-7, max_iter=1e7):
        '''
        :param X:训练样本矩阵
        :param y:训练结果样本向量
        :param learning_rate:学习率
        :param tol:残差收敛条件，又称容忍度，小于该值时认为两个数'差不多'
        :param max_iter:最大迭代次数
        '''
        # 如果fit intercept,将X转换为设计矩阵（操作上就是在矩阵最右侧插入一列1）
        if self.fit_intercept:
            X = np.c_[np.ones(X.shape[0]), X]

        l_prev = np.inf
        # 随机生成一个初始参数组合
        self.beta = np.random.rand(X.shape[1])
        for _ in range(int(max_iter)):
            # 用学习到的参数组合计算y_pred
            y_pred = sigmoid(np.dot(X, self.beta))
            # 提前终止
            loss = self._NLL(X, y, y_pred)
            if l_prev - loss < tol:
                return
            l_prev = loss
            # 将这一次训练的成果加上学习率和参数范数惩罚后再保存
            self.beta -= learning_rate * self._NLL_grad(X, y, y_pred)

    def _NLL(self, X, y, y_pred):
        # negative log likelihood
        N, M = X.shape
        order = 2 if self.penalty_type == "l2" else 1
        nll = -np.log(y_pred[y == 1]).sum() - np.log(1 - y_pred[y == 0]).sum()
        penalty = 0.5 * self.regulation_weight * np.linalg.norm(self.beta, ord=order) ** 2
        return (penalty + nll) / N

    def _NLL_grad(self, X, y, y_pred):
        """
        Gradient of the penalized negative log likelihood wrt beta
        """
        N, M = X.shape
        p = self.penalty_type
        beta = self.beta
        weight = self.regulation_weight
        l1norm = lambda x: np.linalg.norm(x, 1)
        d_penalty = weight * beta if p == "l2" else weight * l1norm(beta) * np.sign(beta)
        return -(np.dot(y - y_pred, X) + d_penalty) / N

    def predict(self, X):
        # convert X to a design matrix if we're fitting an intercept
        if self.fit_intercept:
            X = np.c_[np.ones(X.shape[0]), X]
        return sigmoid(np.dot(X, self.beta))

