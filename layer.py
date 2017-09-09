import numpy as np


class MulLayer(object):
    """
    乗算レイヤ
    """
    def __init__(self):
        self.x = None
        self.y = None

    def forward(self, x, y):
        """
        順伝搬
        """
        self.x = x
        self.y = y
        out = x * y
        return out

    def backward(self, dout):
        """
        逆伝搬
        """
        # xとyをひっくり返す
        dx = dout * self.y
        dy = dout * self.x
        return dx, dy


class AddLayer(object):
    """
    加算レイヤ
    """
    def __init__(self):
        pass

    def forward(self, x, y):
        """
        順伝搬
        """
        out = x + y
        return out

    def backward(self, dout):
        dx = dout * 1
        dy = dout * 1
        return dx, dy


class Relu(object):
    """
    活性化関数として使われるReLU(Rectified Linear Unit)レイヤ
    """
    def __init__(self):
        self.mask = None

    def forward(self, x):
        self.mask = (x <= 0)
        out = x.copy
        out[self.mask] = 0
        return out

    def backwark(self, dout):
        dout[self.mask] = 0
        dx = dout
        return dx


class Sigmoid(object):
    """
    活性化関数として使われるSigmoidレイヤ
    """
    def __init__(self):
        self.out = None

    def forward(self, x):
        out = 1 / 1 + np.exp(-x)
        self.out = out
        return out

    def backwark(self, dout):
        dx = dout * (1.0 - self.out) * self.out
        return dx


class Affine(object):
    """
    ニューララルネットワークの順伝搬で行う行列の積(アフィン変換)
    のレイヤ
    """
    def __init__(self, W, b):
        self.W = W
        self.b = b
        self.x = None
        self.dW = None
        self.db = None

    def forward(self, x):
        self.x = x
        out = np.dot(x, self.W) + self.b
        return out

    def backwark(self, dout):
        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)
        return dx
