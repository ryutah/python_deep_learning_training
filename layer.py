class MulLayer:
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


class AddLayer:
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
