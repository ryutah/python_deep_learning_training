import numpy as np

from common.functions import *
from common.gradieng import numerical_gradient


class TwoLayerNet(object):
    """
    確率的勾配降下方を使用したニューラルネットワーク
    ニューラルネットワークの層は2層

    * 適応可能な重みとバイアスを持つ
    * 重みとバイアスを訓練データに適応するように調整することを「学習」と呼ぶ
    * ニューラルネットワークの学習は下記の4つの手順で行う

    ステップ1 (ミニバッチ)
    訓練データの中からランダムに一部のデータを選ぶ(ミニバッチ)
    ミニバッチの値を減らすことを目的とする

    ステップ2 (勾配の算出)
    ミニバッチの損失関数を減らすために、各重みパラメータの
    勾配を求める
    勾配は損失関数の値を最も減らす方向を示す

    ステップ3 (パラメータの更新)
    重みパラメータを勾配方向に微小量だけ更新

    ステップ4 (繰り返す)
    ステップ1, ステップ2, ステップ3 を繰り返す

    ==> 確率的勾配降下方 (stochastic gradient descent)
    無作為に選び出したデータに対して行う勾配降下方 (SGD)
    """

    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        """
        初期化

        @param input_size 入力層のニューロンの数
        @param 隠れ層のニューロンの数
        @param 出力層のニューロンの数
        """
        # 重みの初期化
        self.params = {}
        self.params["W1"] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params["b1"] = np.zeros(hidden_size)

        self.params["W2"] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params["b2"] = np.zeros(output_size)


    def predict(self, x):
        """
        認識(推論)を行う

        @param x 入力データ
        """
        W1, W2 = self.params["W1"], self.params["W2"]
        b1, b2 = self.params["b1"], self.params["b2"]

        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        y = softmax(a2)

        return y


    def loss(self, x, t):
        """
        損失関数の値を求める

        @param x 入力データ
        @param t 教師データ
        """
        y = self.predict(x)
        return cross_entropy_error(y, t)


    def accuracy(self, x, t):
        """
        認識精度も求める

        @param x 入力データ
        @param t 教師データ
        """
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        t = np.argmax(t, axis=1)

        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy


    def numerical_gradient(self, x, t):
        """
        重みパラメータに対する勾配を求める

        @param x 入力データ
        @param y 教師データ
        """
        loss_W = lambda W: self.loss(x, t)

        grads = {}
        grads["W1"] = numerical_gradient(loss_W, self.params["W1"])
        grads["b1"] = numerical_gradient(loss_W, self.params["b1"])
        grads["W2"] = numerical_gradient(loss_W, self.params["W2"])
        grads["b2"] = numerical_gradient(loss_W, self.params["b2"])

        return grads
