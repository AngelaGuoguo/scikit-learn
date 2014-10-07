.. _isotonic:

===================
保序回归
===================

.. currentmodule:: sklearn.isotonic

:class:`IsotonicRegression` 拟合数据到一个非降函数来解决下面的问题：

  minimize :math:`\sum_i w_i (y_i - \hat{y}_i)^2`

  subject to :math:`\hat{y}_{min} = \hat{y}_1 \le \hat{y}_2 ... \le \hat{y}_n = \hat{y}_{max}`

其中 :math:`w_i` 是正数，每个 :math:`y_i` 是任意实数。其产生一个非降序列有着最小的方差。实际上会产生一个局部线性的函数。

.. figure:: ../auto_examples/images/plot_isotonic_regression_001.png
   :target: ../auto_examples/images/plot_isotonic_regression.html
   :align: center
