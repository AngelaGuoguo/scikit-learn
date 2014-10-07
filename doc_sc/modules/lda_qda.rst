.. _lda_qda:

==========================================
线性和二阶判别分析
==========================================

.. currentmodule:: sklearn

线性判别分析 (linear discriminant analysis :class:`lda.LDA`)和二阶判别分析 (quadratic discriminant analysis :class:`qda.QDA`) 是两个类别分类器。正如其名，它们的判别空间分别为线性和二阶曲面。

这些分类器的优势在于它们有着解析解，支持多类别，且实际应用效果好。此外算法中没有额外参数需要调节。

.. |ldaqda| image:: ../auto_examples/classification/images/plot_lda_qda_001.png
        :target: ../auto_examples/classification/plot_lda_qda.html
        :scale: 80

.. centered:: |ldaqda|

上图展示了判别边界。图中第二行对比了LDA和QDA，可见当边界不是线性时，QDA更为灵活。

.. topic:: 示例:

    :ref:`example_classification_plot_lda_qda.py`: Comparison of LDA and QDA on synthetic data.

.. topic:: 参考:

     .. [3] "The Elements of Statistical Learning", Hastie T., Tibshirani R.,
        Friedman J., 2008.


LDA用于维度降低
==================================

:class:`lda.LDA` 可以用来作监督的维度降低。将输入数据投影到一个包含最主要判别信息的子空间中。这是通过函数 :func:`lda.LDA.transform` 完成。需要的维度数是通过 ``n_components`` 设定，但其对 :func:`lda.LDA.fit` 和 :func:`lda.LDA.predict` 没有意义。


数学想法
=================

这两种方法都是在基于数据来建立类别的条件分布， :math:`P(X|y=k)` 。对于每个分类 :math:`k` 预测可以通过贝叶斯定理得到：

.. math::
    P(y | X) = P(X | y) \cdot P(y) / P(X) = P(X | y) \cdot P(Y) / ( \sum_{y'} P(X | y') \cdot p(y'))

在线性和二阶判别分析中 :math:`P(X|y)` 假设为高斯分布。而且在LDA中，每个类别都假设有相同的协变矩阵。这意味着一个线性的判别平面，这点是可以通过比较两个对数概率得到：
:math:`log[P(y=k | X) / P(y=l | X)]`.

对于QDA，没有假设任何协变矩阵的信息，因此产生一个二阶判别平面。
