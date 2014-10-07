.. _semi_supervised:

===================================================
半监督学习
===================================================

.. currentmodule:: sklearn.semi_supervised

`半监督学习
<http://en.wikipedia.org/wiki/Semi-supervised_learning>`_ 针对的是当训练并不是全部分类的情况。 :mod:`sklearn.semi_supervised` 模块的半监督学习可以利用没有分类的数据来更好的掌握数据的分布和预测。这些算法针对当我们只有一小部分分类的数据和大量未分类的数据非常有效。

.. topic:: 未分类条目 `y`

    用 ``fit``  拟合未分类的数据，我们需要清晰的将其在训练样本中标注出来。本模块采用的是整数值 :math:`-1`.

.. _label_propagation:

标记传递
=================

标记传递是一系列半监督导图算法。

该模型的特征：
  * 可以用作分类或者回归问题
  * 核心方法是将数据投射到另一个维度

`scikit-learn` 提供了两个标记传递的方法 :class:`LabelPropagation` 和 :class:`LabelSpreading` 。两者都是构造一个所有数据的近似图。

.. figure:: ../auto_examples/semi_supervised/images/plot_label_propagation_structure_001.png
    :target: ../auto_examples/semi_supervised/plot_label_propagation_structure.html
    :align: center
    :scale: 60%

    **标记传递示例** *未标记的取样的数据的分布和已标记的分布相同，因此可以在训练样本中通过已有标记的分类来推导未标记的分类。*

:class:`LabelPropagation` 和 :class:`LabelSpreading` 的区别在于其相似矩阵和对类别分布的夹紧。夹紧是指通过算法来改变已分类数据的权重。 :class:`LabelPropagation` 采用严格夹紧： :math:`\alpha=1` 。当 :math:`\alpha=0.8` 时，我们要求有80%的样本还满足原始的类别分布，但算法可以改变剩下20%的分布。

:class:`LabelPropagation` 使用直接从数据的得到的相似矩阵。作为对比, :class:`LabelSpreading` 采用最小化一个含有复杂度参数的成本函数，因此其对噪声更为稳定。这个算法逐步你黑一个更新的原始图，并通过拉普拉斯矩阵重置边权重。这个方法也用在 :ref:`spectral_clustering` 中。

有两个核心函数，对其选择会影响算法的扩展性和效率：

  * rbf (:math:`\exp(-\gamma |x-y|^2), \gamma > 0`). :math:`\gamma` 是需要设定的参数 ``gamma``

  * knn (:math:`1[x' \in kNN(x)]`). :math:`k` 是需要设定的参数 ``n_neighbors`` 。

RBF核心函数可以建构一个完全连通图，这是个密集矩阵，因此会占用很大内存，并且很多矩阵运算会导致计算时间较长。另一方面KKN核心函数会产生一个稀疏矩阵，可以降低运行时间。

.. topic:: 示例

  * :ref:`example_semi_supervised_plot_label_propagation_versus_svm_iris.py`
  * :ref:`example_semi_supervised_plot_label_propagation_structure.py`
  * :ref:`example_semi_supervised_plot_label_propagation_digits_active_learning.py`

.. topic:: 参考

    [1] Yoshua Bengio, Olivier Delalleau, Nicolas Le Roux. In Semi-Supervised
    Learning (2006), pp. 193-216

    [2] Olivier Delalleau, Yoshua Bengio, Nicolas Le Roux. Efficient
    Non-Parametric Function Induction in Semi-Supervised Learning. AISTAT 2005
    http://research.microsoft.com/en-us/people/nicolasl/efficient_ssl.pdf

