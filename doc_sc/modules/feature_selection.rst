.. currentmodule:: sklearn.feature_selection

.. _feature_selection:

=================
特征选择
=================


:mod:`sklearn.feature_selection` 模块可以对取样数据进行特征选择或者维度降低，并以此来增加准确度，或者提高运算效率。


去除低方差的特征
===================================

:class:`VarianceThreshold` 是一个简单的基本操作。它将去除掉所有方差没有超过阈值的特征。默认情况，它将去除所有无方差的特征，如某一个特征在全部取样中的值相同。

假设我们有一个数据只有布尔特征，我们希望移除那些超过80%都相同的特征。因为布尔特征是伯努利随机变量，因此其方差为：

.. math:: \mathrm{Var}[X] = p(1 - p)

因此我们可以选择阈值 ``.8 * (1 - .8)``::

  >>> from sklearn.feature_selection import VarianceThreshold
  >>> X = [[0, 0, 1], [0, 1, 0], [1, 0, 0], [0, 1, 1], [0, 1, 0], [0, 1, 1]]
  >>> sel = VarianceThreshold(threshold=(.8 * (1 - .8)))
  >>> sel.fit_transform(X)
  array([[0, 1],
         [1, 0],
         [0, 0],
         [1, 1],
         [1, 0],
         [1, 1]])

正如所料， ``VarianceThreshold`` 将去除第一列，因为其包含了过多的0 。


单变量特征选择
============================

单变量特征选择是基于单变量统计测试来选择最佳的特征。其可以视作是构建统计量的准备工作。Scikit-learn通过采用类函数 ``transform`` 来实现：

 * :class:`SelectKBest` 仅保留最好的 :math:`k` 个特征

 * :class:`SelectPercentile` 仅保留用户定义的最高百分比的特征

 * 对于每个特征采用单变量统计：假阳性率 :class:`SelectFpr`, 错误发现率
   :class:`SelectFdr`, 或者类别错误 :class:`SelectFwe`.

 * :class:`GenericUnivariateSelect` 进行可调控的单变量特征选择。其通过超函数选择最佳的选择方案

例如，我们可以通过 :math:`\chi^2` 检验来选择最好的两个特征：

  >>> from sklearn.datasets import load_iris
  >>> from sklearn.feature_selection import SelectKBest
  >>> from sklearn.feature_selection import chi2
  >>> iris = load_iris()
  >>> X, y = iris.data, iris.target
  >>> X.shape
  (150, 4)
  >>> X_new = SelectKBest(chi2, k=2).fit_transform(X, y)
  >>> X_new.shape
  (150, 2)

这些类均需要一个返回p-value的评分函数：

 * 对于回归： :func:`f_regression`

 * 对于分类: :func:`chi2` or :func:`f_classif`

.. topic:: 稀疏数据的特征选择

   如果数据是稀疏的（如数据由稀疏矩阵表示），那么只有 :func:`chi2` 可以处理该数据，让其还是稀疏矩阵。

.. warning::

    注意不要对分类问题采用回归问题的评分函数

.. topic:: 示例:

    :ref:`example_feature_selection_plot_feature_selection.py`


递归特征消除
=============================

递归特征消除（recursive feature elimination :class:`RFE` ）作为一个为特征附以权重的统计方法（如线性模型系数），其通过递归方式逐步考虑更小的特征集合来选择特征。首先测量初始的样本并赋予每一个特征权重，接着那些权重最小的特征被消除出去。这样的过程重复直到只留下需要数目的特征。

:class:`RFECV` 通过交叉检验来找到最合适的特征数目。

.. topic:: 示例:

    * :ref:`example_feature_selection_plot_rfe_digits.py`: A recursive feature elimination example
      showing the relevance of pixels in a digit classification task.

    * :ref:`example_feature_selection_plot_rfe_with_cross_validation.py`: A recursive feature
      elimination example with automatic tuning of the number of features
      selected with cross-validation.


.. _l1_feature_selection:

基于L1的特征选择
==========================

.. currentmodule:: sklearn

选择非零系数
---------------------------------

:ref:`线性模型 <linear_model>` 通过L1来惩罚无效的模型，因此很多系数为0。当目的是降低数据维数时，可以采用函数 ``transform`` 来选择那些非零系数。其中尤其是回归问题的 :class:`linear_model.Lasso` 和 :class:`linear_model.LogisticRegression` 以及分类问题的 :class:`svm.LinearSVC` ::

  >>> from sklearn.svm import LinearSVC
  >>> from sklearn.datasets import load_iris
  >>> iris = load_iris()
  >>> X, y = iris.data, iris.target
  >>> X.shape
  (150, 4)
  >>> X_new = LinearSVC(C=0.01, penalty="l1", dual=False).fit_transform(X, y)
  >>> X_new.shape
  (150, 3)

在SVM 和逻辑回归中，参数C控制疏散性。越小的C值，越少的特征被选择。在Lasso中，越大的alpha，越少的特征被选择。

.. topic:: 示例:

    * :ref:`example_text_document_classification_20newsgroups.py`: Comparison
      of different algorithms for document classification including L1-based
      feature selection.

.. _compressive_sensing:

.. topic:: **L1-恢复和压缩传感**

   一个合适的alpha可以使 :ref:`lasso` 在一定情况下只通过少量数据来完全重构所有非零数据。其需要满足样本足够多，或者L1模型是随机应用的。其中足够多是要依赖于非零的系数，特征数目的对数，噪声的大小，绝对值最小的非零系数，以及数据矩阵X的结构。其次X还需要满足一定性质，如不是非常相关。

   这里并没有一个选择alpha参数的一般标准。其可以通过交叉检验确定（ :class:`LassoCV` 或 :class:`LassoLarsCV` ），尽管这可能导致不完全惩罚的模型：有一小部分非相关的变量并不会对预测有不利的影响。BIC  (:class:`LassoLarsIC`) 设定非常高的 alpha。

   **参考** Richard G. Baraniuk "Compressive Sensing", IEEE Signal
   Processing Magazine [120] July 2007
   http://dsp.rice.edu/files/cs/baraniukCSlecture07.pdf

.. _randomized_l1:

随机稀疏模型
-------------------------

.. currentmodule:: sklearn.linear_model

基于L1的系数模型当村长高相关的特征是，其只会选择一个。为了减轻这个问题，随机性被引入其中。每次通过随机取样来扰动数据矩阵，并记录每次选择的特征。

:class:`RandomizedLasso` 在Lasso中采用了上述策略。而 :class:`RandomizedLogisticRegression` 则对应逻辑回归，并适用于分类问题。如果要得到完整的稳定得分，可以使用函数 :func:`lasso_stability_path`.

.. figure:: ../auto_examples/linear_model/images/plot_lasso_model_selection_001.png
   :target: ../auto_examples/linear_model/plot_lasso_model_selection.html
   :align: center
   :scale: 60

注意，随机稀疏模型比标准的F检验更有效地发现非零特征，当仅有一小部分特征非零时。

.. topic:: 示例:

   * :ref:`example_linear_model_plot_sparse_recovery.py`: An example
     comparing different feature selection approaches and discussing in
     which situation each approach is to be favored.

.. topic:: 参考:

   * N. Meinshausen, P. Buhlmann, "Stability selection",
     Journal of the Royal Statistical Society, 72 (2010)
     http://arxiv.org/pdf/0809.2932

   * F. Bach, "Model-Consistent Sparse Estimation through the Bootstrap"
     http://hal.inria.fr/hal-00354771/

基于树的特征选择
============================

基于数的统计量（参见 :mod:`sklearn.tree` 和树的集合 :mod:`sklearn.ensemble` ）可以用来计算特征的重要性，并排除非相关特征::

  >>> from sklearn.ensemble import ExtraTreesClassifier
  >>> from sklearn.datasets import load_iris
  >>> iris = load_iris()
  >>> X, y = iris.data, iris.target
  >>> X.shape
  (150, 4)
  >>> clf = ExtraTreesClassifier()
  >>> X_new = clf.fit(X, y).transform(X)
  >>> clf.feature_importances_  # doctest: +SKIP
  array([ 0.04...,  0.05...,  0.4...,  0.4...])
  >>> X_new.shape               # doctest: +SKIP
  (150, 2)

.. topic:: Examples:

    * :ref:`example_ensemble_plot_forest_importances.py`: example on
      synthetic data showing the recovery of the actually meaningful
      features.

    * :ref:`example_ensemble_plot_forest_importances_faces.py`: example
      on face recognition data.

模型选择流水线
=======================================

特征选择通常是正式学习前的预处理。一个推荐的方式是通过 scikit-learn 的 :class:`sklearn.pipeline.Pipeline` 类来完成::

  clf = Pipeline([
    ('feature_selection', LinearSVC(penalty="l1")),
    ('classification', RandomForestClassifier())
  ])
  clf.fit(X, y)

在上例中我们使用 :class:`sklearn.svm.LinearSVC` 来评估特征的重要性，并选择最重要的特征。进而， :class:`sklearn.ensemble.RandomForestClassifier` 对输出的特征进行拟合。你可以通过类似的方式来应用其他特征选择方法和分类器。更多信息参见 :class:`sklearn.pipeline.Pipeline` 。
