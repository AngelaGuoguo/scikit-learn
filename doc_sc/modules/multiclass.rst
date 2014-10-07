
.. _multiclass:

====================================
多类别和多标签算法
====================================

.. currentmodule:: sklearn.multiclass

.. warning::
    所有scikit-learn的分类器都可以直接进行多类别分类。你无需使用 :mod:`sklearn.multiclass` 模板。除非你打算尝试不同的分类算法。

:mod:`sklearn.multiclass` 采用 *元统计* 来 ``多类别`` 和 ``多标签`` 分类问题简化为二元分类问题进行解决。

  - **多类别分类** 是指分类问题有两个以上的类别。例如将一系列水果的图片分为橘子，苹果或者梨。多类别分类假设每一个样本都会对应且仅对应一个类别。一个水果只可能是苹果或梨等，但不可以同时是两种。

  - **多标签分类** 将每个取样标记上若干个标签。这可以理解为预测一个数据的不互相排斥的若干性质，如一个文档的若干话题。一个文章可以同时有宗教，政治，金融或者教育等多个话题。

  - **多输出-多重分类问题** 和 **多任务分类** 代表一个统计模型要预测几个联合的分类任务。这是对多标签分类的推广，where the set of classification problem is restricted to binary classification, and of the multi-class classification task. *其输出是一个2维numpy数列，或者稀疏矩阵* 。

    对于输出变量，其每个的类别集合可以是不同的。例如一个样本可以被标记为“梨”的同时，又被标记为“黄色”。

    这意味着任何一个能够处理多输出多类别或者多任务的分类器也支持作为特殊情况的多标签分类。多任务分类与多输出分类类似，只是有不同的模型公式。更多的信息请参考相关文献。

所有的scikit-learn的分类器都支持多类别分类。但是 :mod:`sklearn.multiclass`  元分析允许改变其处理多类别的方式。由此会改变分类器的表现（误差或者计算效率）。

以下是对scikit-learn支持的分类器的一个总结。如果你使用一下的分类器，那么你不需要使用原统计。

  - 继承性多类别 :ref:`朴素贝叶斯 <naive_bayes>` ， :class:`sklearn.lda.LDA` ，  :ref:`决策树 <tree>`, :ref:`随机森林 <forest>` ， :ref:`最近邻 <neighbors>` ，设定 "multi_class=multinomial" 的  :class:`sklearn.linear_model.LogisticRegression`.
  - 一对一： :class:`sklearn.svm.SVC`.
  - 一对多： 除了 :class:`sklearn.svm.SVC` 其他的线性模型。

Some estimators also support multioutput-multiclass classification
tasks :ref:`Decision Trees <tree>`, :ref:`Random Forests <forest>`,
:ref:`Nearest Neighbors <neighbors>`.

.. warning::

    目前 :mod:`sklearn.metrics` 不支持多输出多类别分类。

多标签分类的格式
================================

在多标签分类中，其二元分类任务可以通过数组表示：每一个样本是大小为(n_samples, n_classes)的二维数列中的一行。一个数列 ``np.array([[1, 0, 0], [0, 1, 1], [0, 0, 0]])`` 表示第一个样本的标签为0，第二个样本的标签为1和2，第三个样本没有标签。

创建一个多标签的数据更为直接：转换器 :class:`MultiLabelBinarizer <preprocessing.MultiLabelBinarizer>` 可以自动将标签转换为需要的格式。

  >>> from sklearn.datasets import make_multilabel_classification
  >>> from sklearn.preprocessing import MultiLabelBinarizer
  >>> X, Y = make_multilabel_classification(n_samples=5, random_state=0,
  ...                                       return_indicator=False)
  >>> Y
  [[2, 3, 4], [2], [0, 1, 3], [0, 1, 2, 3, 4], [0, 1, 2]]
  >>> MultiLabelBinarizer().fit_transform(Y)
  array([[0, 0, 1, 1, 1],
         [0, 0, 1, 0, 0],
         [1, 1, 0, 1, 0],
         [1, 1, 1, 1, 1],
         [1, 1, 1, 0, 0]])

一对多
===============

一对多策略是由 :class:`OneVsRestClassifier` 执行的。这个算法是对每一个分类拟合一次。对于每一个分类器，该分类将对余下的所有分类进行区别。除了其计算的高效（只需要 `n_classes` 个分类器），另一个优势在于容易理解。因为每一个类别都是由一个分类器所对应，所以理解分类器有助于理解该分类的特征。因此此策略被作为默认策略。

多类别学习
-------------------

以下是一个采用一对多进行多类别学习的例子

  >>> from sklearn import datasets
  >>> from sklearn.multiclass import OneVsRestClassifier
  >>> from sklearn.svm import LinearSVC
  >>> iris = datasets.load_iris()
  >>> X, y = iris.data, iris.target
  >>> OneVsRestClassifier(LinearSVC(random_state=0)).fit(X, y).predict(X)
  array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
         1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 1, 1, 1, 1, 1, 1, 1,
         1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
         2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 2, 2, 2, 1, 2, 2, 2, 2,
         2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2])

多标签学习
-------------------

:class:`OneVsRestClassifier` 也支持多标签学习。在此，只需要提供给分类器一个标签矩阵，每一个值 [i, j] 代表i样本有jlabel。

.. figure:: ../auto_examples/images/plot_multilabel_001.png
    :target: ../auto_examples/plot_multilabel.html
    :align: center
    :scale: 75%

.. topic:: 示例:

    * :ref:`example_plot_multilabel.py`


一对一
==========

:class:`OneVsOneClassifier` 构造每一个类别对另一个类别的分类器。在预测时，被选最多的分类将被选作最终分类。因为其需要拟合 ``n_classes * (n_classes - 1) / 2`` 分类器，所以它比一对多的分类器速度慢。然而这个方法当核心算法的计算时间不正比于 ``n_samples`` 时有一定的优势。因为每一个独立的学习问题只涉及一小部分的数据，而一对多需要考虑全部数据 ``n_classes`` 次。

多类别学习
-------------------

以下是采用一对一方法的多类别学习::

  >>> from sklearn import datasets
  >>> from sklearn.multiclass import OneVsOneClassifier
  >>> from sklearn.svm import LinearSVC
  >>> iris = datasets.load_iris()
  >>> X, y = iris.data, iris.target
  >>> OneVsOneClassifier(LinearSVC(random_state=0)).fit(X, y).predict(X)
  array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
         1, 2, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1,
         1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
         2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
         2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2])


纠错输出码
=============================

基于输出码的算法与一对一和一对多的算法不同。在这个算法中，每一个类别被表示为一个欧几里得空间中的点，而每个维度只有0和1。另一个理解方式是，每一个类别是由一个二进制码表示（一系列0和1）。这个记录类别的矩阵被称之为编码本。其编码的大小是此前提到的空间的维度。直观上将，每一个类是是有一个编码标记，且编码本的目标是优化分类的准确性。在本实现中，我们采用 [2]_ 所提倡的随机生成编码本，尽管更好的方法会在日后添加。

在你和的时候，一个二元分类器拟合编码本中的一个字节。在预测时，一个分类器用来预测样本在空间中的位置，而最近的分类将被采用。

在 :class:`OutputCodeClassifier` 中 ``code_size`` 控制分类器的数目，其是总类别数的百分比。

一个介于0和1之间的数对应少于一对多的分类器。在理论上， ``log2(n_classes) / n_classes`` 足以表示所有的类别，但实际应用上，其准确度不高。

一个大于1的数值会产生比一对多更多的的分类器。在理论上有些分类器是在更正其他分类器带来的差错。，因此被称为“纠错”。在实际上，并不一定如此，因为错误并不是相关的。纠错输出码的效果与bagging类似。


多类别学习
-------------------

以下是一个通过输出码进行多类别学习的例子：

  >>> from sklearn import datasets
  >>> from sklearn.multiclass import OutputCodeClassifier
  >>> from sklearn.svm import LinearSVC
  >>> iris = datasets.load_iris()
  >>> X, y = iris.data, iris.target
  >>> clf = OutputCodeClassifier(LinearSVC(random_state=0),
  ...                            code_size=2, random_state=0)
  >>> clf.fit(X, y).predict(X)
  array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1,
         1, 2, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 2, 2, 2, 1, 1, 1, 1, 1, 1,
         1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
         2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 2, 2, 2, 1, 1, 2, 2, 2,
         2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2])

.. topic:: 参考:

    .. [1] "Solving multiclass learning problems via error-correcting output codes",
        Dietterich T., Bakiri G.,
        Journal of Artificial Intelligence Research 2,
        1995.

    .. [2] "The error coding method and PICTs",
        James G., Hastie T.,
        Journal of Computational and Graphical statistics 7,
        1998.

    .. [3] "The Elements of Statistical Learning",
        Hastie T., Tibshirani R., Friedman J., page 606 (second-edition)
        2008.
