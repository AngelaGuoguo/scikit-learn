.. _ensemble:

================
集成方法
================

.. currentmodule:: sklearn.ensemble

**集成方法** 的目标是通过结合某一种算法的若干个基本统计来提高预测的准确性。

有两类集成方法需要区别

- **平均方法** 其核心是构建若干个独立的模型，在平均他们的预测。平均而言，由于方差的降低，综合结果一般比任何一个单一的结果要好。

  **示例：** :ref:`Bagging methods <bagging>`, :ref:`Forests of randomized trees <forest>`, ...

- 相对而言， **增强方法** 是按顺序构建基本的统计来降低综合模型的误差。其想法是综合一般的模型来或等更好的模型。

  **示例** :ref:`AdaBoost <adaboost>`, :ref:`Gradient Tree Boosting <gradient_boosting>`, ...


.. _bagging:

Bagging的元统计
======================

在集成方法中，bagging是一类算法。其通过原训练数据的随机子集来构建若干基本黑盒统计，在将他们合并成为最终的预测。这些方法通过在拟合中引入随机性来降低统计的方差。bagging方法相对简单，因为其对模型的优化并不需要了解底层的模型的具体信息此外它可以降低过度拟合的影响，因此对复杂的模型（如完全展开的决策树）时，有很好的作用。而增强方法则对弱的模型（如浅的决策树）有更好的表现。

Bagging方法有若干区别，其中最显著的是他们产生随机性的方式：

  * Pasting方法通过选择随机选择子集来作为取样数据 [B1999]_ 。

  * Bagging方法允许子集的样本存在重复取样 [B1996]_ 。

  * 随机子空间（randome subsets）的方法选取样本的随机特征进行训练 [H1998]_ 。

  * 随机区域（Random Patches）则选择样本与特征的随机子集进行训练 [LG2012]_ 。

在scikit-learn中bagging方法是由统一的类 :class:`BaggingClassifier` 来进行分析的（以及 :class:`BaggingRegressor` ）。其输入变量是一个用户自定义的基本统计器（分类或回归），以及相对应的选取随机子集的方式。其中 ``max_samples`` 和 ``max_features`` 控制子集的大小（样本数和特征数）而 ``bootstrap`` 和 ``bootstrap_features`` 控制是否重复取样样本及特征。譬如下例展示如何采用bagging集成方法通过 :class:`KNeighborsClassifier` 统计来学习 50% 的样本和 50% 的特征。

    >>> from sklearn.ensemble import BaggingClassifier
    >>> from sklearn.neighbors import KNeighborsClassifier
    >>> bagging = BaggingClassifier(KNeighborsClassifier(),
    ...                             max_samples=0.5, max_features=0.5)

.. topic:: 示例:

 * :ref:`example_ensemble_plot_bias_variance.py`

.. topic:: References

  .. [B1999] L. Breiman, "Pasting small votes for classification in large
         databases and on-line", Machine Learning, 36(1), 85-103, 1999.

  .. [B1996] L. Breiman, "Bagging predictors", Machine Learning, 24(2),
         123-140, 1996.

  .. [H1998] T. Ho, "The random subspace method for constructing decision
         forests", Pattern Analysis and Machine Intelligence, 20(8), 832-844,
         1998.

  .. [LG2012] G. Louppe and P. Geurts, "Ensembles on Random Patches",
         Machine Learning and Knowledge Discovery in Databases, 346-361, 2012.

.. _forest:

随机树与森林
===========================

:mod:`sklearn.ensemble` 提供两个平均算法基于随机化 :ref:`决策树 <tree>` ：随机森林（RandomForest）和极端树（Extra-Trees）。两种算法都是采用针对树的扰动-合并技术 [B1998]_ 。这是说在模型构造阶段通过随机性创立一系列不同的分类器。最终的预测是基于每一个独立预测的平均。

如其他分类器一样，森林分类需要两个输入数列：一个大小为 ``[n_samples, n_features]`` 的训练数据X，和一个大小为 ``[n_samples]`` 的训练样本的类别数列Y::

    >>> from sklearn.ensemble import RandomForestClassifier
    >>> X = [[0, 0], [1, 1]]
    >>> Y = [0, 1]
    >>> clf = RandomForestClassifier(n_estimators=10)
    >>> clf = clf.fit(X, Y)

如 :ref:`决策树 <tree>` 随机森林也可以延伸到 :ref:`多输出问题 <tree_multioutput>` （当Y是一个大小为 ``[n_samples, n_outputs]`` 的类别数列）。


随机森林
--------------

在随机森林中（ :class:`RandomForestClassifier` 和 :class:`RandomForestRegressor` 类），每一个树在集合中都是基于一个可重复取样的子样本（如bootstrap取样）。此外，当在拟合中对于一个决策点，其不在是一个对于全部数据的最佳决策而是这个子样本所选取的特征的最佳决策。 由于这个随机性，因此随机森林的偏差会有所增大（相对一个没有随机的树）。但由于平均的作用，通常方差会减少，因此往往会平衡偏差的影响，进而产生更好的模型。

对比初始的文献 [B2001]_ ，scikit-learn采用合并预测的概率，而不是每个分类器来对类别进行投票。

极端（随机）树
--------------------------

在极端树中（参见 :class:`ExtraTreesClassifier` 和 :class:`ExtraTreesRegressor` 类），随机性被进一步增强。对比随机森林选用随机的特征，并选择最优的分类，极端随机树采用随机的特征，并选择随机的决策阈值中最佳的一个作为分类决策。这通常可以进一步降低模型的方差，但是也进一步增加偏差::

    >>> from sklearn.cross_validation import cross_val_score
    >>> from sklearn.datasets import make_blobs
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> from sklearn.ensemble import ExtraTreesClassifier
    >>> from sklearn.tree import DecisionTreeClassifier

    >>> X, y = make_blobs(n_samples=10000, n_features=10, centers=100,
    ...     random_state=0)

    >>> clf = DecisionTreeClassifier(max_depth=None, min_samples_split=1,
    ...     random_state=0)
    >>> scores = cross_val_score(clf, X, y)
    >>> scores.mean()                             # doctest: +ELLIPSIS
    0.97...

    >>> clf = RandomForestClassifier(n_estimators=10, max_depth=None,
    ...     min_samples_split=1, random_state=0)
    >>> scores = cross_val_score(clf, X, y)
    >>> scores.mean()                             # doctest: +ELLIPSIS
    0.999...

    >>> clf = ExtraTreesClassifier(n_estimators=10, max_depth=None,
    ...     min_samples_split=1, random_state=0)
    >>> scores = cross_val_score(clf, X, y)
    >>> scores.mean() > 0.999
    True

.. figure:: ../auto_examples/ensemble/images/plot_forest_iris_001.png
    :target: ../auto_examples/ensemble/plot_forest_iris.html
    :align: center
    :scale: 75%

参数
----------

在采用这个模型时，最主要的参数是 ``n_estimators`` 和 ``max_features`` 。前者是森林中树的数目。数目越大越好，但是会导致更长的计算时间。此外，当超过一定数目的树后，结果将不会有更显著的提高。后者是子集中用以进行决策的特征数目。较低的数值可以较大的降低方差，但是带来偏差的增加。一个经验上较好的默认数值是在回归问题中 ``max_features=n_features`` ， 而在分类问题中 ``max_features=sqrt(n_features)`` （其中 ``n_features`` 是样本的特征总数）。最佳的结果往往是选择 ``max_depth=None`` 及 ``min_samples_split=1`` （如一个完整发展的树）。但是请记住这些数值并不是最优选择。一个最佳的参数需要经过交叉检验。而且注意在随机森林中，bootstrap取样是默认的（ ``bootstrap=True`` ）但是在极端树中却没有采用 (``bootstrap=False``)。

并行化
---------------

本模块可以通过设定参数 ``n_jobs``来并行构建树和计算预测。当 ``n_jobs=k`` 时，计算被分为
``k`` 个工作，并在 ``k`` 计算处理器上进行。当 ``n_jobs=-1`` 时，所有空闲的处理器将被用以计算。注意由于内部信息沟通损耗，这个加速过程并不是与处理器数目成正比（例如采用 ``k`` 个处理器并不会快 ``k`` 倍）。当然当构建很多树的时候或者构建单个树很耗时（很大的数据）的时候，其增速还是相当客观的。

.. topic:: 示例:

 * :ref:`example_ensemble_plot_forest_iris.py`
 * :ref:`example_ensemble_plot_forest_importances_faces.py`
 * :ref:`example_plot_multioutput_face_completion.py`

.. topic:: 参考

 .. [B2001] L. Breiman, "Random Forests", Machine Learning, 45(1), 5-32, 2001.

 .. [B1998] L. Breiman, "Arcing Classifiers", Annals of Statistics 1998.

 .. [GEW2006] P. Geurts, D. Ernst., and L. Wehenkel, "Extremely randomized
   trees", Machine Learning, 63(1), 3-42, 2006.

.. _random_forest_feature_impotance:

特征重要性评估
-----------------------------

一个特征在决策树的相对排名（如深度）是由其在决策点的相对重要性决定。在树顶层的特征将影响以大部分的样本的最终预测。因此，这样一个 **预期的样本比例** 可以用来决定 **特征的相对重要性** 。

通过 **平均** 若干随机树的预期样本比例，我们使得该特征的 **方差降低** 。

以下离子展示了面部预测的例子中每一个像素点的 :class:`ExtraTreesClassifier` 中的相对权重。

.. figure:: ../auto_examples/ensemble/images/plot_forest_importances_faces_001.png
   :target: ../auto_examples/ensemble/plot_forest_importances_faces.html
   :align: center
   :scale: 75

在实际应用中，这些存储在 ``feature_importances_`` ，骑士一个大小为 ``(n_features,)`` 的数列。其中值是正数，且和为1。越大的数对应其特征在机器学习中更重要的地位。

.. topic:: 示例:

 * :ref:`example_ensemble_plot_forest_importances_faces.py`
 * :ref:`example_ensemble_plot_forest_importances.py`

.. _random_trees_embedding:

完全随机树集合
------------------------------

:class:`RandomTreesEmbedding` implements an unsupervised transformation of the
data.  Using a forest of completely random trees, :class:`RandomTreesEmbedding`
encodes the data by the indices of the leaves a data point ends up in.  This
index is then encoded in a one-of-K manner, leading to a high dimensional,
sparse binary coding.
This coding can be computed very efficiently and can then be used as a basis
for other learning tasks.
The size and sparsity of the code can be influenced by choosing the number of
trees and the maximum depth per tree. For each tree in the ensemble, the coding
contains one entry of one. The size of the coding is at most ``n_estimators * 2
** max_depth``, the maximum number of leaves in the forest.

As neighboring data points are more likely to lie within the same leaf of a tree,
the transformation performs an implicit, non-parametric density estimation.

.. topic:: Examples:

 * :ref:`example_ensemble_plot_random_forest_embedding.py`

 * :ref:`example_manifold_plot_lle_digits.py` compares non-linear
   dimensionality reduction techniques on handwritten digits.

.. seealso::

   :ref:`manifold` techniques can also be useful to derive non-linear
   representations of feature space, also these approaches focus also on
   dimensionality reduction.


.. _adaboost:

AdaBoost
========

The module :mod:`sklearn.ensemble` includes the popular boosting algorithm
AdaBoost, introduced in 1995 by Freund and Schapire [FS1995]_.

The core principle of AdaBoost is to fit a sequence of weak learners (i.e.,
models that are only slightly better than random guessing, such as small
decision trees) on repeatedly modified versions of the data. The predictions
from all of them are then combined through a weighted majority vote (or sum) to
produce the final prediction. The data modifications at each so-called boosting
iteration consist of applying weights :math:`w_1`, :math:`w_2`, ..., :math:`w_N`
to each of the training samples. Initially, those weights are all set to
:math:`w_i = 1/N`, so that the first step simply trains a weak learner on the
original data. For each successive iteration, the sample weights are
individually modified and the learning algorithm is reapplied to the reweighted
data. At a given step, those training examples that were incorrectly predicted
by the boosted model induced at the previous step have their weights increased,
whereas the weights are decreased for those that were predicted correctly. As
iterations proceed, examples that are difficult to predict receive
ever-increasing influence. Each subsequent weak learner is thereby forced to
concentrate on the examples that are missed by the previous ones in the sequence
[HTF]_.

.. figure:: ../auto_examples/ensemble/images/plot_adaboost_hastie_10_2_001.png
   :target: ../auto_examples/ensemble/plot_adaboost_hastie_10_2.html
   :align: center
   :scale: 75

AdaBoost can be used both for classification and regression problems:

  - For multi-class classification, :class:`AdaBoostClassifier` implements
    AdaBoost-SAMME and AdaBoost-SAMME.R [ZZRH2009]_.

  - For regression, :class:`AdaBoostRegressor` implements AdaBoost.R2 [D1997]_.

Usage
-----

The following example shows how to fit an AdaBoost classifier with 100 weak
learners::

    >>> from sklearn.cross_validation import cross_val_score
    >>> from sklearn.datasets import load_iris
    >>> from sklearn.ensemble import AdaBoostClassifier

    >>> iris = load_iris()
    >>> clf = AdaBoostClassifier(n_estimators=100)
    >>> scores = cross_val_score(clf, iris.data, iris.target)
    >>> scores.mean()                             # doctest: +ELLIPSIS
    0.9...

The number of weak learners is controlled by the parameter ``n_estimators``. The
``learning_rate`` parameter controls the contribution of the weak learners in
the final combination. By default, weak learners are decision stumps. Different
weak learners can be specified through the ``base_estimator`` parameter.
The main parameters to tune to obtain good results are ``n_estimators`` and
the complexity of the base estimators (e.g., its depth ``max_depth`` or
minimum required number of samples at a leaf ``min_samples_leaf`` in case of
decision trees).

.. topic:: Examples:

 * :ref:`example_ensemble_plot_adaboost_hastie_10_2.py` compares the
   classification error of a decision stump, decision tree, and a boosted
   decision stump using AdaBoost-SAMME and AdaBoost-SAMME.R.

 * :ref:`example_ensemble_plot_adaboost_multiclass.py` shows the performance
   of AdaBoost-SAMME and AdaBoost-SAMME.R on a multi-class problem.

 * :ref:`example_ensemble_plot_adaboost_twoclass.py` shows the decision boundary
   and decision function values for a non-linearly separable two-class problem
   using AdaBoost-SAMME.

 * :ref:`example_ensemble_plot_adaboost_regression.py` demonstrates regression
   with the AdaBoost.R2 algorithm.

.. topic:: References

 .. [FS1995] Y. Freund, and R. Schapire, "A Decision-Theoretic Generalization of
             On-Line Learning and an Application to Boosting", 1997.

 .. [ZZRH2009] J. Zhu, H. Zou, S. Rosset, T. Hastie. "Multi-class AdaBoost",
               2009.

 .. [D1997] H. Drucker. "Improving Regressors using Boosting Techniques", 1997.

 .. [HTF] T. Hastie, R. Tibshirani and J. Friedman, "Elements of
              Statistical Learning Ed. 2", Springer, 2009.


.. _gradient_boosting:

Gradient Tree Boosting
======================

`Gradient Tree Boosting <http://en.wikipedia.org/wiki/Gradient_boosting>`_
or Gradient Boosted Regression Trees (GBRT) is a generalization
of boosting to arbitrary
differentiable loss functions. GBRT is an accurate and effective
off-the-shelf procedure that can be used for both regression and
classification problems.  Gradient Tree Boosting models are used in a
variety of areas including Web search ranking and ecology.

The advantages of GBRT are:

  + Natural handling of data of mixed type (= heterogeneous features)

  + Predictive power

  + Robustness to outliers in output space (via robust loss functions)

The disadvantages of GBRT are:

  + Scalability, due to the sequential nature of boosting it can
    hardly be parallelized.

The module :mod:`sklearn.ensemble` provides methods
for both classification and regression via gradient boosted regression
trees.

Classification
---------------

:class:`GradientBoostingClassifier` supports both binary and multi-class
classification.
The following example shows how to fit a gradient boosting classifier
with 100 decision stumps as weak learners::

    >>> from sklearn.datasets import make_hastie_10_2
    >>> from sklearn.ensemble import GradientBoostingClassifier

    >>> X, y = make_hastie_10_2(random_state=0)
    >>> X_train, X_test = X[:2000], X[2000:]
    >>> y_train, y_test = y[:2000], y[2000:]

    >>> clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,
    ...     max_depth=1, random_state=0).fit(X_train, y_train)
    >>> clf.score(X_test, y_test)                 # doctest: +ELLIPSIS
    0.913...

The number of weak learners (i.e. regression trees) is controlled by the parameter ``n_estimators``; :ref:`The size of each tree <gradient_boosting_tree_size>` can be controlled either by setting the tree depth via ``max_depth`` or by setting the number of leaf nodes via ``max_leaf_nodes``. The ``learning_rate`` is a hyper-parameter in the range (0.0, 1.0] that controls overfitting via :ref:`shrinkage <gradient_boosting_shrinkage>` .

.. note::

   Classification with more than 2 classes requires the induction
   of ``n_classes`` regression trees at each at each iteration,
   thus, the total number of induced trees equals
   ``n_classes * n_estimators``. For datasets with a large number
   of classes we strongly recommend to use
   :class:`RandomForestClassifier` as an alternative to :class:`GradientBoostingClassifier` .

Regression
----------

:class:`GradientBoostingRegressor` supports a number of
:ref:`different loss functions <gradient_boosting_loss>`
for regression which can be specified via the argument
``loss``; the default loss function for regression is least squares (``'ls'``).

::

    >>> import numpy as np
    >>> from sklearn.metrics import mean_squared_error
    >>> from sklearn.datasets import make_friedman1
    >>> from sklearn.ensemble import GradientBoostingRegressor

    >>> X, y = make_friedman1(n_samples=1200, random_state=0, noise=1.0)
    >>> X_train, X_test = X[:200], X[200:]
    >>> y_train, y_test = y[:200], y[200:]
    >>> est = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1,
    ...     max_depth=1, random_state=0, loss='ls').fit(X_train, y_train)
    >>> mean_squared_error(y_test, est.predict(X_test))    # doctest: +ELLIPSIS
    5.00...

The figure below shows the results of applying :class:`GradientBoostingRegressor`
with least squares loss and 500 base learners to the Boston house price dataset
(:func:`sklearn.datasets.load_boston`).
The plot on the left shows the train and test error at each iteration.
The train error at each iteration is stored in the
:attr:`~GradientBoostingRegressor.train_score_` attribute
of the gradient boosting model. The test error at each iterations can be obtained
via the :meth:`~GradientBoostingRegressor.staged_predict` method which returns a
generator that yields the predictions at each stage. Plots like these can be used
to determine the optimal number of trees (i.e. ``n_estimators``) by early stopping.
The plot on the right shows the feature importances which can be obtained via
the ``feature_importances_`` property.

.. figure:: ../auto_examples/ensemble/images/plot_gradient_boosting_regression_001.png
   :target: ../auto_examples/ensemble/plot_gradient_boosting_regression.html
   :align: center
   :scale: 75

.. topic:: Examples:

 * :ref:`example_ensemble_plot_gradient_boosting_regression.py`
 * :ref:`example_ensemble_plot_gradient_boosting_oob.py`

.. _gradient_boosting_warm_start:

Fitting additional weak-learners
--------------------------------

Both :class:`GradientBoostingRegressor` and :class:`GradientBoostingClassifier`
support ``warm_start=True`` which allows you to add more estimators to an already
fitted model.

::

  >>> _ = est.set_params(n_estimators=200, warm_start=True)  # set warm_start and new nr of trees
  >>> _ = est.fit(X_train, y_train) # fit additional 100 trees to est
  >>> mean_squared_error(y_test, est.predict(X_test))    # doctest: +ELLIPSIS
  3.84...

.. _gradient_boosting_tree_size:

Controlling the tree size
-------------------------

The size of the regression tree base learners defines the level of variable
interactions that can be captured by the gradient boosting model. In general,
a tree of depth ``h`` can capture interactions of order ``h`` .
There are two ways in which the size of the individual regression trees can
be controlled.

If you specify ``max_depth=h`` then complete binary trees
of depth ``h`` will be grown. Such trees will have (at most) ``2**h`` leaf nodes
and ``2**h - 1`` split nodes.

Alternatively, you can control the tree size by specifying the number of
leaf nodes via the parameter ``max_leaf_nodes``. In this case,
trees will be grown using best-first search where nodes with the highest improvement
in impurity will be expanded first.
A tree with ``max_leaf_nodes=k`` has ``k - 1`` split nodes and thus can
model interactions of up to order ``max_leaf_nodes - 1`` .

We found that ``max_leaf_nodes=k`` gives comparable results to ``max_depth=k-1``
but is significantly faster to train at the expense of a slightly higher
training error.
The parameter ``max_leaf_nodes`` corresponds to the variable ``J`` in the
chapter on gradient boosting in [F2001]_ and is related to the parameter
``interaction.depth`` in R's gbm package where ``max_leaf_nodes == interaction.depth + 1`` .

Mathematical formulation
-------------------------

GBRT considers additive models of the following form:

  .. math::

    F(x) = \sum_{m=1}^{M} \gamma_m h_m(x)

where :math:`h_m(x)` are the basis functions which are usually called
*weak learners* in the context of boosting. Gradient Tree Boosting
uses :ref:`decision trees <tree>` of fixed size as weak
learners. Decision trees have a number of abilities that make them
valuable for boosting, namely the ability to handle data of mixed type
and the ability to model complex functions.

Similar to other boosting algorithms GBRT builds the additive model in
a forward stagewise fashion:

  .. math::

    F_m(x) = F_{m-1}(x) + \gamma_m h_m(x)

At each stage the decision tree :math:`h_m(x)` is chosen to
minimize the loss function :math:`L` given the current model
:math:`F_{m-1}` and its fit :math:`F_{m-1}(x_i)`

  .. math::

    F_m(x) = F_{m-1}(x) + \arg\min_{h} \sum_{i=1}^{n} L(y_i,
    F_{m-1}(x_i) - h(x))

The initial model :math:`F_{0}` is problem specific, for least-squares
regression one usually chooses the mean of the target values.

.. note:: The initial model can also be specified via the ``init``
          argument. The passed object has to implement ``fit`` and ``predict``.

Gradient Boosting attempts to solve this minimization problem
numerically via steepest descent: The steepest descent direction is
the negative gradient of the loss function evaluated at the current
model :math:`F_{m-1}` which can be calculated for any differentiable
loss function:

  .. math::

    F_m(x) = F_{m-1}(x) + \gamma_m \sum_{i=1}^{n} \nabla_F L(y_i,
    F_{m-1}(x_i))

Where the step length :math:`\gamma_m` is chosen using line search:

  .. math::

    \gamma_m = \arg\min_{\gamma} \sum_{i=1}^{n} L(y_i, F_{m-1}(x_i)
    - \gamma \frac{\partial L(y_i, F_{m-1}(x_i))}{\partial F_{m-1}(x_i)})

The algorithms for regression and classification
only differ in the concrete loss function used.

.. _gradient_boosting_loss:

Loss Functions
...............

The following loss functions are supported and can be specified using
the parameter ``loss``:

  * Regression

    * Least squares (``'ls'``): The natural choice for regression due
      to its superior computational properties. The initial model is
      given by the mean of the target values.
    * Least absolute deviation (``'lad'``): A robust loss function for
      regression. The initial model is given by the median of the
      target values.
    * Huber (``'huber'``): Another robust loss function that combines
      least squares and least absolute deviation; use ``alpha`` to
      control the sensitivity with regards to outliers (see [F2001]_ for
      more details).
    * Quantile (``'quantile'``): A loss function for quantile regression.
      Use ``0 < alpha < 1`` to specify the quantile. This loss function
      can be used to create prediction intervals
      (see :ref:`example_ensemble_plot_gradient_boosting_quantile.py`).

  * Classification

    * Binomial deviance (``'deviance'``): The negative binomial
      log-likelihood loss function for binary classification (provides
      probability estimates).  The initial model is given by the
      log odds-ratio.
    * Multinomial deviance (``'deviance'``): The negative multinomial
      log-likelihood loss function for multi-class classification with
      ``n_classes`` mutually exclusive classes. It provides
      probability estimates.  The initial model is given by the
      prior probability of each class. At each iteration ``n_classes``
      regression trees have to be constructed which makes GBRT rather
      inefficient for data sets with a large number of classes.
    * Exponential loss (``'exponential'``): The same loss function
      as :class:`AdaBoostClassifier`. Less robust to mislabeled
      examples than ``'deviance'``; can only be used for binary
      classification.

Regularization
----------------

.. _gradient_boosting_shrinkage:

Shrinkage
..........

[F2001]_ proposed a simple regularization strategy that scales
the contribution of each weak learner by a factor :math:`\nu`:

.. math::

    F_m(x) = F_{m-1}(x) + \nu \gamma_m h_m(x)

The parameter :math:`\nu` is also called the **learning rate** because
it scales the step length the the gradient descent procedure; it can
be set via the ``learning_rate`` parameter.

The parameter ``learning_rate`` strongly interacts with the parameter
``n_estimators``, the number of weak learners to fit. Smaller values
of ``learning_rate`` require larger numbers of weak learners to maintain
a constant training error. Empirical evidence suggests that small
values of ``learning_rate`` favor better test error. [HTF2009]_
recommend to set the learning rate to a small constant
(e.g. ``learning_rate <= 0.1``) and choose ``n_estimators`` by early
stopping. For a more detailed discussion of the interaction between
``learning_rate`` and ``n_estimators`` see [R2007]_.

Subsampling
............

[F1999]_ proposed stochastic gradient boosting, which combines gradient
boosting with bootstrap averaging (bagging). At each iteration
the base classifier is trained on a fraction ``subsample`` of
the available training data. The subsample is drawn without replacement.
A typical value of ``subsample`` is 0.5.

The figure below illustrates the effect of shrinkage and subsampling
on the goodness-of-fit of the model. We can clearly see that shrinkage
outperforms no-shrinkage. Subsampling with shrinkage can further increase
the accuracy of the model. Subsampling without shrinkage, on the other hand,
does poorly.

.. figure:: ../auto_examples/ensemble/images/plot_gradient_boosting_regularization_001.png
   :target: ../auto_examples/ensemble/plot_gradient_boosting_regularization.html
   :align: center
   :scale: 75

Another strategy to reduce the variance is by subsampling the features
analogous to the random splits in :class:`RandomForestClassifier` .
The number of subsampled features can be controlled via the ``max_features``
parameter.

.. note:: Using a small ``max_features`` value can significantly decrease the runtime.

Stochastic gradient boosting allows to compute out-of-bag estimates of the
test deviance by computing the improvement in deviance on the examples that are
not included in the bootstrap sample (i.e. the out-of-bag examples).
The improvements are stored in the attribute
:attr:`~GradientBoostingRegressor.oob_improvement_`. ``oob_improvement_[i]`` holds
the improvement in terms of the loss on the OOB samples if you add the i-th stage
to the current predictions.
Out-of-bag estimates can be used for model selection, for example to determine
the optimal number of iterations. OOB estimates are usually very pessimistic thus
we recommend to use cross-validation instead and only use OOB if cross-validation
is too time consuming.

.. topic:: Examples:

 * :ref:`example_ensemble_plot_gradient_boosting_regularization.py`
 * :ref:`example_ensemble_plot_gradient_boosting_oob.py`

Interpretation
--------------

Individual decision trees can be interpreted easily by simply
visualizing the tree structure. Gradient boosting models, however,
comprise hundreds of regression trees thus they cannot be easily
interpreted by visual inspection of the individual trees. Fortunately,
a number of techniques have been proposed to summarize and interpret
gradient boosting models.

Feature importance
..................

Often features do not contribute equally to predict the target
response; in many situations the majority of the features are in fact
irrelevant.
When interpreting a model, the first question usually is: what are
those important features and how do they contributing in predicting
the target response?

Individual decision trees intrinsically perform feature selection by selecting
appropriate split points. This information can be used to measure the
importance of each feature; the basic idea is: the more often a
feature is used in the split points of a tree the more important that
feature is. This notion of importance can be extended to decision tree
ensembles by simply averaging the feature importance of each tree (see
:ref:`random_forest_feature_importance` for more details).

The feature importance scores of a fit gradient boosting model can be
accessed via the ``feature_importances_`` property::

    >>> from sklearn.datasets import make_hastie_10_2
    >>> from sklearn.ensemble import GradientBoostingClassifier

    >>> X, y = make_hastie_10_2(random_state=0)
    >>> clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,
    ...     max_depth=1, random_state=0).fit(X, y)
    >>> clf.feature_importances_  # doctest: +ELLIPSIS
    array([ 0.11,  0.1 ,  0.11,  ...

.. topic:: Examples:

 * :ref:`example_ensemble_plot_gradient_boosting_regression.py`

.. currentmodule:: sklearn.ensemble.partial_dependence

Partial dependence
..................

Partial dependence plots (PDP) show the dependence between the target response
and a set of 'target' features, marginalizing over the
values of all other features (the 'complement' features).
Intuitively, we can interpret the partial dependence as the expected
target response [1]_ as a function of the 'target' features [2]_.

Due to the limits of human perception the size of the target feature
set must be small (usually, one or two) thus the target features are
usually chosen among the most important features.

The Figure below shows four one-way and one two-way partial dependence plots
for the California housing dataset:

.. figure:: ../auto_examples/ensemble/images/plot_partial_dependence_001.png
   :target: ../auto_examples/ensemble/plot_partial_dependence.html
   :align: center
   :scale: 70

One-way PDPs tell us about the interaction between the target
response and the target feature (e.g. linear, non-linear).
The upper left plot in the above Figure shows the effect of the
median income in a district on the median house price; we can
clearly see a linear relationship among them.

PDPs with two target features show the
interactions among the two features. For example, the two-variable PDP in the
above Figure shows the dependence of median house price on joint
values of house age and avg. occupants per household. We can clearly
see an interaction between the two features:
For an avg. occupancy greater than two, the house price is nearly independent
of the house age, whereas for values less than two there is a strong dependence
on age.

The module :mod:`partial_dependence` provides a convenience function
:func:`~sklearn.ensemble.partial_dependence.plot_partial_dependence`
to create one-way and two-way partial dependence plots. In the below example
we show how to create a grid of partial dependence plots: two one-way
PDPs for the features ``0`` and ``1`` and a two-way PDP between the two
features::

    >>> from sklearn.datasets import make_hastie_10_2
    >>> from sklearn.ensemble import GradientBoostingClassifier
    >>> from sklearn.ensemble.partial_dependence import plot_partial_dependence

    >>> X, y = make_hastie_10_2(random_state=0)
    >>> clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,
    ...     max_depth=1, random_state=0).fit(X, y)
    >>> features = [0, 1, (0, 1)]
    >>> fig, axs = plot_partial_dependence(clf, X, features) #doctest: +SKIP

For multi-class models, you need to set the class label for which the
PDPs should be created via the ``label`` argument::

    >>> from sklearn.datasets import load_iris
    >>> iris = load_iris()
    >>> mc_clf = GradientBoostingClassifier(n_estimators=10,
    ...     max_depth=1).fit(iris.data, iris.target)
    >>> features = [3, 2, (3, 2)]
    >>> fig, axs = plot_partial_dependence(mc_clf, X, features, label=0) #doctest: +SKIP

If you need the raw values of the partial dependence function rather
than the plots you can use the
:func:`~sklearn.ensemble.partial_dependence.partial_dependence` function::

    >>> from sklearn.ensemble.partial_dependence import partial_dependence

    >>> pdp, axes = partial_dependence(clf, [0], X=X)
    >>> pdp  # doctest: +ELLIPSIS
    array([[ 2.46643157,  2.46643157, ...
    >>> axes  # doctest: +ELLIPSIS
    [array([-1.62497054, -1.59201391, ...

The function requires either the argument ``grid`` which specifies the
values of the target features on which the partial dependence function
should be evaluated or the argument ``X`` which is a convenience mode
for automatically creating ``grid`` from the training data. If ``X``
is given, the ``axes`` value returned by the function gives the axis
for each target feature.

For each value of the 'target' features in the ``grid`` the partial
dependence function need to marginalize the predictions of a tree over
all possible values of the 'complement' features. In decision trees
this function can be evaluated efficiently without reference to the
training data. For each grid point a weighted tree traversal is
performed: if a split node involves a 'target' feature, the
corresponding left or right branch is followed, otherwise both
branches are followed, each branch is weighted by the fraction of
training samples that entered that branch. Finally, the partial
dependence is given by a weighted average of all visited leaves. For
tree ensembles the results of each individual tree are again
averaged.

.. rubric:: 脚注

.. [1] For classification with ``loss='deviance'``  the target
   response is logit(p).

.. [2] More precisely its the expectation of the target response after
   accounting for the initial model; partial dependence plots
   do not include the ``init`` model.

.. topic:: 示例:

 * :ref:`example_ensemble_plot_partial_dependence.py`


.. topic:: 参考

 .. [F2001] J. Friedman, "Greedy Function Approximation: A Gradient Boosting Machine",
   The Annals of Statistics, Vol. 29, No. 5, 2001.

 .. [F1999] J. Friedman, "Stochastic Gradient Boosting", 1999

 .. [HTF2009] T. Hastie, R. Tibshirani and J. Friedman, "Elements of Statistical Learning Ed. 2", Springer, 2009.

 .. [R2007] G. Ridgeway, "Generalized Boosted Models: A guide to the gbm package", 2007
