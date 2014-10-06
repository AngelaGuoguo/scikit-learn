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

:class:`RandomTreesEmbedding` 采用无监督的数据转换，是一个完全随机树的森林。其将每一个数据编码到每个叶枝点上。这个编码的是按照k中选一个的方式进行的，所以会产生一个高维度的稀疏二进制编码。而这个编码非常迅速，所以可以作为其他学习过程的基本。编码的大小和稀疏性是由树的数目和深度决定的。编码的最大数目是 ``n_estimators * 2 ** max_depth`` ，即森林中叶枝点的数目。

当相邻的数据点会更加倾向于处在同一个叶枝点，编码转换会采用一个非参数的密度估计。

.. topic:: 示例:

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

:mod:`sklearn.ensemble` 包含了一个常用的增强方法， AdaBoost, 由 Freund and Schapire 在 1995年提出 [FS1995]_.

AdaBoost的核心是去对不断重复的数据做拟合一系列弱的统计（如只比随机猜测好一点的模型，譬如小的决策树）。在预测时，是通过加权的多数分类（或者求和）来做出最后的分类选择。重复的数据在一个称之为增强的阶段获得不同的权重 :math:`w_1`, :math:`w_2`, ..., :math:`w_N` 作为N个训练样本。初始的时候这些权重均为 :math:`w_i = 1/N` ，所以第一步是在原始数据上拟合这些弱的统计。接下来的每步，样本的权重会更改，进而学习算法重新应用到这些样本上去。在每一步，此前错误预测的训练样本的权重将会增加，而正确预测的样本的权重将会减少。此后，那些难以预测的样本将逐步增加权重。即这些弱的统计量将会分配更多的权重到迁移那些预测错误的取样中（参见 [HTF]_ ）。

.. figure:: ../auto_examples/ensemble/images/plot_adaboost_hastie_10_2_001.png
   :target: ../auto_examples/ensemble/plot_adaboost_hastie_10_2.html
   :align: center
   :scale: 75

AdaBoost 可以用到分类和回归问题中：

  - 对于多分类问题， :class:`AdaBoostClassifier` 采用 AdaBoost-SAMME 和 AdaBoost-SAMME.R [ZZRH2009]_ 。

  - 对于回归问题， :class:`AdaBoostRegressor` 采用 AdaBoost.R2 [D1997]_.

用法
-----

下例展示如何在 AdaBoost 分类器中应用100个弱统计 ::

    >>> from sklearn.cross_validation import cross_val_score
    >>> from sklearn.datasets import load_iris
    >>> from sklearn.ensemble import AdaBoostClassifier

    >>> iris = load_iris()
    >>> clf = AdaBoostClassifier(n_estimators=100)
    >>> scores = cross_val_score(clf, iris.data, iris.target)
    >>> scores.mean()                             # doctest: +ELLIPSIS
    0.9...

参数 ``n_estimators`` 调节弱统计的数目。参数 ``learning_rate`` 控制每个弱统计在最终的贡献。默认的设置是弱统计是一层决策树。不同的若统计可以通过 ``base_estimator`` 参数来控制。主要需要调节的参数是 ``n_estimators`` 和弱统计的复杂度参数（如在决策树中，深度 ``max_depth`` 或者叶节点上的最小取样数 ``min_samples_leaf`` ）。

.. topic:: 示例:

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

梯度树增强
======================

`梯度树增强 <http://en.wikipedia.org/wiki/Gradient_boosting>`_
或者梯度增强回归树（Gradient Boosted Regression Trees GBRT）是一类将增强扩展到任意可微分成本函数。GBRT是一个可以用来回归和分类的准确且有效的现成方法。 其应用有网页检索排序和生态学。

GBRT的优势有：

  + 对于不同类型的数据有统一处理 （混杂heterogeneous特征）

  + 预测的能力

  + 对输出异常稳定（通过稳定的成本函数）

GBRT的劣势有：

  + 扩展性。由于是增强需要顺序操作，因此很难并行化。

:mod:`sklearn.ensemble` 提供了分类和回归的GBRT。

分类
---------------

:class:`GradientBoostingClassifier` 支持二分类，或者多分类问题。下面的例子展示了利用100个弱统计的方法::

    >>> from sklearn.datasets import make_hastie_10_2
    >>> from sklearn.ensemble import GradientBoostingClassifier

    >>> X, y = make_hastie_10_2(random_state=0)
    >>> X_train, X_test = X[:2000], X[2000:]
    >>> y_train, y_test = y[:2000], y[2000:]

    >>> clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,
    ...     max_depth=1, random_state=0).fit(X_train, y_train)
    >>> clf.score(X_test, y_test)                 # doctest: +ELLIPSIS
    0.913...

弱统计的数目（如回归树）是由系数 ``n_estimators`` 控制。 :ref:`树的大小 <gradient_boosting_tree_size>` 是通过设定树的深度 ``max_depth`` 或叶节点的最大数目 ``max_leaf_nodes`` 来确定。学习速率 ``learning_rate`` 是一个在 (0.0, 1.0] 间的超函数，用来控制过度拟合，参见（ :ref:`收缩 <gradient_boosting_shrinkage>` ）。

.. note::

   对于分类超过两类的问题需要在回归树中引入 ``n_classes`` 。 因此总共树的数目为 ``n_classes * n_estimators`` 。对于一个有着很多分类的样本，我们强烈建议采用 :class:`RandomForestClassifier` 替代 :class:`GradientBoostingClassifier` 。

回归
----------

:class:`GradientBoostingRegressor` 通过设定参数 ``loss`` 来选择 :ref:`不同的成本函数 <gradient_boosting_loss>` 来解决回归问题。默认的成本函数树 ``ls`` ，即最小二乘法::

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

下图展示了 :class:`GradientBoostingRegressor` 通过500个弱统计来分析Boston房价数据（ :func:`sklearn.datasets.load_boston` ）。作图展示了每一步的训练和测试误差。每一步的训练误差被存储在模型的 :attr:`~GradientBoostingRegressor.train_score_` 属性中。测试的误差可以通过 :meth:`~GradientBoostingRegressor.staged_predict` 方法来比较模型的预测。这个图可以用来寻找最佳树的数目（ ``n_estimators`` ）。右图展示了特征的重要性，其存储在 ``feature_importances_`` 。

.. figure:: ../auto_examples/ensemble/images/plot_gradient_boosting_regression_001.png
   :target: ../auto_examples/ensemble/plot_gradient_boosting_regression.html
   :align: center
   :scale: 75

.. topic:: 示例:

 * :ref:`example_ensemble_plot_gradient_boosting_regression.py`
 * :ref:`example_ensemble_plot_gradient_boosting_oob.py`

.. _gradient_boosting_warm_start:

拟合额外的弱统计
--------------------------------

:class:`GradientBoostingRegressor` 和 :class:`GradientBoostingClassifier` 都支持 ``warm_start=True`` 来添加进一步的模型到已拟合的模型中去::

  >>> _ = est.set_params(n_estimators=200, warm_start=True)  # set warm_start and new nr of trees
  >>> _ = est.fit(X_train, y_train) # fit additional 100 trees to est
  >>> mean_squared_error(y_test, est.predict(X_test))    # doctest: +ELLIPSIS
  3.84...

.. _gradient_boosting_tree_size:

控制树的大小
-------------------------

The size of the regression tree base learners defines the level of variable
interactions that can be captured by the gradient boosting model. 
回归树弱统计的数目决定了在梯度增强模型中变量间相互作用的机会。大致上一个深度为 ``h`` 的树可以刻画 ``h`` 阶的相互作用。此间，有两种方法可以控制树的大小。

如果设定 ``max_depth=h`` ，那么深度为 ``h`` 将会被拟合。这样的树将会有（最多） ``2**h`` 个叶节点，和 ``2**h - 1`` 个决策点。

另一方面，你可以通过每个叶点的最大样本数 ``max_leaf_nodes`` ，来决定树的大小。这时，树是通过选择那些有着最大改善程度的决策进行扩展。当树 ``max_leaf_nodes=k`` 拥有 ``k - 1`` 个决策点时，模型的相互作用的阶数为： ``max_leaf_nodes - 1`` 。

我们发现 ``max_leaf_nodes=k`` 和 ``max_depth=k-1`` 的结果相近，但是速度会有显著提升。参数 ``max_leaf_nodes`` 相当于在文献 [F2001]_ 中梯度增强部分的变量 ``J`` ，并与R的gbm包中的 ``interaction.depth`` 有如下关系： ``max_leaf_nodes == interaction.depth + 1`` 。

数学基础
-------------------------

GBRT 采用如下的叠加模型：

  .. math::

    F(x) = \sum_{m=1}^{M} \gamma_m h_m(x)

其中 :math:`h_m(x)` 是基础模型，通常在增强方法中被称为 *弱统计 （weak learner）* 。梯度树增强采用固定大小的 :ref:`决策树 <tree>` 作为弱统计。决策树的优势在于可以处理不同类型的数据，和复杂的二模型。

与其他增强算法类似，GBRT通过向前递进的方式构建模型：

  .. math::

    F_m(x) = F_{m-1}(x) + \gamma_m h_m(x)

在每一步，决策树 :math:`h_m(x)` 都会基于当前模型 :math:`F_{m-1}` 来最小化成本函数 :math:`L` 。

  .. math::

    F_m(x) = F_{m-1}(x) + \arg\min_{h} \sum_{i=1}^{n} L(y_i,
    F_{m-1}(x_i) - h(x))

初始模型 :math:`F_{0}` 是由问题决定的。如最小二乘法回归时，统产选择目标值的平均。

.. note:: 初始模型可以由参数 ``init`` 设置，该类需要具有函数 ``fit`` 和 ``predict`` 。

梯度增强尝试通过梯度算法解决下面的最小化问题。最陡的方向通过对成本函数的微分得到:

  .. math::

    F_m(x) = F_{m-1}(x) + \gamma_m \sum_{i=1}^{n} \nabla_F L(y_i,
    F_{m-1}(x_i))

其中步长 :math:`\gamma_m` 是通过线性搜索确定：

  .. math::

    \gamma_m = \arg\min_{\gamma} \sum_{i=1}^{n} L(y_i, F_{m-1}(x_i)
    - \gamma \frac{\partial L(y_i, F_{m-1}(x_i))}{\partial F_{m-1}(x_i)})

回归和分类的差别仅在于成本函数的选择。

.. _gradient_boosting_loss:

成本函数
...............

下面的函数均可以通过参数 ``loss`` 进行选择：

  * 回归

    * 最小二乘法 （Least squares ``'ls'`` ）：回归问题的自然选择，初始模型是目标值的平均。
    * 最小绝对偏差（Least absolute deviation ``'lad'`` ）：一个稳定的成本函数，其初始模型是目标值的中值。
    * Huber (``'huber'``)：另一类稳定的成本函数，结合了上述两种方式。通过 ``alpha`` 来控制对异常值的敏感度（参见 [F2001]_ ）。
    * Quantile (``'quantile'``)：一类用来做分位数回归的成本函数，调节 ``0 < alpha < 1`` 来控制分位数。这个成本函数可以用来做预测区间（参见 :ref:`example_ensemble_plot_gradient_boosting_quantile.py` ）。

  * 分类

    * 二项式偏离 (``'deviance'``)：负的二项式对数概率成本函数是针对二元分类问题（提供概率估计）。初始模型提供相对概率分布。
    * 多项式偏离 (``'deviance'``): 通过 ``n_classes`` 来控制多项式对数概率的成本函数。其提供对互斥分类的概率估计。初始模型是每个类型的先验概率。每一步 ``n_classes`` 个回归树将本构建。因此对于大数据，其效率较低。
    * 指数成本 (``'exponential'``)：与 :class:`AdaBoostClassifier` 的成本函数一致。对于错误标记的取样不如二项式偏离稳定。只能被用于二元分类。

复杂度控制
----------------

.. _gradient_boosting_shrinkage:

收缩
..........

[F2001]_ 提出一个简单的控制复杂度的方法，对每一个弱统计加以系数 :math:`\nu`:

.. math::

    F_m(x) = F_{m-1}(x) + \nu \gamma_m h_m(x)

:math:`\nu` 被称为 **学习效率** ，因为它控制了每一步学习步长。它可以通过 ``learning_rate`` 来设定。

``learning_rate`` 与若统计的数目参数 ``n_estimators`` 有很强相关。一个较小的 ``learning_rate`` 需要很多的弱统计来维持同样的误差。经验证据表明，小的 ``learning_rate`` 会有更高的准确度。 [HTF2009]_ 推荐设置 ``learning_rate <= 0.1`` 并逐步调节 ``n_estimators`` 关于``learning_rate`` 与 ``n_estimators`` 的相互影响，参见 [R2007]_ 。

子取样
............

[F1999]_ 提出随机梯度增强，结合了bagging的思想。在每一步，弱统计都是应用到一部分子取样的数据上。这部分子取样不重复取样。一个通常的选择是 ``subsample=0.5`` 。

下图展示了收缩和子取样对模型拟合的影响。我们可以看到有收缩时结果更好，包括子取样后会更进一步提高准确度。没有收缩的子取样，却表现不佳。

.. figure:: ../auto_examples/ensemble/images/plot_gradient_boosting_regularization_001.png
   :target: ../auto_examples/ensemble/plot_gradient_boosting_regularization.html
   :align: center
   :scale: 75

另一个降低方差的方法是类似 :class:`RandomForestClassifier` 对特征进行取样。这个取样数目可以由``max_features`` 进行控制。

.. note:: 采用较小的 ``max_features`` 可以显著降低计算时间。

随机梯度增强可以计算out-of-bag的测试样本偏差。这是通过计算不在bootstrap取样中的数据的提高来实现的。这个是存储在 :attr:`~GradientBoostingRegressor.oob_improvement_` 。 ``oob_improvement_[i]`` 记录着在第i步是加入所带来的提高。这个OOB的改善可以用来做模型选择，如选择最佳的计算步数。OOB的统计通常比较悲观，所以我们推荐当交叉检验不是很耗时时，使用交叉检验。

.. topic:: 示例:

 * :ref:`example_ensemble_plot_gradient_boosting_regularization.py`
 * :ref:`example_ensemble_plot_gradient_boosting_oob.py`

阐释
--------------

每一个独立的决策树可以很容易图像展示和理解。梯度增强树则包含了上百个回归树，因此很难轻易的图像化。幸好，我们有一些技术来概括和理解梯度增强模型。

特征重要性
..................

通常，特征对模型预测的贡献并不是相同的。大多数时候，大部分特征都是不相关的。当理解一个模型，第一个问题通常是：那些特征是重要的，他们是如何影响预测的。

单独的决策树通过在每个决策点选择相应的特征。这个信息可以用来测量特征的重要性。其基本想法是：常见的特征则更为重要。将这个概念扩展到决策树集的化，我们可以简单的平均每一个特征在所有树中的重要性（参见 :ref:`random_forest_feature_importance` ）。

特征重要性评分存储在模型的 ``feature_importances_`` 变量之中::

    >>> from sklearn.datasets import make_hastie_10_2
    >>> from sklearn.ensemble import GradientBoostingClassifier

    >>> X, y = make_hastie_10_2(random_state=0)
    >>> clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,
    ...     max_depth=1, random_state=0).fit(X, y)
    >>> clf.feature_importances_  # doctest: +ELLIPSIS
    array([ 0.11,  0.1 ,  0.11,  ...

.. topic:: 示例:

 * :ref:`example_ensemble_plot_gradient_boosting_regression.py`

.. currentmodule:: sklearn.ensemble.partial_dependence

部分依赖
..................

部分依赖图 (PDP) 展示了目标响应与一系列‘目标’特征之间的关系。直觉上，我们可以理解部分依赖为目标相应 [1]_ 是‘目标’特征的函数 [2]_ 。

由于人有限的理解力，目标特征需要比较小（通常只有一，两个），并且目标特征通常为最重要的特征。

下图展示了四个单路和一个双路部分依赖图：

.. figure:: ../auto_examples/ensemble/images/plot_partial_dependence_001.png
   :target: ../auto_examples/ensemble/plot_partial_dependence.html
   :align: center
   :scale: 70

单路部分依赖图告诉我们目标相应和目标特征间的关系（如，线性，非线性）。在上图中，坐上展示了一个区域中中值收入对房价中值的影响。我们可以看到明显的线性关系。

两个变量的部分依赖图展示了两个特征间的相互关系。例如上图中，房屋的中间价是房龄和平均入住数的函数。我们可以清晰的看到两个特征间的相互作用：如果平均入住数大于2，那么房屋的价格基本与房龄无关，然而当入住数小于2，那么有高相关。

:mod:`partial_dependence` 提供了一个便捷的函数 :func:`~sklearn.ensemble.partial_dependence.plot_partial_dependence` 来绘制单路和双路部分依赖图。在下面的例子中，我们将绘制一系列部分依赖图，两个单路图和一个双路图::

    >>> from sklearn.datasets import make_hastie_10_2
    >>> from sklearn.ensemble import GradientBoostingClassifier
    >>> from sklearn.ensemble.partial_dependence import plot_partial_dependence

    >>> X, y = make_hastie_10_2(random_state=0)
    >>> clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,
    ...     max_depth=1, random_state=0).fit(X, y)
    >>> features = [0, 1, (0, 1)]
    >>> fig, axs = plot_partial_dependence(clf, X, features) #doctest: +SKIP

对于多分类模型，你需要通过 ``label`` 设置那些分类将被画出::

    >>> from sklearn.datasets import load_iris
    >>> iris = load_iris()
    >>> mc_clf = GradientBoostingClassifier(n_estimators=10,
    ...     max_depth=1).fit(iris.data, iris.target)
    >>> features = [3, 2, (3, 2)]
    >>> fig, axs = plot_partial_dependence(mc_clf, X, features, label=0) #doctest: +SKIP

如果你需要原始的部分依赖函数而不是图，那么请选择函数 :func:`~sklearn.ensemble.partial_dependence.partial_dependence` ::

    >>> from sklearn.ensemble.partial_dependence import partial_dependence

    >>> pdp, axes = partial_dependence(clf, [0], X=X)
    >>> pdp  # doctest: +ELLIPSIS
    array([[ 2.46643157,  2.46643157, ...
    >>> axes  # doctest: +ELLIPSIS
    [array([-1.62497054, -1.59201391, ...

这个函数需要 ``grid`` 来明确那些特征将被画出，或者便捷模式 ``X`` 来自动画出。如果给定 ``X`` ，那么将返回 ``axis`` 来代表每一个目标特征。

对于目标特征中的每一个值， ``grid`` 都需要里边所有其他特征的可能值。在决策树中，这个计算可以很快的完成，而不需要原始数据。每一个加权的树都将被遍历：如果一个节点需要‘目标’特征，那么其左右枝将被进一步探索，否则两枝都将被探索，每支的权重正比于进入该分支的样本数。最终，部分依赖图是探索分支的加权平均。对于树集，所有单独的树将被进一步平均。

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
