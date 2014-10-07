.. _mixture:

.. _gmm:

===================================================
高斯混合模型
===================================================

.. currentmodule:: sklearn.mixture

`sklearn.mixture` 是进行高斯混合模型学习的模板（支持对角，球 ，tied和完全协方差矩阵）。其也提供估算组分数目的辅助工具。

 .. figure:: ../auto_examples/mixture/images/plot_gmm_pdf_001.png
   :target: ../auto_examples/mixture/plot_gmm_pdf.html
   :align: center
   :scale: 50%

   **两组分高斯混合模型：** *数据点和模型的等概率面*。

高斯混合模型是一个概率模型，其假设所有的数据点是服从若干个高斯分布的组合。我们可以认为混合模型是将k-平均聚类算法推广到包含数据的协方差的高斯分布。

Scikit-learn采用不同的统计方式来进行模型拟合，以下介绍不同的类。

GMM分类器
===============

:class:`GMM` 采用 :ref:`预期最大化 <expectation_maximization>` （expectation maximization， EM）的算法来拟合高斯混合模型。其可以绘制多变量模型的置信椭球和通过计算贝叶斯信息准则（BIC）得到聚类的数目。 :meth:`GMM.fit` 函数用来从训练样本中拟合模型。对于测试数据，通过 :meth:`GMM.predict` 函数估计数据所属的高斯类别。

.. note::
  
    此外样本在每一个高斯分布中的概率可以通过函数 :meth:`GMM.predict_proba` 获得。

:class:`GMM` 通过不同的选项来控制高斯分布的协方差矩阵：球形，对角，tied和完整协方差。

.. figure:: ../auto_examples/mixture/images/plot_gmm_classifier_001.png
   :target: ../auto_examples/mixture/plot_gmm_classifier.html
   :align: center
   :scale: 75%

.. topic:: 示例 :

    * See :ref:`example_mixture_plot_gmm_classifier.py` for an example of
      using a GMM as a classifier on the iris dataset.

    * See :ref:`example_mixture_plot_gmm_pdf.py` for an example on plotting the 
      density estimation.

:class:`GMM` 的优缺点：预期最大化推测
------------------------------------------------------------------------

优势
.....

:速度: 最快的学习混合模型的算法

:不可知论: 由于这个算法将概率最小化，因此它不会将平均人为的置为0，也不会导致聚类的大小偏向一个不合适的尺度。

劣势
....

:奇异点: 如果每个混合模型的数据点都不足，那么协方差矩阵将难以估计，而算法将发散，进而无法求解正确的聚类。此时需要人为干预。

:组分的数目: 这个算法总是使用全部给定的组分数目。因而需要额外的信息来调节。

经典GMM中对组分数目的选择
------------------------------------------------------

BIC准则是一个有效的选择GMM中组分数目的方法。理论上，其可以无限趋近真实组分的数目（当数据足够大时）。注意在使用 :ref:`DPGMM <dpgmm>` 时不必再给定组分数目。

.. figure:: ../auto_examples/mixture/images/plot_gmm_selection_001.png
   :target: ../auto_examples/mixture/plot_gmm_selection.html
   :align: center
   :scale: 50%

.. topic:: 示例 :

    * See :ref:`example_mixture_plot_gmm_selection.py` for an example
      of model selection performed with classical GMM.

.. _expectation_maximization:

预期最大化的估计算法
-----------------------------------------------

对于高斯混合模型学习的主要困难在于无法从每个无标记的数据点得知其属于哪一个潜在的组分（如果我们知道这个信息，那么问题将会非常简单，只需要对每个分类拟合一个高斯分布）。 `预期最大化 <http://en.wikipedia.org/wiki/Expectation%E2%80%93maximization_algorithm>`_ 是一个解决这个问题的牢靠的统计算法。首先我们假设随机的组分（以数据为中心的高斯分布，k-平均，或者只是中心在原点的分布），再计算数据点在每一个组分中的概率。接下来，我们调整组分的系数来最大化概率。重复这个过程直到结果收敛到一个局域最佳。


VBGMM分类器：变分高斯混合模型
================================================

:class:`VBGMM` 采用 :ref:`变分推理 <variational_inference>` 算法来拟合高斯混合模型。这个方法的使用与 :class:`GMM` 一致。它是介于 :class:`GMM` 和 :class:`DPGMM` 之间，并具有Dirichlet过程的性质。

:class:`VBGMM` 的优缺点：变分推理
-------------------------------------------------------------

优势
.....

:复杂度: 由于包含了先验信息，变分解比预期最大化解更为稳定。我们可以应用完全协变矩阵信息，即使数据的维度很高，或者有些组分仅会包含一个数据点。

缺点
.....

:偏差: 由于对复杂度的控制，这样会导致偏差。 这个辩分方法会导致所有的平均值均趋向于0（部分由于先验信息加入了一个在原点的“影子数据”到每个混合组分中）。并且会导致协方差变圆。取决于系数，其还会导致聚类的结构要么更均匀或者更集中。

:超系数: 这个算法需要一个额外的超系数，通过经验调节或者交叉检验。

.. _variational_inference:

估算算法：变分推理
---------------------------------------------

变分推理是预期最大化的一个扩展，其最大化模型概然的下界（包含先验概率）而不是数据的概率 。其原理与预期最大化一样（逐步改善概率来寻找每个组分的最佳系数），但变分方法通过系数的先验分布加入了复杂度参数。这可以避免预期最大化方法中的奇异性，但会引入一定的误差。拟合过程会明显变慢，但是还是有一定的实际应用意义。

由于其贝叶斯的特质，变分方法需要比预期最大化方法更多的超系数。其中最重要的系数是聚合系数 ``alpha`` ，一个较大的alpha往往会导致大小相近的混合模型，而一个较小的值（介于0和1）则会让一些组分涵盖大部分数据，而剩下的模型仅包含几个剩余的数据。

.. _dpgmm:

DPGMM分类器：无限高斯混合
============================================

:class:`DPGMM` 通过Dirichlet过程来控制组分的数目（受限）。其使用与 :class:`GMM` 一致，不需要用户给定组分的数目，而是数目上限和聚合系数。其计算时间会更长。

.. |plot_gmm| image:: ../auto_examples/mixture/images/plot_gmm_001.png
   :target: ../auto_examples/mixture/plot_gmm.html
   :scale: 48%

.. |plot_gmm_sin| image:: ../auto_examples/mixture/images/plot_gmm_sin_001.png
   :target: ../auto_examples/mixture/plot_gmm_sin.html
   :scale: 48%

.. centered:: |plot_gmm| |plot_gmm_sin|

上面的例子对比了不同的高斯混合模型。 **左图** 对只有两个类别的数据，GMM拟合了5个组分。我们可以看到DPGMM限制自身，成功的找到了两个组分，而GMM的结果组分过多了。注意，当样本较少时，DPGMM会比较保守，有的时候仅会划分一个组分。 **右图** 我们拟合一个并不适合高斯混合模型的数据。调节系数 `alpha` ，DPGMM会相应产生不同的组分数目。

.. topic:: 示例 :

    * See :ref:`example_mixture_plot_gmm.py` for an example on plotting the
      confidence ellipsoids for both :class:`GMM` and :class:`DPGMM`.

    * :ref:`example_mixture_plot_gmm_sin.py` shows using :class:`GMM` and
      :class:`DPGMM` to fit a sine wave

:class:`DPGMM` 的优缺点： Dirichlet过程混合模型
----------------------------------------------------------------------

优势
.....

:对参数数目不敏感: 不像有限模型采用所有的组分，不同的组分数目会产生很不一样的结果，Dirichlet过程的解不会因为参数而产生很大的变化。因此其更稳定且减少调节。

:不需要给定组分数目: 只需要一个组分数目的上限。注意DPGMM并不是一个有效的特征选择方法，不能依赖这方面的结果。

劣势
.....

:速度: 额外对变分推理和Dirichlet过程的计算会导致拟合过程更加缓慢，虽然不多。

:偏差: 类似于变分方法，但是有稍微多一点的偏差。在Dirichlet过程中有很多隐含的偏差。当这些偏差与数据不符时，可能有限的混合模型会有更好的结果。

.. _dirichlet_process:

Dirichlet过程
---------------------

这里我们介绍一下Dirichlet过程中的变分推理算法。Dirichlet过程是一个 *无限，无限制分割聚类* 的先验概率分布。变分方法使得我们可以应用这个先验结构到高斯混合模型中去，而不会显著影响计算时间。

一个重要的问题是Dirichlet过程是如何在一个无限，不限制聚类数目的情况下保持一致的。一个完整的阐述并不适合这个手册，但我们可以想想 `中餐馆过程 <http://en.wikipedia.org/wiki/Chinese_restaurant_process>`_ 的类比来理解它。中餐馆过程是一个对Dirichlet过程的概括性描述。想象一个中餐馆有无限张桌子，一开始它们全是空的。当第一个客人道德时候，他可以坐在第一个桌子上。接下来的每一个客户要么坐到已经被占的桌子上，概率正比于客户数，或者一个完全新的桌子上，概率正比于聚合系数 `alpha` 。当有限数目的客人就坐后，很容易可以看到只有有限数目的桌子将会被用到。而越大的 `alpha` 会需要越多的桌子。所以Dirichlet过程在聚类中不限制混合组分的数目，但假设一个非对称的先验分布结构来限制数据点被分类的方式。

变分推理技术在Dirichlet过程中对这个无限的混合模型采用一个有限的近似。我们不需要给定一个组分数目的先验估计，但需要给定聚合系数和一个组分数目的上限（这个上限需要高于真实组分的数目。其仅影响计算的复杂度）。

.. topic:: Derivation:

   * `这里 <dp-derivation.html>`是本算法的完整推导 `_ 。

.. toctree::
    :hidden:

    dp-derivation.rst


