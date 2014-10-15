
.. currentmodule:: sklearn.manifold

.. _manifold:

=================
流形学习
=================

.. rst-class:: quote

                 | Look for the bare necessities
                 | The simple bare necessities
                 | Forget about your worries and your strife
                 | I mean the bare necessities
                 | Old Mother Nature's recipes
                 | That bring the bare necessities of life
                 |
                 |             -- Baloo's song [The Jungle Book]



.. figure:: ../auto_examples/manifold/images/plot_compare_methods_001.png
   :target: ../auto_examples/manifold/plot_compare_methods.html
   :align: center
   :scale: 60

流行学习是一种非线性维度降低的方法。其算法的想法是很多数据的维度只是虚假的高。


引言
============

高维度的数据往往难于可视化。对于二维或者三维的数据，很容易通过绘图来表述其内在的结构，而高维数据很难如此直接。为了能够进行可视化，我们往往需要将维度降低。

最简单的方法是将数据随机投影。尽管这个可以一定程度上的将数据可视化，但是随机选择会丧失很多细节。

.. |digits_img| image:: ../auto_examples/manifold/images/plot_lle_digits_001.png
    :target: ../auto_examples/manifold/plot_lle_digits.html
    :scale: 50

.. |projected_img| image::  ../auto_examples/manifold/images/plot_lle_digits_002.png
    :target: ../auto_examples/manifold/plot_lle_digits.html
    :scale: 50

.. centered:: |digits_img| |projected_img|

为了克服这个问题，设计了很多监督的和非监督的线性维度降低方案，如主成分分析（Principal Component Analysis PCA），独立分量分析（Independent Component Analysis），线性判别分析（Linear Discriminant Analysis）以及其他。这些方法都是通过选择一个“有意义”的线性投影方向。这些方法很有效，但是常常会丢失非线性信息。

.. |PCA_img| image:: ../auto_examples/manifold/images/plot_lle_digits_003.png
    :target: ../auto_examples/manifold/plot_lle_digits.html
    :scale: 50

.. |LDA_img| image::  ../auto_examples/manifold/images/plot_lle_digits_004.png
    :target: ../auto_examples/manifold/plot_lle_digits.html
    :scale: 50

.. centered:: |PCA_img| |LDA_img|

流形学习（Manifold Learning）可以视作对PCA方法的概括，来进一步描述数据中的非线性特征。虽然有监督学习的方法，但是典型的流形学习问题是非监督的：其需要从高维数据中自行找到结构，而不依赖于外在的分类信息。

.. topic:: 示例

    * See :ref:`example_manifold_plot_lle_digits.py` for an example of
      dimensionality reduction on handwritten digits.

    * See :ref:`example_manifold_plot_compare_methods.py` for an example of
      dimensionality reduction on a toy "S-curve" dataset.

以下介绍在sklearn中采用的流形学习方法。

.. _isomap:

等距映射 Isomap
==============

最早采用的流行学习方法是等距映射（Isometric Mapping Isomap）。Isomap可以视作是多维比例（Multi-dimensional Scaling MDS）或者核PCA的拓展。Isomap寻找一个更低维度的嵌套，同时保持各个点间的距离一致。Isomap通过 :class:`Isomap` 实现。

.. figure:: ../auto_examples/manifold/images/plot_lle_digits_005.png
   :target: ../auto_examples/manifold/plot_lle_digits.html
   :align: center
   :scale: 50

计算复杂性
----------
Isomap算法有三步：

1. **最近邻搜索**  Isomap 通过 :class:`sklearn.neighbors.BallTree` 来进行快速的近邻搜索。其计算复杂度为 :math:`O[D \log(k) N \log(N)]` ，其中 :math:`k` 为近邻数目， :math:`N` 为样本数， :math:`D` 为维度。

2. **搜索最近路线图** 最有效的算法是 *Dijkstra's Algorithm* ，其计算复杂度为 :math:`O[N^2(k + \log(N))]` ，或者 *Floyd-Warshall algorithm* ，复杂度为 :math:`O[N^3]` 。用户可以通过调节参数  ``Isomap`` 中的 ``path_method`` 。如果没有设置，程序尝试采用最快速的方法。

3. **部分本征值分解** 嵌套是通过对应于  :math:`N \times N`  等距映射矩阵最大的 :math:`d` 个本征值的本征向量代表的。对于一个密集矩阵，计算复杂度为 :math:`O[d N^2]` ，通过 ``ARPACK`` 求解。可以通过 ``Isomap`` 的参数 ``eigen_solver`` 求解。如果没有给定，程序自动选择最佳解决方案。

Isomap的总体复杂度为
:math:`O[D \log(k) N \log(N)] + O[N^2(k + \log(N))] + O[d N^2]`.

* :math:`N` : 训练取样数目
* :math:`D` : 输入维度
* :math:`k` : 最近邻数目
* :math:`d` : 输出维度

.. topic:: 参考：

   * `"A global geometric framework for nonlinear dimensionality reduction"
     <http://www.sciencemag.org/content/290/5500/2319.full>`_
     Tenenbaum, J.B.; De Silva, V.; & Langford, J.C.  Science 290 (5500)


局域线性嵌套
========================

局域线性嵌套（Locally linear embedding LLE）寻找一个更低维度的投影，并且保持其局域的近邻。其可以视作一系列局域的PCA再合并为最佳的非线性嵌套。

局域线性嵌套通过函数 :func:`locally_linear_embedding` 或者类 :class:`LocallyLinearEmbedding` 实现。

.. figure:: ../auto_examples/manifold/images/plot_lle_digits_006.png
   :target: ../auto_examples/manifold/plot_lle_digits.html
   :align: center
   :scale: 50

计算复杂性
----------

标准的LLE采用三步：

1. **最近邻搜索** 参见 :ref:`Isomap <isomap>` 

2. **构建权重矩阵**. :math:`O[D N k^3]` 。LEE权重矩阵的构建需要对 :math:`N` 个近邻解决 :math:`k \times k` 的线性方程。

3. **部分本征值分解** 参见 :ref:`Isomap <isomap>` 

标准LLE的计算复杂度为
:math:`O[D \log(k) N \log(N)] + O[D N k^3] + O[d N^2]`.

* :math:`N` : 训练取样数目
* :math:`D` : 输入维度
* :math:`k` : 最近邻数目
* :math:`d` : 输出维度

.. topic:: 参考
   
   * `"Nonlinear dimensionality reduction by locally linear embedding"
     <http://www.sciencemag.org/content/290/5500/2323.full>`_
     Roweis, S. & Saul, L.  Science 290:2323 (2000)


改进的局域线性嵌套
=================================

LLE的一个问题在于无法空寂模型复杂度。当邻居的数目大于数据的维度的时候，描述近邻的矩阵秩不足。对此，标准的LLE采用了一个任意的复杂度参数 :math:`r` ，其根据矩阵的迹。尽管当 :math:`r \to 0` 时，结果收敛到需要的嵌套。但是这不保证当 :math:`r > 0` 一定会收敛到最佳解。这个问题会扰动流形的几何结构。

一个解决这个问题的方法是对每个近邻采用多重的权重向量。这边是 *改进的局域线性嵌套（modified locally linear embedding MLLE）* 的核心。MLLE可以通过函数 :func:`locally_linear_embedding` 或 :class:`LocallyLinearEmbedding` 类中的参数 ``method = 'modified'`` 启用。其要求 ``n_neighbors > n_components`` 。

.. figure:: ../auto_examples/manifold/images/plot_lle_digits_007.png
   :target: ../auto_examples/manifold/plot_lle_digits.html
   :align: center
   :scale: 50
   
计算复杂度
----------

计算复杂度涉及三步：

1. **最近邻搜索** 参见 :ref:`Isomap <isomap>`

2. **构建权重矩阵** 。大致为 :math:`O[D N k^3] + O[N (k-D) k^2]` 。前一部分于LLE一致。第二部分是构建权重矩阵时的不同权重的影响。实际中，这部分额外的计算相对于其他三步来讲很小。

3. **部分本征值分解** 参见 :ref:`Isomap <isomap>`

MLLE的计算复杂度为
:math:`O[D \log(k) N \log(N)] + O[D N k^3] + O[N (k-D) k^2] + O[d N^2]`.

* :math:`N` : 训练取样数目
* :math:`D` : 输入维度
* :math:`k` : 最近邻数目
* :math:`d` : 输出维度

.. topic:: 参考
     
   * `"MLLE: Modified Locally Linear Embedding Using Multiple Weights"
     <http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.70.382>`_
     Zhang, Z. & Wang, J.


Hessian本征映射
====================

Hessian本征映射是一个基于Hessian的LLE方法（HLLE）来解决模型复杂度的问题。其基于Hessian二项式来求解每个近邻的局域线性结构。尽管和其他方法比，其在大数据的情况下会效率低下，但 ``sklearn`` 采用一些算法来使其在输出维度低的时候与其他LLE的方法相当。HLLE可以通过函数 :func:`locally_linear_embedding` 或者
:class:`LocallyLinearEmbedding` 类中的参数 ``method = 'hessian'`` 启用。其要求 ``n_neighbors > n_components * (n_components + 3) / 2`` 。

.. figure:: ../auto_examples/manifold/images/plot_lle_digits_008.png
   :target: ../auto_examples/manifold/plot_lle_digits.html
   :align: center
   :scale: 50
   
计算复杂度
----------

The HLLE algorithm comprises three stages:

1. **最近邻搜索** 参见 :ref:`Isomap <isomap>`

2. **构建权重矩阵** 大致为 :math:`O[D N k^3] + O[N d^6]` 第一项与一般的LLE一致，第二部分是对局域Hessian估计的QR分解。

3. **部分本征值分解** 参见 :ref:`Isomap <isomap>` 

HLLE的计算复杂度为
:math:`O[D \log(k) N \log(N)] + O[D N k^3] + O[N d^6] + O[d N^2]`.

* :math:`N` : 训练取样数目
* :math:`D` : 输入维度
* :math:`k` : 最近邻数目
* :math:`d` : 输出维度

.. topic:: 参考

   * `"Hessian Eigenmaps: Locally linear embedding techniques for
     high-dimensional data" <http://www.pnas.org/content/100/10/5591>`_
     Donoho, D. & Grimes, C. Proc Natl Acad Sci USA. 100:5591 (2003)

.. _spectral_embedding:

谱嵌套
====================

谱嵌套（Spectral Embedding 或 Laplacian Eigenmaps）是计算非线性嵌套的方法。其对图拉普拉斯进行谱分解方法，来寻找一个数据的低维表征。其生成的图可被视作低维流形的在高维空间的近似。最小化成本函数来保证接近的样本在低维流形中依然与彼此接近，即保留局域距离。谱嵌套通过函数 :func:`spectral_embedding` 或者
:class:`SpectralEmbedding` 类应用。

计算复杂度
----------

谱嵌套算法有三步：

1. **构建权重函数** 将数据通过仿射矩阵转换为图表达。

2. **构建图拉普拉斯** 非标准化的图拉普拉斯为 :math:`L = D - A` ，对于标准化的为 :math:`L = D^{-\frac{1}{2}} (D - A) D^{-\frac{1}{2}}` 。

3. **部分本征值分解** 对图拉普拉斯进行本征值分解。

普嵌套的计算复杂度为
:math:`O[D \log(k) N \log(N)] + O[D N k^3] + O[d N^2]`.

* :math:`N` : 训练取样数目
* :math:`D` : 输入维度
* :math:`k` : 最近邻数目
* :math:`d` : 输出维度

.. topic:: 参考

   * `"Laplacian Eigenmaps for Dimensionality Reduction
     and Data Representation" 
     <http://www.cse.ohio-state.edu/~mbelkin/papers/LEM_NC_03.pdf>`_
     M. Belkin, P. Niyogi, Neural Computation, June 2003; 15 (6):1373-1396


局域切空间调整
=============================

尽管不是一个LLE算法，局域切空间调整（Local tangent space alignment LTSA）算法上和LLE归于一类。区别于保留近邻的距离，LTSA尝试通过近邻的切空间来刻画局域的集合，并作全局优化来学习空间的嵌套。LTSA通过函数 :func:`locally_linear_embedding` 或者 :class:`LocallyLinearEmbedding` 中的 ``method = 'ltsa'`` 进行应用。

.. figure:: ../auto_examples/manifold/images/plot_lle_digits_009.png
   :target: ../auto_examples/manifold/plot_lle_digits.html
   :align: center
   :scale: 50

计算复杂度
----------

LTSA算法有三步：

1. **最近邻搜索** 参见 :ref:`Isomap <isomap>`

2. **构建权重矩阵** 大致为 :math:`O[D N k^3] + O[k^2 d]` 其中第一项为LLE的计算成本。

3. **部分本征值分解** 参见 :ref:`Isomap <isomap>` 

The overall complexity of standard LTSA is
:math:`O[D \log(k) N \log(N)] + O[D N k^3] + O[k^2 d] + O[d N^2]`.

* :math:`N` : 训练取样数目
* :math:`D` : 输入维度
* :math:`k` : 最近邻数目
* :math:`d` : 输出维度

.. topic:: 参考

   * `"Principal manifolds and nonlinear dimensionality reduction via
     tangent space alignment"
     <http://citeseer.ist.psu.edu/viewdoc/summary?doi=10.1.1.4.3693>`_
     Zhang, Z. & Zha, H. Journal of Shanghai Univ. 8:406 (2004)

.. _multidimensional_scaling:

多维比例
===============================

`多维比例（Multidimensional scaling）<http://en.wikipedia.org/wiki/Multidimensional_scaling>`_
(:class:`MDS`)尝试寻找一个低维空间来表示高维数据，同时距离不变。

一般而言，这个技术用来分析数据的相似性。 :class:`MDS` 尝试将数据的相似性表达为距离，譬如相似性，分子相互作用次数，或者国家间贸易指数。

这里存在两类MDS算法：测度的或者非测度的。在scikit-learn中  :class:`MDS` 包含两种。在测度的MDS中要求数据的相似性符合距离测量（如三角不等式）。在非测度情况下，算法将保留距离的顺序，并在嵌套空间中，寻找距离间的单调关系。

.. figure:: ../auto_examples/manifold/images/plot_lle_digits_010.png
   :target: ../auto_examples/manifold/plot_lle_digits.html
   :align: center
   :scale: 50
 
记 :math:`S` 为相似矩阵， :math:`X` 是 :math:`n` 输入样本的坐标，分离 :math:`\hat{d}_{ij}` 是转换相似性的最佳方式。那么目标，压力，则被定义为 :math:`sum_{i < j} d_{ij}(X) - \hat{d}_{ij}(X)` 。


测度MDS
----------

最简单的测度 :class:`MDS` 称之为 *绝对 MDS* ，分离定义为 :math:`\hat{d}_{ij} = S_{ij}` 。 在绝对MDS中 :math:`S_{ij}` 是点 :math:`i` 和 :math:`j` 在空间中的绝对距离。

通常，分离采用 :math:`\hat{d}_{ij} = b S_{ij}` 。

非测度MDS
-------------

非测度 :class:`MDS` 数据的协调性。如果 :math:`S_{ij} < S_{kl}` 那么嵌套中 :math:`d_{ij} < d_{jk}` 。一个简单的方法是采用对  :math:`S_{ij}`  单调回归 :math:`d_{ij}` 来得到分离 :math:`\hat{d}_{ij}` ，使其与 :math:`S_{ij}` 的顺序一致。

一个无意义的解是所有的样本都在原点。规避这个问题的方法是对分离 :math:`\hat{d}_{ij}` 进行重整化。


.. figure:: ../auto_examples/manifold/images/plot_mds_001.png
   :target: ../auto_examples/manifold/plot_mds.html
   :align: center
   :scale: 60
  

.. topic:: 参考：

  * `"Modern Multidimensional Scaling - Theory and Applications"
    <http://www.springer.com/statistics/social+sciences+%26+law/book/978-0-387-25150-9>`_
    Borg, I.; Groenen P. Springer Series in Statistics (1997)

  * `"Nonmetric multidimensional scaling: a numerical method"
    <http://www.springerlink.com/content/tj18655313945114/>`_
    Kruskal, J. Psychometrika, 29 (1964)

  * `"Multidimensional scaling by optimizing goodness of fit to a nonmetric hypothesis"
    <http://www.springerlink.com/content/010q1x323915712x/>`_
    Kruskal, J. Psychometrika, 29, (1964)

.. _t_sne:

t分布随机近邻嵌套 (t-SNE)
===================================================

t-SNE (:class:`TSNE`) 将数据的仿射关系转化为概率。在原空间的近邻关系被表示为高斯概率，而在嵌套空间中被表达为学生t分布。这两者间的Kullback-Leibler (KL)分歧将被通过梯度下降的方法最小化。注意KL分析并不是凸的，即不同的初始化条件会产生不同的局域最小解。因此往往需要尝试不同的初始条件并从中选择最小的KL分歧。

.. figure:: ../auto_examples/manifold/images/plot_lle_digits_013.png
   :target: ../auto_examples/manifold/plot_lle_digits.html
   :align: center
   :scale: 50

t-SNE的主要目的是将高维数据可视化。因此当目标维度为二维或者三维时，其表现最好。

优化KL分歧是一个复杂的过程。有三个参数来调节优化t-SNE：

* 早期夸大系数
* 学习效率
* 步数的最大值

通常步数的最大值比较大，因此不需要额外的调节。那么优化过程主要分两步：早期夸大过程，和最终优化。早起夸大过程中，在原空间中的联合概率会被人为的增加。放大系数越大，会导致不同团间距离的增大。如果这个参数过大，那么KL分歧会增大。通常这个参数不需要调节。另一个关键的参数是学习效率。如果太低，那么优化会停留在局域最小值，而如果太大，那么KL分歧会在优化过程中增大。更多的技巧可以参见 Laurens van der Maaten's FAQ （见参考）。

标准的t-SNE通常比其他流形学习算法慢。优化是困难的，梯度计算的复杂度为 :math:`O[d N^2]` 其中 :math:`d` 是输出维度的数目， :math:`N` 是样本数。

通常 Isomap, LLE 通常在于寻找一个连通的低维流形， t-SNE尝试在结构中找到不同的局域团。这个将样本分团的能力有利于分析数据中潜在的不同类别，如数字的数据。

另外注意，在数字类别的分类的例子中，t-SNE的分类与真实相近，而PCA的二维投影会导致类别区域的重叠。这是一个很好的例子来说明非线性方法的优势（如SVM配合高斯RBF核）。然而t-SNE无法在2D中区分并不意味着在监督学习模型中无法分析。很有可能是二维不足以准确的表达数据的结构。


.. topic:: 参考

  * `"Visualizing High-Dimensional Data Using t-SNE"
    <http://jmlr.org/papers/v9/vandermaaten08a.html>`_
    van der Maaten, L.J.P.; Hinton, G. Journal of Machine Learning Research
    (2008)

  * `"t-Distributed Stochastic Neighbor Embedding"
    <http://homepage.tudelft.nl/19j49/t-SNE.html>`_
    van der Maaten, L.J.P.

应用技巧
=====================

* 确保每一个特征都有相同的尺度。因为流形学习方法基于最近邻的搜索，而算法依赖于其距离。参见 :ref:`StandardScaler <preprocessing_scaler>` 来简便的完成这项任务。

* 每个程序所计算的重构误差可以用来确定最佳的输出维度。 对于 :math:`d` 维流形嵌套到 :math:`D` 维的系数空间，重构误差会随着 ``n_components`` 的增加而降低，直至 ``n_components == d`` 。

* 噪声大的数据可能会错误的连通不同的流形。目前这是一个热门的研究领域。

* 部分输入会导致发散的权重矩阵。例如当有两个以上的取样完全一样的时候，或者数据可以被分割为两个独立的群。在这种情况， ``solver='arpack'`` 会失败。而采用 ``solver='dense'`` 可以得到正确的结果，尽管会变得很慢。此外，人们可以试图找到导致发散的问题，如果是分离的群，那么增加 ``n_neighbors`` 可以缓解。如果是因为有相同的数据点，那么移除它们。

.. seealso::

   :ref:`random_trees_embedding` 同样可以找到特征空间的非线性结构，但是其不进行维度降低。

