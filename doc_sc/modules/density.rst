.. _density_estimation:

==================
密度估计
==================
.. sectionauthor:: Jake Vanderplas <vanderplas@astro.washington.edu>

密度估计介于非监督学习，特征构造和数据建模之间。其中比较著名且有效的方法有混合模型（如高斯混合 :class:`sklearn.mixture.GMM` ）和近邻方法（如核密度统计 :class:`sklearn.neighbors.KernelDensity` ）。高斯混合模型在 :ref:`聚类 <clustering>` 中有详细的介绍。

密度估计的概念比较简单，而且大多数人都熟悉最常用的统计方法：柱状图。m.

密度统计：柱状图
==============================

柱状图是一个简单的数据可视化。其中每个柱代表数据点在其中的数目。下图的座上部分为柱状图的示例：

.. |hist_to_kde| image:: ../auto_examples/neighbors/images/plot_kde_1d_001.png
   :target: ../auto_examples/neighbors/plot_kde_1d.html
   :scale: 80

.. centered:: |hist_to_kde|

柱状图的一个主要问题是区间的选择会有不良的影响。譬如上图的右上部分是对同样数据的重新统计，每一个区间向右移动了。其结果与左边的迥然不同，会导致对数据的不同理解。

直观上，我们可以认为柱状图是一些区块的叠加。通过在合适的网格中叠加，我们可以重绘柱状图。然而如果我们将每个数据作为处在该位置的一个区块，直接叠加全部区块的高度，那么我们可以得到左下的表达。虽然其不像是柱状图，但是由于区块的位置由数据决定，其对数据的表达更准确。

这个例子是一个 *核密度统计* 的示例。我们可以通过光滑的核心函数来平滑结果。譬如右下的图展示了Gaussian和密度统计，其中每个数据贡献一个高斯曲线来叠加。一in次结果是一个光滑的密度统计，而其中函数可以作为一个强大的非参数模型来刻画数据分布。

.. _kernel_density:

核密度统计
=========================
scikit-learn中的核密度统计是采用 :class:`sklearn.neighbors.KernelDensity` 。其中球树或者KD树被用作有效的检索（参见 :ref:`neighbors` ）。尽管以上的例子用到的 1维数据作为简单的例子，核密度统计可以应用到任意维中。但是高维数据会降低计算的效率。

在以下的图中，通过双峰分布产生100个数据点，而核密度分析采用了三个不同的选择：

.. |kde_1d_distribution| image:: ../auto_examples/neighbors/images/plot_kde_1d_003.png
   :target: ../auto_examples/neighbors/plot_kde_1d.html
   :scale: 80

.. centered:: |kde_1d_distribution|

这个可以很直白的展示不同核对结果的影响。scikit-learn的使用方法如下::

   >>> from sklearn.neighbors.kde import KernelDensity
   >>> import numpy as np
   >>> X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
   >>> kde = KernelDensity(kernel='gaussian', bandwidth=0.2).fit(X)
   >>> kde.score_samples(X)
   array([-0.41075698, -0.41075698, -0.41076071, -0.41075698, -0.41075698,
          -0.41076071])

其中我们采用 ``kernel='gaussian'`` 。 数学上讲，核是一个正的函数 :math:`K(x;h)` ，其中控制带宽的参数 :math:`h` 。给定核的形式，在每一个点 :math:`y` 处，密度统计通过若干近邻的点 :math:`x_i; i=1\cdots N` 完成：

.. math::
    \rho_K(y) = \sum_{i=1}^{N} K((y - x_i) / h)

其中带宽的作用是一个平滑参数，控制偏差和方差的大小。一个大的带宽会让曲线很平滑（大的偏差）。一个小的带宽会导致曲线不平滑（大的方差）。

:class:`sklearn.neighbors.KernelDensity`采用如下常见的核：

.. |kde_kernels| image:: ../auto_examples/neighbors/images/plot_kde_1d_002.png
   :target: ../auto_examples/neighbors/plot_kde_1d.html
   :scale: 80

.. centered:: |kde_kernels|

以上核的数学形式为：

* 高斯核 (``kernel = 'gaussian'``)
  
  :math:`K(x; h) \propto \exp(- \frac{x^2}{2h^2} )`

* Tophat核 (``kernel = 'tophat'``)

  :math:`K(x; h) \propto 1` if :math:`x < h`

* Epanechnikov核 (``kernel = 'epanechnikov'``)
  
  :math:`K(x; h) \propto 1 - \frac{x^2}{h^2}`

* 指数核 (``kernel = 'exponential'``)

  :math:`K(x; h) \propto \exp(-x/h)`

* 线性核 (``kernel = 'linear'``)

  :math:`K(x; h) \propto 1 - x/h` if :math:`x < h`

* 余弦核 (``kernel = 'cosine'``)

  :math:`K(x; h) \propto \cos(\frac{\pi x}{2h})` if :math:`x < h`

核密度统计可以采用任何正确的距离矩阵（参见 :class:`sklearn.neighbors.DistanceMetric` ），但结果只有当采用欧几里得空间的时候是正确标准化的。其中比较有用的核是 `Haversine distance <http://en.wikipedia.org/wiki/Haversine_formula>`_ 其测量数据点在球面上距离。这对于可视化地理信息尤为重要。下图展示了两个不同物种在南美洲大陆的分布：

.. |species_kde| image:: ../auto_examples/neighbors/images/plot_species_kde_001.png
   :target: ../auto_examples/neighbors/plot_species_kde.html
   :scale: 80

.. centered:: |species_kde|

核密度统计的另一个应用是学习非参数的概括模型，以此来更有效的提取新样本。以下是一个采用此过程生成新的手写数字的示例。其中高斯核应用于数据的PCA投影：

.. |digits_kde| image:: ../auto_examples/neighbors/images/plot_digits_kde_sampling_001.png
   :target: ../auto_examples/neighbors/plot_digits_kde_sampling.html
   :scale: 80

.. centered:: |digits_kde|

这些“新”数据是原有数据的线性组合，每个的权重是有KDE模型决定。

.. topic:: 示例

  * :ref:`example_neighbors_plot_kde_1d.py`: computation of simple kernel
    density estimates in one dimension.

  * :ref:`example_neighbors_plot_digits_kde_sampling.py`: an example of using
    Kernel Density estimation to learn a generative model of the hand-written
    digits data, and drawing new samples from this model.

  * :ref:`example_neighbors_plot_species_kde.py`: an example of Kernel Density
    estimation using the Haversine distance metric to visualize geospatial data
