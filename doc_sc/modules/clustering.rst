.. _clustering:

==========
聚类
==========

`聚类 <http://en.wikipedia.org/wiki/Cluster_analysis>`__ 是通过 :mod:`sklearn.cluster` 对无分类数据进行学习。

每一个聚类算法都有两个部分：一个包含 ``fit`` 函数的类，用来对训练数据进行分类和一个函数返回一个整数数列来标记不同的类别。对于类，其分类信息可以在 ``labels_`` 属性中找到。

.. currentmodule:: sklearn.cluster

.. topic:: 输入数据

    需要注意的是本模块的算法采用不同类型的输入矩阵。一方面， :class:`MeanShift` 和 :class:`KMeans` 输入矩阵的大小为 [n_samples, n_features]。 这个可以通过模块 :mod:`sklearn.feature_extraction` 获得。另一方面， :class:`AffinityPropagation` 和 :class:`SpectralClustering` 需要大小为[n_samples, n_samples]的相近矩阵。这个可以通过 :mod:`sklearn.metrics.pairwise` 模块获得。换而言之， :class:`MeanShift` 和 :class:`KMeans` 将数据作为向量操作。而 :class:`AffinityPropagation` 和 :class:`SpectralClustering` 可以对任何类别操作，只要其相近测度存在。

聚类算法概述
===============================

.. figure:: ../auto_examples/cluster/images/plot_cluster_comparison_001.png
   :target: ../auto_examples/cluster/plot_cluster_comparison.html
   :align: center
   :scale: 50

   scikit-learn中的聚类算法比较


.. list-table::
   :header-rows: 1
   :widths: 14 15 19 25 20

   * - 方法名称
     - 参数
     - 算法复杂度
     - 应用
     - 几何（测度）

   * - :ref:`K-平均 <k_means>`
     - 类别数目
     - 非常大的 ``n_samples`` ， 正常的 ``n_clusters`` （采用 :ref:`MiniBatch code <mini_batch_kmeans>` ）
     - 一般应用，类别数目相近，分类数目不多，几何平面距离。
     - 样本点间距离

   * - :ref:`仿射传播 <affinity_propagation>`
     - 阻尼，样本偏好
     - 正比于 n_samples
     - 大量分类，类别数目不均匀，非平几何
     - 图距离（如最近邻图）

   * - :ref:`平均偏移 <mean_shift>`
     - 贷款
     - 与 ``n_samples`` 无关
     - 多类群，不均衡类别数目，非平几何
     - 点间距离

   * - :ref:`谱聚类 <spectral_clustering>`
     - 类别数目
     - 适量的 ``n_samples`` ，比较少的 ``n_clusters``
     - 少量类别，类别数目均衡，非平几何
     - 图距离（如最近邻图）

   * - :ref:`Ward 等级聚类 <hierarchical_clustering>`
     - 类别数目
     - 大量数据 ``n_samples`` 和 ``n_clusters``
     - 很多类别，可能的联通限制
     - 点间距离

   * - :ref:`等级聚类 <hierarchical_clustering>`
     - 类别数目，链接类型，距离
     - 大的 ``n_samples`` 和 ``n_clusters``
     - 很多类别，可能的联通限制，非平距离（非欧几何）
     - 任何点对间距

   * - :ref:`DBSCAN <dbscan>`
     - 邻居大小
     - 非常大 ``n_samples`` ，中等的 ``n_clusters``
     - 非平几何，类别数目不均衡
     - 最近邻点距离

   * - :ref:`高斯混合 <mixture>`
     - 参见具体细节
     - 不可扩展
     - 平面几何，方便作密度分析
     - 到中心的Mahalanobis距离

非平几何聚类对于类别分布有着特别的形状时很有效，如非平流形，标准欧几里得距离并不适用。这种情况在上图的最上两行中有所展示。

高斯混合模型是聚类的有效方法，我们在 :ref:`前一章 <mixture>` 中具体解释。K-平均可以视作高斯混合模型的特例：每一个高斯分布的协变矩阵都相同。

.. _k_means:

K-平均
=======

:class:`KMeans` 算法将数据分为n组，每组的方差相同。这个过程通过最小化 `惯性 <inertia>` 也就是组内平方和来实现。这个算法需要给定组分数目。它可以很容易的扩展到大量样本，也被应用到众多领域。

k-平均算法将 :math:`N` 个样本 :math:`X` 分配到 :math:`K` 不相接的类别 :math:`C` 中，每一个类别通过其中样本的平均值 :math:`\mu_j` 来表征。这些平均值通常成为类别的中心（"centroids"）。注意，它们通常并不是X中的数据点，尽管它们处在同一个空间。k-平均算法旨在选择中心来降低惯性：

.. math::    \sum_{i=0}^{n}\min_{\mu_j \in C}(||x_j - \mu_i||^2)

惯性可以被视作一个类别的一致性。但是其有如下缺点：

- 惯性假设类别的数据是凸的，且各向同性。但有时并不如此，因此它对于拉长的类别或者形状不规则的形态表现不佳。

- 惯性不是一个归一化的标准。我们只是知道越低越好，零是最佳。但是在非常高维数据情况，距离会被夸大（所谓维数的诅咒）。因此事先采用维度降低算法，如 `PCA <PCA>` ，可以减轻这个问题，并提高k-平均的计算效率。

K平均也被称为Lloyd算法。简单说，其包含三步：第一步选择初始中心，最基本方法是从样本 :math:`X` 中选择出 :math:`k` 个；这个初始步骤结束之后，算法不断的在后两步中重复，先是将所有样本划分到最近的类别中，第二部是计算每个类中样本的平均值作为新的中心。当旧的和新的中心的的变化小于一定阈值时，算法结束计算。换而言之，其结束时，中心不在明显变化。

.. image:: ../auto_examples/cluster/images/plot_kmeans_digits_001.png
   :target: ../auto_examples/cluster/plot_kmeans_digits.html
   :align: right
   :scale: 35

k-平均算法等价于预期最大化算法当协变矩阵是全等，对角且非常小的特例。

这个算法可以通过 `Voronoi图 <https://en.wikipedia.org/wiki/Voronoi_diagram>` 的概念来理解。首先Voronoi图的点对应于中心。每个Voronoi图块作为一个独立的类别。接着，中心按照图块的平均做更新。算法不断重复知道一个停止条件被满足，通常是当目标函数的相对减少低于一个给定阈值时。在本算法中，却不是这样，其是当中心移动小于某个阈值时。

给定充分时间，k-平均算法总会收敛。但这可能是个局域最优。这个很取决于中心的初始条件。因此，往往需要计算若干次不同初始条件的结果。另一个解决问题的方法是采用k-means++ 初始策略。在scikit-learn通过设置 ``init='kmeans++'`` 参数。其将初始各个中心的位置远离彼此，这意味着一个比随机初始化更好的结果（证明见参考）。

通过 ``n_jobs`` 参数可以进行并行k-平均计算。一个正数标志着处理器的数目（默认是1），而-1表示使用所有的处理器，-2表示少用一个，等等。平行计算在提高速度的同时需要更大的内存（因为多个中心在每个任务中存储一份拷贝）。

.. warning::

    k-平均的并行版本在Mac上存在问题。问题在于numpy使用的 Accelerate Framework. This is expected behavior: Accelerate can be called after a fork but you need to execv the subprocess with the Python binary (which multiprocessing does not do under posix).


K-平均可以进行vector quantization。这是在 :class:`KMeans` 中对对拟合数据应用转换函数。

.. topic:: 示例:

 * :ref:`example_cluster_plot_kmeans_digits.py`: 数字识别

.. topic:: 参考:

 * `"k-means++: The advantages of careful seeding"
   <http://ilpubs.stanford.edu:8090/778/1/2006-13.pdf>`_
   Arthur, David, and Sergei Vassilvitskii,
   *Proceedings of the eighteenth annual ACM-SIAM symposium on Discrete
   algorithms*, Society for Industrial and Applied Mathematics (2007)

.. _mini_batch_kmeans:

小批量K-平均
------------------

:class:`MiniBatchKMeans` 是 :class:`KMeans` 的改进，通过小批量来减少计算时间。其仍然优化同样的目标函数。Mini-batches 是输入训练样本的随机抽取的子集。这个过程可以显著的降低k-平均收敛的计算时间。小批量K-平均算法的结果仅比标准算法略微差些。

与直接的k-平均算法一样，这个算法在两个主要步骤中反复。第一步随机抽取 :math:`b`  个样本作为mini-batch，并分配到最近的中心。第二步，更新中心的位置。与k-平均不同之处在于更新是基于取样的。对于在mini-batch中的每一个取样，中心是基于这个取样，和此前结果的平均。因此中心变动的大小会随时间降低。这些步骤会重复直至收敛或者达到特定的步数。

:class:`MiniBatchKMeans` 比 :class:`KMeans` 收敛更快，但是精确度略低。但在实际应用中，这点不同可以被完全忽略（参见示例和参考）。

.. figure:: ../auto_examples/cluster/images/plot_mini_batch_kmeans_001.png
   :target: ../auto_examples/cluster/plot_mini_batch_kmeans.html
   :align: center
   :scale: 100


.. topic:: 示例:

 * :ref:`example_cluster_plot_mini_batch_kmeans.py`: Comparison of KMeans and
   MiniBatchKMeans

 * :ref:`example_text_document_clustering.py`: Document clustering using sparse
   MiniBatchKMeans

 * :ref:`example_cluster_plot_dict_face_patches.py`


.. topic:: 文献:

 * `"Web Scale K-Means clustering"
   <http://www.eecs.tufts.edu/~dsculley/papers/fastkmeans.pdf>`_
   D. Sculley, *Proceedings of the 19th international conference on World
   wide web* (2010)

.. _affinity_propagation:

仿射传播
====================

:class:`AffinityPropagation` 通过在样本间传递信息来达到分类的收敛。一个数据最终是靠少数的范例进行分类的，而范例是那些最具代表性的样本。样本间的信息表征一个样本更适合哪一个范例。这个更新的过程重复进行直至收敛，进而确定分类。

.. figure:: ../auto_examples/cluster/images/plot_affinity_propagation_001.png
   :target: ../auto_examples/cluster/plot_affinity_propagation.html
   :align: center
   :scale: 50


仿射传播的意义在于其从样本数据中选择范例。因此有两个重要的参数需要调节，一个是范例的数目，另一个是 *阻尼系数* 。

仿射传播的缺点在于其复杂度。这个算法的计算复杂度是 :math:`O(N^2 T)` ，其中 :math:`N` 是样本的数目，而 :math:`T` 是重复的步数。而且其空间复杂度是 
:math:`O(N^2)` 如果矩阵是致密的。当然在稀疏矩阵的情况下问题可以得到缓解。这些问题使得仿射传播只适合中小型的数据。

.. topic:: 示例:

 * :ref:`example_cluster_plot_affinity_propagation.py`: Affinity
   Propagation on a synthetic 2D datasets with 3 classes.

 * :ref:`example_applications_plot_stock_market.py` Affinity Propagation on
   Financial time series to find groups of companies

**算法**

数据点间相互传递的信息分为两类。第一个是“责任” :math:`r(i, k)` ，一个样本 :math:`k` 应该成为样本 :math:`i` 示例的累积证据。第二个是“容量” :math:`a(i, k)` ，一个样本 :math:`i` 可以选择样本 :math:`k` 作为其范例的累积证据。在这基础上，一个样本被选为示例需要满足以下两点： (1) 和其他样本相似 (2) 被其他样本选作示例。


更正是的说，样本 :math:`k` 作为样本 :math:`i` 范例的责任是：

.. math::

    r(i, k) \leftarrow s(i, k) - max [ a(i, \acute{k}) + s(i, \acute{k}) \forall \acute{k} \neq k ]

其中 :math:`s(i, k)` 是样本 :math:`i` 和 :math:`k` 间的相似度。样本 :math:`k` 成为样本 :math:`i` 的容量是：

.. math::

    a(i, k) \leftarrow min [0, r(k, k) + \sum_{\acute{i}~s.t.~\acute{i} \notin \{i, k\}}{r(\acute{i}, k)}]

起始时，所有的 :math:`r` 和 :math:`a` 都是0。之后不断重复知道收敛。

.. _mean_shift:

平均偏移
==========
:class:`MeanShift` 旨在发现样本分布中的结点。其是一个基于中心的算法：更新属于中心的样本，在基于样本更新中心的位置。在后期需要过滤样本来去除重复的样本。

给定在第 :math:`t` 步的一个候选中心 :math:`x_i` ，其根据如下公式进行更新：

.. math::

    x_i^{t+1} = x_i^t + m(x_i^t)

其中 :math:`m` 是每一个中心的平均偏移向量，来最大的增加点的密度。其计算方法如下：

.. math::

    m(x_i) = \frac{\sum_{x_j \in N(x_i)}K(x_j - x_i)x_j}{\sum_{x_j \in N(x_i)}K(x_j - x_i)}

其中 :math:`N(x_i)` 是距样本 :math:`x_i` 一定范围内的邻居数目。这个过程等效于将中心更新为近邻的平均。

这个算法自动设定类别的数目，但需要一个参数 ``bandwidth`` 来设定搜寻的范围。这个参数可以人为设定，也可以通过函数 ``estimate_bandwidth`` 进行估计（当没有设定时会自动计算）。

这个算法的扩展能力不大，因为其需要寻找最近邻。这个算法一定会收敛。但算法一般会在中心变化不大的时候自动停止。

一个新样本的分类是寻找距离最近的中心。

.. figure:: ../auto_examples/cluster/images/plot_mean_shift_001.png
   :target: ../auto_examples/cluster/plot_mean_shift.html
   :align: center
   :scale: 50


.. topic:: 示例:

 * :ref:`example_cluster_plot_mean_shift.py`: Mean Shift clustering
   on a synthetic 2D datasets with 3 classes.

.. topic:: 参考:

 * `"Mean shift: A robust approach toward feature space analysis."
   <http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.76.8968&rep=rep1&type=pdf>`_
   D. Comaniciu, & P. Meer *IEEE Transactions on Pattern Analysis and Machine Intelligence* (2002)


.. _spectral_clustering:

谱聚类
===================

:class:`SpectralClustering` 首先将样本间的仿射矩阵嵌入到低维，再对其进行k-平均分类。当仿射矩阵比较稀疏且有安装 `pyamg <http://pyamg.org/>`_ 时，计算很有效率。谱聚类需要给定类别的数目。当类别数目比较小的时候比较有效，但是不建议针对很多分类的情况。

对于两个类别，其旨在相似图上解决 `normalised cuts <http://www.cs.berkeley.edu/~malik/papers/SM-ncut.pdf>`_ 问题的凸松弛：将图分为两部分使得在被切断的边的权重相对于图中的权重最小。这个方法在处理图像问题尤为有效：图的顶点是像素点，而边是图像梯度的函数。

.. |noisy_img| image:: ../auto_examples/cluster/images/plot_segmentation_toy_001.png
    :target: ../auto_examples/cluster/plot_segmentation_toy.html
    :scale: 50

.. |segmented_img| image:: ../auto_examples/cluster/images/plot_segmentation_toy_002.png
    :target: ../auto_examples/cluster/plot_segmentation_toy.html
    :scale: 50

.. centered:: |noisy_img| |segmented_img|

.. warning:: 转换距离为相似度

    注意，如果相似矩阵的数值分布不佳，例如有负值，或者是个距离矩阵而不是相似性，那么谱问题可能会是发散，不可解的。在这些时候，往往需要将矩阵的数据进行转换。例如，对于有负值的矩阵，通常采用一个热核::

        similarity = np.exp(-beta * distance / distance.std())

    对于其应用，参见示例

.. topic:: 示例：

 * :ref:`example_cluster_plot_segmentation_toy.py`: Segmenting objects
   from a noisy background using spectral clustering.

 * :ref:`example_cluster_plot_lena_segmentation.py`: Spectral clustering
   to split the image of lena in regions.

.. |lena_kmeans| image:: ../auto_examples/cluster/images/plot_lena_segmentation_001.png
    :target: ../auto_examples/cluster/plot_lena_segmentation.html
    :scale: 65

.. |lena_discretize| image:: ../auto_examples/cluster/images/plot_lena_segmentation_002.png
    :target: ../auto_examples/cluster/plot_lena_segmentation.html
    :scale: 65

不同标签分配策略
---------------------------------------

不同的标签分配策略可以通过参数  :class:`SpectralClustering` 中 ``assign_labels`` 的调节。 ``"kmeans"`` 策略可以匹配数据的细节，但是可能会不稳定。尤其，当不控制 ``random_state`` ，其每次运行的结果可能是不同的，由于以来一个随机的初始条件。另一方面， ``"discretize"`` 策略是完全可重复的，但是其往往导致大块区域，或者平直的几何结构。

=====================================  =====================================
 ``assign_labels="kmeans"``              ``assign_labels="discretize"``
=====================================  =====================================
|lena_kmeans|                          |lena_discretize|
=====================================  =====================================


.. topic:: 参考:

 * `"A Tutorial on Spectral Clustering"
   <http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.165.9323>`_
   Ulrike von Luxburg, 2007

 * `"Normalized cuts and image segmentation"
   <http://citeseer.ist.psu.edu/viewdoc/summary?doi=10.1.1.160.2324>`_
   Jianbo Shi, Jitendra Malik, 2000

 * `"A Random Walks View of Spectral Segmentation"
   <http://citeseer.ist.psu.edu/viewdoc/summary?doi=10.1.1.33.1501>`_
   Marina Meila, Jianbo Shi, 2001

 * `"On Spectral Clustering: Analysis and an algorithm"
   <http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.19.8100>`_
   Andrew Y. Ng, Michael I. Jordan, Yair Weiss, 2001


.. _hierarchical_clustering:

等级聚类
=======================

等级聚类（Hierarchical clustering）是一类聚类算法，其通过不断的合并和分割来构建嵌套的类别。这个类别的等级结构可以表示为树结构（或者dendrogram）。树的树根是一个包含所有样本的群，而叶节点是包含一个取样的类别。更多信息参考 `Wikipedia page <http://en.wikipedia.org/wiki/Hierarchical_clustering>`_ 。

:class:`AgglomerativeClustering` 通过自下而上的方式构建等级聚类：每一个取样首先是自己独立的类别，然后类别不断的合并。合并的策略取决于以下的链接条件：

- **Ward** 最小化所有类别内的方差和。这是一个最小化方差的过程，因此和k-平均方法类似，只是通过凝聚层次聚类的方式进行。
- **Maximum** 或者 **complete linkage** 最小化两个样本分类间的最大距离。
- **Average linkage** 最小化两个分类间数据的平均距离。

当采用连接矩阵时， :class:`AgglomerativeClustering`  可以用来处理大量数据。但是当不存在链接的限制时，计算复杂度很高：由于需要考虑所有的合并情况。

.. topic:: :class:`FeatureAgglomeration`

:class:`FeatureAgglomeration` 通过等级聚类方式将相近的特征合并，从而减少特征的数目。这是一个降低数据维度的工具，参见 :ref:`data_reduction` 。


不同的链接类型
-----------------------------------------------------------

:class:`AgglomerativeClustering` 支持三种连接的策略： Ward， average，和 complete。

.. image:: ../auto_examples/cluster/images/plot_digits_linkage_001.png
    :target: ../auto_examples/cluster/plot_digits_linkage.html
    :scale: 43

.. image:: ../auto_examples/cluster/images/plot_digits_linkage_002.png
    :target: ../auto_examples/cluster/plot_digits_linkage.html
    :scale: 43

.. image:: ../auto_examples/cluster/images/plot_digits_linkage_003.png
    :target: ../auto_examples/cluster/plot_digits_linkage.html
    :scale: 43

等级聚类会导致类别的大小不同，大的会变得更大。 在这个意义上，complete linkage是最差的策略，而Ward的大小相近。然而在Ward中仿射（或者用距离分类）的仿是是固定的，因此对于非欧集合，平均连接策略是个很好的替代。

.. topic:: 示例：

 * :ref:`example_cluster_plot_digits_linkage.py`: exploration of the
   different linkage strategies in a real dataset.


添加连接限制
-------------------------------

:class:`AgglomerativeClustering` 的一个有用的性质是可以添加连接限制（只有近邻的类别可以被合并）。
这是通过一个连接矩阵完成的。例如在下面的swiss-roll示例中，连接矩限制将连个不相邻的数据合并到一起。这样可以避免将不同层的数据合并到一起。

.. |unstructured| image:: ../auto_examples/cluster/images/plot_ward_structured_vs_unstructured_001.png
        :target: ../auto_examples/cluster/plot_ward_structured_vs_unstructured.html
        :scale: 49

.. |structured| image:: ../auto_examples/cluster/images/plot_ward_structured_vs_unstructured_002.png
        :target: ../auto_examples/cluster/plot_ward_structured_vs_unstructured.html
        :scale: 49

.. centered:: |unstructured| |structured|

这些限制不但可以限定局域的结构，还可以提高计算速度，尤其是当样本数目很大的时候。

这个连接的限制是通过连接矩阵完成的：一个 `scipy` 系数矩阵，仅在可连通的位置有数值。这个矩阵可以通过先验信息：例如你可以仅合并那些有互相连接的网页。其也可以从数据中学习。例如通过函数 :func:`sklearn.neighbors.kneighbors_graph` 来限制只和最邻近的取样连接（参见 :ref:`this example <example_cluster_plot_agglomerative_clustering.py>` ）或者使用函数 :func:`sklearn.feature_extraction.image.grid_to_graph` 来允许合并图像中近邻的像素（参见 :ref:`Lena <example_cluster_plot_lena_ward_segmentation.py>` ）。

.. topic:: 示例：

 * :ref:`example_cluster_plot_lena_ward_segmentation.py`: Ward clustering
   to split the image of lena in regions.

 * :ref:`example_cluster_plot_ward_structured_vs_unstructured.py`: Example of
   Ward algorithm on a swiss-roll, comparison of structured approaches
   versus unstructured approaches.

 * :ref:`example_cluster_plot_feature_agglomeration_vs_univariate_selection.py`:
   Example of dimensionality reduction with feature agglomeration based on
   Ward hierarchical clustering.

 * :ref:`example_cluster_plot_agglomerative_clustering.py`

.. warning:: **average和complete连接中的连接限制**

    连接限制和complete或average可能回进一步强化大的变得更大的效果。尤其当采用 :func:`sklearn.neighbors.kneighbors_graph` 的时候。在分类数目很小的极限下，他们会给出几个涵盖所有的类别和几个几乎为空的类别（参见 :ref:`example_cluster_plot_agglomerative_clustering.py` ）。

.. image:: ../auto_examples/cluster/images/plot_agglomerative_clustering_001.png
    :target: ../auto_examples/cluster/plot_agglomerative_clustering.html
    :scale: 38

.. image:: ../auto_examples/cluster/images/plot_agglomerative_clustering_002.png
    :target: ../auto_examples/cluster/plot_agglomerative_clustering.html
    :scale: 38

.. image:: ../auto_examples/cluster/images/plot_agglomerative_clustering_003.png
    :target: ../auto_examples/cluster/plot_agglomerative_clustering.html
    :scale: 38

.. image:: ../auto_examples/cluster/images/plot_agglomerative_clustering_004.png
    :target: ../auto_examples/cluster/plot_agglomerative_clustering.html
    :scale: 38


测度的选择
-------------------

Average和complete 连接可以与不同的距离定义相结合（或仿射）。如Euclidean距离(*l2*)， Manhattan距离（或城市街区， *l1* ），余弦距离，或者计算好的仿射矩阵。

* *l1* 距离通常适用于稀疏的特征，或者噪声：这是对应很多特征是零，如文本挖掘中低频词。

* *cosine* （余弦）距离的意义在于在全局的变量标量变换中，其大小不变。

测度的选择在于最大化类别间的距离，最小化类别内的距离。

.. image:: ../auto_examples/cluster/images/plot_agglomerative_clustering_metrics_005.png
    :target: ../auto_examples/cluster/plot_agglomerative_clustering_metrics.html
    :scale: 32

.. image:: ../auto_examples/cluster/images/plot_agglomerative_clustering_metrics_006.png
    :target: ../auto_examples/cluster/plot_agglomerative_clustering_metrics.html
    :scale: 32

.. image:: ../auto_examples/cluster/images/plot_agglomerative_clustering_metrics_007.png
    :target: ../auto_examples/cluster/plot_agglomerative_clustering_metrics.html
    :scale: 32

.. topic:: 示例

 * :ref:`example_cluster_plot_agglomerative_clustering_metrics.py`


.. _dbscan:

DBSCAN
======

:class:`DBSCAN` 算法视类别为一些高密度的区域，其间由低密度的区域分割。基于这个宏观的角度，由DBSCAN找到的类别可以有任意的几何结构，对比K-平均方法假设类别都是凸出的。DBSCAN的精髓是 *核心样本* 的概念，它们是在高密度区的取样。一个类别就是一个核心样本集合（每一个与另一个靠近），与一个非核心样本集合（靠近核心样本）。本算法有两个参数 ``min_samples`` 和 ``eps`` ， 来定义 *密度* 。高的 ``min_samples`` 或者低的 ``eps`` 对应一个类别需要高的密度来形成。

更正式的表述是，我们定义一个核心取样为在样本中的一个取样，其周边 ``eps`` 范围内存在 ``min_samples`` 个样本，它们被视作核心样本的 *邻居* 。这告诉我们核心样本是在向量空间中的高密度区域。一个类别是这样互为邻居的核心取样群。一个非核心取样属于其邻居核心取样的类别，但是其自身不是核心样本。因此这些样本在类别的外缘。

任何核心样本都是一个类别的一部分。而任何类别均需要存在至少 ``min_samples`` 个样本，否则不存在核心样本。对于任意一个非核心样本，如果其距离任何核心样本的距离都远于 ``eps`` ，本算法视其为异常值。

在下图中，颜色代表类别的分属。其中大圈代表算法找到的核心取样，小圈代表非核心取样。而黑点被视作异常值。

.. |dbscan_results| image:: ../auto_examples/cluster/images/plot_dbscan_001.png
        :target: ../auto_examples/cluster/plot_dbscan.html
        :scale: 50

.. centered:: |dbscan_results|

.. topic:: 示例：

    * :ref:`example_cluster_plot_dbscan.py`

.. topic:: 算法实现

    这个算法是非确定性的，但是核心样本总会属于同一个类别（尽管标签可能会不同）。这个非确定性来自于分类那些非核心样本。一个非核心样本可以距离多个类别小于 ``eps``  基于三角不等式，这两个类别的距离是大于 ``eps`` 的。而非核心取样将被分类到首先生成的分类中，这是取决于数据的顺序。除了数据的顺序，算法是确定的，因此结果是稳定的。

   目前的实现采用的是球树和kd树来确定临近的数据，而不需要计算全部的距离矩阵。对于用户定义的距离，可以参见 :class:`NearestNeighbors` 的使用方法。

.. topic:: 参考：

 * "A Density-Based Algorithm for Discovering Clusters in Large Spatial Databases
   with Noise"
   Ester, M., H. P. Kriegel, J. Sander, and X. Xu,
   In Proceedings of the 2nd International Conference on Knowledge Discovery
   and Data Mining, Portland, OR, AAAI Press, pp. 226–231. 1996

.. _clustering_evaluation:

聚类评测
=================================

评价聚类的表现并不是简单的依赖于错误的数目，或者如监督分类算法的精度和召回。尤其是任何评测的方法都不应该以绝对的分类值最为标准。而是应以其分类是否更好的将数据分组，使其同一类的样本更近似。

.. currentmodule:: sklearn.metrics

调整后的芮氏指标 
-------------------

表达和用法
~~~~~~~~~~~~~~~~~~~~~~

给定已知正确的分类 ``labels_true`` 和聚类算法给出的分类 ``labels_pred`` ， **调整后的芮氏指数 （Adjusted Rand Index ARI）** 是描述两个分类 **相似性** 的函数（忽略置换，并重新归一）::

  >>> from sklearn import metrics
  >>> labels_true = [0, 0, 0, 1, 1, 1]
  >>> labels_pred = [0, 0, 1, 1, 2, 2]

  >>> metrics.adjusted_rand_score(labels_true, labels_pred)  # doctest: +ELLIPSIS
  0.24...

如果我们置换0和1，我们仍然会得到同样的结果::

  >>> labels_pred = [1, 1, 0, 0, 3, 3]
  >>> metrics.adjusted_rand_score(labels_true, labels_pred)  # doctest: +ELLIPSIS
  0.24...

此外，函数 :func:`adjusted_rand_score` 是 **对称的**： 交换两个输入变量，结果仍然一样。因此可以作为一个 **共识测量**::

  >>> metrics.adjusted_rand_score(labels_pred, labels_true)  # doctest: +ELLIPSIS
  0.24...

一个完美的分类的得分是1::

  >>> labels_pred = labels_true[:]
  >>> metrics.adjusted_rand_score(labels_true, labels_pred)
  1.0

一个差的分类（随机分类）的得分接近于0::

  >>> labels_true = [0, 1, 2, 0, 3, 4, 5, 1]
  >>> labels_pred = [1, 1, 0, 0, 2, 2, 2, 2]
  >>> metrics.adjusted_rand_score(labels_true, labels_pred)  # doctest: +ELLIPSIS
  -0.12...


优势
~~~~~~~~~~

- **随机（均匀）分类的ARI得分接近于0** 对于任意分类 ``n_clusters`` 和样本 ``n_samples`` 。（对于直接的芮氏指标，或者V指标，并不如此）

- **取值范围 [-1, 1]** 负分数是不好的结果，相似的分类的ARI得分接近于1。

- **无需假设分类的结构** ：可以用来与比较k平均算法（圆形）与谱聚类算法（任意形状）。


劣势
~~~~~~~~~

-**ARI需要知道真实分类** 这点在实际中常常不满足，除非采用人工分类（监督的学习）

  然而ARI对于在纯粹非监督的环境下，可以用来做模型选择（TODO）


.. topic:: 示例:

 * :ref:`example_cluster_plot_adjusted_for_chance_measures.py`: Analysis of
   the impact of the dataset size on the value of clustering measures
   for random assignments.


数学基础
~~~~~~~~~~~~~~~~~~~~~~~~

如果C是真实的分类，而K是预测的分类，定义 :math:`a` 和 :math:`b` 如下：

- :math:`a` 在C与K中处于同一分类的数目

- :math:`b` 在C与K中分别处于不同分类的数目

直接的芮氏指标是：

.. math::    \text{RI} = \frac{a + b}{C_2^{n_{samples}}}

其中  :math:`C_2^{n_{samples}}` 数据中所有可能组合的数目。

然而这个芮氏制表并不保证一个随机的分类会得到一个接近0的值（尤其当类别的数目和样本数目接近的时候）。

为了抵消这个影响，我们可以削弱预期的随机分类的影响 :math:`E[\text{RI}]` 来定义修正后的芮氏指标如下：

.. math:: \text{ARI} = \frac{\text{RI} - E[\text{RI}]}{\max(\text{RI}) - E[\text{RI}]}

.. topic:: 参考

 * `Comparing Partitions
   <http://www.springerlink.com/content/x64124718341j1j0/>`_
   L. Hubert and P. Arabie, Journal of Classification 1985

 * `Wikipedia entry for the adjusted Rand index
   <http://en.wikipedia.org/wiki/Rand_index#Adjusted_Rand_index>`_


基于相互信息的分数
-------------------------------

表达和用法
~~~~~~~~~~~~~~~~~~~~~~

给定真实的分类 ``labels_true`` 和聚类算法给出的分类 ``labels_pred`` ， **相互信息** 是测量两个分类 **一致性** 的函数（忽略置换，排序）。两个不同的版本如下： **标准化的相互信息（Normalized Mutual Information NMI）**  和 **调整后的相互信息（Adjusted Mutual Information AMI）**. NMI 在文献中经常被引用，而AMI是最近被提出，并 **相对于机会标准化**::

  >>> from sklearn import metrics
  >>> labels_true = [0, 0, 0, 1, 1, 1]
  >>> labels_pred = [0, 0, 1, 1, 2, 2]

  >>> metrics.adjusted_mutual_info_score(labels_true, labels_pred)  # doctest: +ELLIPSIS
  0.22504...

当交换0和1时，结果不变。

  >>> labels_pred = [1, 1, 0, 0, 3, 3]
  >>> metrics.adjusted_mutual_info_score(labels_true, labels_pred)  # doctest: +ELLIPSIS
  0.22504...

:func:`mutual_info_score` ， :func:`adjusted_mutual_info_score` 和 :func:`normalized_mutual_info_score` 都是对称的，所以交换输入变量不改变结果。因此他们可以作为 **共识测量** 。

  >>> metrics.adjusted_mutual_info_score(labels_pred, labels_true)  # doctest: +ELLIPSIS
  0.22504...

一个完美的分类的得分是0::

  >>> labels_pred = labels_true[:]
  >>> metrics.adjusted_mutual_info_score(labels_true, labels_pred)
  1.0

  >>> metrics.normalized_mutual_info_score(labels_true, labels_pred)
  1.0

然而对于 ``mutual_info_score`` 结果并不直接::

  >>> metrics.mutual_info_score(labels_true, labels_pred)  # doctest: +ELLIPSIS
  0.69...

一个差的分类（随机分类）的得分为负值:

  >>> labels_true = [0, 1, 2, 0, 3, 4, 5, 1]
  >>> labels_pred = [1, 1, 0, 0, 2, 2, 2, 2]
  >>> metrics.adjusted_mutual_info_score(labels_true, labels_pred)  # doctest: +ELLIPSIS
  -0.10526...


优势
~~~~~~~~~~

- **随机（均匀）分类的AMI得分接近于0** 对于任意分类 n_clusters 和样本 n_samples 。（对于直接的相互信息，或者V指标，并不如此）

- **取值范围 [-1, 1]**:  接近0的值表明分类的结果相互不相关，而接近1的结果表明两个分类结果很一致。一个绝对的0值表示完全独立的分类结果，而1表明两个分类完全一致。

- **无需假设分类的结构** ：可以用来与比较k平均算法（圆形）与谱聚类算法（任意形状）。



劣势
~~~~~~~~~

-**基于相互信息的指标需要真实的分类** 这点在实际中常常不满足，除非采用人工分类（监督的学习）

  然而基于相互信息的对于在纯粹非监督的环境下，可以用来做模型选择

- NMI 和 MI 都没有针对机会做出调整。


.. topic:: 示例：

 * :ref:`example_cluster_plot_adjusted_for_chance_measures.py`: Analysis of
   the impact of the dataset size on the value of clustering measures
   for random assignments. This example also includes the Adjusted Rand
   Index.


数学基础
~~~~~~~~~~~~~~~~~~~~~~~~

假设两个分类结果（对同样的N个样本）是 :math:`U` 和 :math:`V` 。它们的熵被定义为分割的不确定性：

.. math:: H(U) = \sum_{i=1}^{|U|}P(i)\log(P(i))

其中 :math:`P(i) = |U_i| / N` 是一个样本从 :math:`U` 中被随机分类到  :math:`U_i` 类中的概率。类似的有：

.. math:: H(V) = \sum_{j=1}^{|V|}P'(j)\log(P'(j))

其中 :math:`P'(j) = |V_j| / N` 。 :math:`U` 与 :math:`V` 间的相互信息（MI）定义如下：

.. math:: \text{MI}(U, V) = \sum_{i=1}^{|U|}\sum_{j=1}^{|V|}P(i, j)\log\left(\frac{P(i,j)}{P(i)P'(j)}\right)

其中 :math:`P(i, j) = |U_i \cap V_j| / N` 是一个取样随机落入两个分类 :math:`U_i` 和 :math:`V_j` 的概率。

标准化的相互信息的定义为：

.. math:: \text{NMI}(U, V) = \frac{\text{MI}(U, V)}{\sqrt{H(U)H(V)}}

以上的相互信息和标准化的相互信息都没有针对机会进行调节。因此当类别的数目增多时，指标会增加。

而相互信息的预期可以有以下计算得来（参见Vinh, Epps, and Bailey, (2009)），其中 :math:`a_i = |U_i|` （ :math:`U_i` 中样本数）和 :math:`b_j = |V_j|` （ :math:`V_j` 中的样本数）。


.. math:: E[\text{MI}(U,V)]=\sum_{i=1}^|U| \sum_{j=1}^|V| \sum_{n_{ij}=(a_i+b_j-N)^+
   }^{\min(a_i, b_j)} \frac{n_{ij}}{N}\log \left( \frac{ N.n_{ij}}{a_i b_j}\right)
   \frac{a_i!b_j!(N-a_i)!(N-b_j)!}{N!n_{ij}!(a_i-n_{ij})!(b_j-n_{ij})!
   (N-a_i-b_j+n_{ij})!}

通过预期值，我们可以类比ARI得到调节后的相互信息如下：

.. math:: 

   \text{AMI} = \frac{\text{MI} - E[\text{MI}]}{\max(H(U), H(V)) - E[\text{MI}]}

.. topic:: 参考

 * Strehl, Alexander, and Joydeep Ghosh (2002). "Cluster ensembles – a
   knowledge reuse framework for combining multiple partitions". Journal of
   Machine Learning Research 3: 583–617. doi:10.1162/153244303321897735

 * Vinh, Epps, and Bailey, (2009). "Information theoretic measures
   for clusterings comparison". Proceedings of the 26th Annual International
   Conference on Machine Learning - ICML '09.
   doi:10.1145/1553374.1553511. ISBN 9781605585161.

 * Vinh, Epps, and Bailey, (2010). Information Theoretic Measures for
   Clusterings Comparison: Variants, Properties, Normalization and
   Correction for Chance}, JMLR
   http://jmlr.csail.mit.edu/papers/volume11/vinh10a/vinh10a.pdf

 * `Wikipedia entry for the (normalized) Mutual Information
   <http://en.wikipedia.org/wiki/Mutual_Information>`_

 * `Wikipedia entry for the Adjusted Mutual Information
   <http://en.wikipedia.org/wiki/Adjusted_Mutual_Information>`_

一致性，完整性和V指标
---------------------------------------

表达和用法
~~~~~~~~~~~~~~~~~~~~~~

在知道准确分类的情况下，我们可以通过条件熵分析来定义直观的指标。

其中Rosenberg and Hirschberg (2007) 对任意聚类定义了如下两个期待的目标：

- **一致性** ：一个聚类中的取样来自于同一个类。

- **完整性** ：来自同一个类的取样被分到同一个聚类中。

我们可以将这些概念转换为分数 :func:`homogeneity_score` 和 :func:`completeness_score` 。这两个分数的取值都在0和1之间，而且越高越好。

  >>> from sklearn import metrics
  >>> labels_true = [0, 0, 0, 1, 1, 1]
  >>> labels_pred = [0, 0, 1, 1, 2, 2]

  >>> metrics.homogeneity_score(labels_true, labels_pred)  # doctest: +ELLIPSIS
  0.66...

  >>> metrics.completeness_score(labels_true, labels_pred) # doctest: +ELLIPSIS
  0.42...

调和平均被称为 **V指标** ，并通过函数 :func:`v_measure_score` 计算::

  >>> metrics.v_measure_score(labels_true, labels_pred)    # doctest: +ELLIPSIS
  0.51...

这个V指标等价上述的NMI除以所有分类的熵  [B2011]_ 。

一致性，完整性和V指标可以通过 :func:`homogeneity_completeness_v_measure` 计算::

  >>> metrics.homogeneity_completeness_v_measure(labels_true, labels_pred)
  ...                                                      # doctest: +ELLIPSIS
  (0.66..., 0.42..., 0.51...)

以下的聚类更好，因为其一致性更好，而完整性略有不足::

  >>> labels_pred = [0, 0, 0, 1, 2, 2]
  >>> metrics.homogeneity_completeness_v_measure(labels_true, labels_pred)
  ...                                                      # doctest: +ELLIPSIS
  (1.0, 0.68..., 0.81...)

.. note::

  :func:`v_measure_score` 是 **对称的** ：其可以用来测量两个类别的一致性。

  但对于 :func:`completeness_score` 和 :func:`homogeneity_score` 并非如此，其关系如下::

    homogeneity_score(a, b) == completeness_score(b, a)


优势
~~~~~~~~~~

- **取值范围[0,1]** ：0是最坏可能性，1是最好的情况

- 解释直观：V指标低的聚类可以通过具体的一致性和完整性来了解有问题的分类。

- **无需假设分类的结构**  ：可以用来与比较k平均算法（圆形）与谱聚类算法（任意形状）。


劣势
~~~~~~~~~

- 这个指标并不没有考虑 **随机分类** 的情况。这意味着结果取决于样本的数目，类别的数目。一个完全随机的分类并不会得到一致的分数。尤其是 **当类别很多时，随机分类的指标不为0** 。

  这个问题再样本上千而类别小于10的时候可以忽略。 **对于比较小的样本，或者比较多的类别，采用修正的指标，如ARI会更可靠。**

.. figure:: ../auto_examples/cluster/images/plot_adjusted_for_chance_measures_001.png
   :target: ../auto_examples/cluster/plot_adjusted_for_chance_measures.html
   :align: center
   :scale: 100

- V指标需要真实的分类 这点在实际中常常不满足，除非采用人工分类（监督的学习）


.. topic:: 示例：

 * :ref:`example_cluster_plot_adjusted_for_chance_measures.py`: Analysis of
   the impact of the dataset size on the value of clustering measures
   for random assignments.


数学基础
~~~~~~~~~~~~~~~~~~~~~~~~

一致性和完整性被定义为：

.. math:: h = 1 - \frac{H(C|K)}{H(C)}

.. math:: c = 1 - \frac{H(K|C)}{H(K)}

其中 :math:`H(C|K)` 是 **类别的条件熵** 其定义如下

.. math:: H(C|K) = - \sum_{c=1}^{|C|} \sum_{k=1}^{|K|} \frac{n_{c,k}}{n}
          \cdot \log\left(\frac{n_{c,k}}{n_k}\right)

其中 :math:`H(C)` 是类别的熵，定义为：

.. math:: H(C) = - \sum_{c=1}^{|C|} \frac{n_c}{n} \cdot \log\left(\frac{n_c}{n}\right)

其中 :math:`n` 是样本总数， :math:`n_c` 和 :math:`n_k` 是样本分别属于类别 :math:`c` 和 :math:`k` 的数目，而 :math:`n_{c,k}` 是属于类别 :math:`c` 的取样被 识别为聚类 :math:`k` 的数目。

**类别的条件熵** :math:`H(K|C)` 以及 **类别的熵** :math:`H(K)` 的定义式对称的。

Rosenberg and Hirschberg 进一步定义 **一致性和完整性的调和平均** 为 **V指标** ：

.. math:: v = 2 \cdot \frac{h \cdot c}{h + c}

.. topic:: 参考

 .. [RH2007] `V-Measure: A conditional entropy-based external cluster evaluation
   measure <http://acl.ldc.upenn.edu/D/D07/D07-1043.pdf>`_
   Andrew Rosenberg and Julia Hirschberg, 2007

 .. [B2011] `Identication and Characterization of Events in Social Media
   <http://www.cs.columbia.edu/~hila/hila-thesis-distributed.pdf>`_, Hila
   Becker, PhD Thesis.

.. _silhouette_coefficient:

Silhouette 系数
----------------------

表达和用法
~~~~~~~~~~~~~~~~~~~~~~

如果不知道真实的分类，那么就只能靠模型本身的特征。Silhouette系数（ :func:`sklearn.metrics.silhouette_score` ）是一个这样的评价系统。一个高的分数对应着更好的分类。对于每一个样本，其包含两个分数：

- **a**: 一个取样到该类别中其他取样的平均距离

- **b**: 一个取样到与该类别 *最近的类别* 中取样的平均距离

Silhoeutte系数 *s*  则是：

.. math:: s = \frac{b - a}{\max(a, b)}

对于一个样本，Silhouette是每一个取样的参数的平均。


  >>> from sklearn import metrics
  >>> from sklearn.metrics import pairwise_distances
  >>> from sklearn import datasets
  >>> dataset = datasets.load_iris()
  >>> X = dataset.data
  >>> y = dataset.target

一般而言，以下是应用到聚类分析中

  >>> import numpy as np
  >>> from sklearn.cluster import KMeans
  >>> kmeans_model = KMeans(n_clusters=3, random_state=1).fit(X)
  >>> labels = kmeans_model.labels_
  >>> metrics.silhouette_score(X, labels, metric='euclidean')
  ...                                                      # doctest: +ELLIPSIS
  0.55...

.. topic:: 参考

 * Peter J. Rousseeuw (1987). "Silhouettes: a Graphical Aid to the
   Interpretation and Validation of Cluster Analysis". Computational
   and Applied Mathematics 20: 53–65. doi:10.1016/0377-0427(87)90125-7.


优势
~~~~~~~~~~

- 分数在-1（错误分类）和+1（高密度分类）。0则对应相互重叠的分类

- 等分越高，对应约分离独立的分类——更符合聚类的概念


劣势
~~~~~~~~~

- Silhouette的系数对凸出的类的得分比较高，（对比基于密度的分类DBSCAN）。

