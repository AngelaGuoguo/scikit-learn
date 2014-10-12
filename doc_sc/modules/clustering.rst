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

k-平均算法将 :math:`N` 个样本 :math:`X` 分配到 :math:`K` 不相接的类别 :math:`C` 中，每一个类别通过其中样本的平均值 :math:`\mu_j` 来表征。这些平均值通常成为类别的中心（"centroids"）。注意，它们通常并不是X中的数据点，尽管它们处在同一个空间。k-平均算法旨在选择中心来降低惯性::

.. math:: \sum_{i=0}^{n}\min_{\mu_j \in C}(||x_j - \mu_i||^2)

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

Mini Batch K-平均
------------------

:class:`MiniBatchKMeans` 是 :class:`KMeans` 的改进，通过mini-batches来减少计算时间。其仍然优化同样的目标函数。Mini-batches 是输入训练样本的随机抽取的子集。这个过程可以显著的降低k-平均收敛的计算时间。mini-batch k-平均算法的结果仅比标准算法略微差些。

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

其中 :math:`m` 是每一个中心的平均偏移向量，来最大的增加点的密度。其计算方法如下： is the neighborhood of samples within a given distance
around  and :math:`m` is the *mean shift* vector that is computed
for each centroid that
points towards a region of the maximum increase in the density of points. This
is computed using the following equation, effectively updating a centroid to be
the mean of the samples within its neighborhood:

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

对于两个类别，其旨在解决

For two clusters, it solves a convex relaxation of the `normalised
cuts <http://www.cs.berkeley.edu/~malik/papers/SM-ncut.pdf>`_ problem on
the similarity graph: cutting the graph in two so that the weight of the
edges cut is small compared to the weights of the edges inside each
cluster. This criteria is especially interesting when working on images:
graph vertices are pixels, and edges of the similarity graph are a
function of the gradient of the image.


.. |noisy_img| image:: ../auto_examples/cluster/images/plot_segmentation_toy_001.png
    :target: ../auto_examples/cluster/plot_segmentation_toy.html
    :scale: 50

.. |segmented_img| image:: ../auto_examples/cluster/images/plot_segmentation_toy_002.png
    :target: ../auto_examples/cluster/plot_segmentation_toy.html
    :scale: 50

.. centered:: |noisy_img| |segmented_img|

.. warning:: Transforming distance to well-behaved similarities

    Note that if the values of your similarity matrix are not well
    distributed, e.g. with negative values or with a distance matrix
    rather than a similarity, the spectral problem will be singular and
    the problem not solvable. In which case it is advised to apply a
    transformation to the entries of the matrix. For instance, in the
    case of a signed distance matrix, is common to apply a heat kernel::

        similarity = np.exp(-beta * distance / distance.std())

    See the examples for such an application.

.. topic:: Examples:

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

Different label assignment strategies
---------------------------------------

Different label assignment strategies can be used, corresponding to the
``assign_labels`` parameter of :class:`SpectralClustering`.
The ``"kmeans"`` strategy can match finer details of the data, but it can be
more unstable. In particular, unless you control the ``random_state``, it
may not be reproducible from run-to-run, as it depends on a random
initialization. On the other hand, the ``"discretize"`` strategy is 100%
reproducible, but it tends to create parcels of fairly even and
geometrical shape.

=====================================  =====================================
 ``assign_labels="kmeans"`              ``assign_labels="discretize"``
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

等级聚类（Hierarchical clustering）是一类聚类算法，其通过不断的合并和分割来构建嵌套的类别。这个类别的等级结构可以表示为树结构（或者dendrogram）。树的树根是一个包含所有样本的群，而叶节点是包含一个取样的类别。更多信息参考 `Wikipedia page
<http://en.wikipedia.org/wiki/Hierarchical_clustering>` 。

:class:`AgglomerativeClustering` 通过自下而上的方式构建等级聚类：每一个取样首先是自己独立的类别，然后类别不断的合并。合并的策略取决于以下的链接条件：

- **Ward** 最小化所有类别内的方差和。这是一个最小化方差的过程，因此和k-平均方法类似。ith an agglomerative hierarchical
  approach.
- **Maximum** or **complete linkage** minimizes the maximum distance between
  observations of pairs of clusters.
- **Average linkage** minimizes the average of the distances between all
  observations of pairs of clusters.

:class:`AgglomerativeClustering` can also scale to large number of samples
when it is used jointly with a connectivity matrix, but is computationally
expensive when no connectivity constraints are added between samples: it
considers at each step all the possible merges.

.. topic:: :class:`FeatureAgglomeration`

   The :class:`FeatureAgglomeration` uses agglomerative clustering to
   group together features that look very similar, thus decreasing the
   number of features. It is a dimensionality reduction tool, see
   :ref:`data_reduction`.

Different linkage type: Ward, complete and average linkage
-----------------------------------------------------------

:class:`AgglomerativeClustering` supports Ward, average, and complete
linkage strategies.

.. image:: ../auto_examples/cluster/images/plot_digits_linkage_001.png
    :target: ../auto_examples/cluster/plot_digits_linkage.html
    :scale: 43

.. image:: ../auto_examples/cluster/images/plot_digits_linkage_002.png
    :target: ../auto_examples/cluster/plot_digits_linkage.html
    :scale: 43

.. image:: ../auto_examples/cluster/images/plot_digits_linkage_003.png
    :target: ../auto_examples/cluster/plot_digits_linkage.html
    :scale: 43


Agglomerative cluster has a "rich get richer" behavior that leads to
uneven cluster sizes. In this regard, complete linkage is the worst
strategy, and Ward gives the most regular sizes. However, the affinity
(or distance used in clustering) cannot be varied with Ward, thus for non
Euclidean metrics, average linkage is a good alternative.

.. topic:: Examples:

 * :ref:`example_cluster_plot_digits_linkage.py`: exploration of the
   different linkage strategies in a real dataset.


Adding connectivity constraints
-------------------------------

An interesting aspect of :class:`AgglomerativeClustering` is that
connectivity constraints can be added to this algorithm (only adjacent
clusters can be merged together), through a connectivity matrix that defines
for each sample the neighboring samples following a given structure of the
data. For instance, in the swiss-roll example below, the connectivity
constraints forbid the merging of points that are not adjacent on the swiss
roll, and thus avoid forming clusters that extend across overlapping folds of
the roll.

.. |unstructured| image:: ../auto_examples/cluster/images/plot_ward_structured_vs_unstructured_001.png
        :target: ../auto_examples/cluster/plot_ward_structured_vs_unstructured.html
        :scale: 49

.. |structured| image:: ../auto_examples/cluster/images/plot_ward_structured_vs_unstructured_002.png
        :target: ../auto_examples/cluster/plot_ward_structured_vs_unstructured.html
        :scale: 49

.. centered:: |unstructured| |structured|

These constraint are useful to impose a certain local structure, but they
also make the algorithm faster, especially when the number of the samples
is high.

The connectivity constraints are imposed via an connectivity matrix: a
scipy sparse matrix that has elements only at the intersection of a row
and a column with indices of the dataset that should be connected. This
matrix can be constructed from a-priori information: for instance, you
may wish to cluster web pages by only merging pages with a link pointing
from one to another. It can also be learned from the data, for instance
using :func:`sklearn.neighbors.kneighbors_graph` to restrict
merging to nearest neighbors as in :ref:`this example
<example_cluster_plot_agglomerative_clustering.py>`, or
using :func:`sklearn.feature_extraction.image.grid_to_graph` to
enable only merging of neighboring pixels on an image, as in the
:ref:`Lena <example_cluster_plot_lena_ward_segmentation.py>` example.

.. topic:: Examples:

 * :ref:`example_cluster_plot_lena_ward_segmentation.py`: Ward clustering
   to split the image of lena in regions.

 * :ref:`example_cluster_plot_ward_structured_vs_unstructured.py`: Example of
   Ward algorithm on a swiss-roll, comparison of structured approaches
   versus unstructured approaches.

 * :ref:`example_cluster_plot_feature_agglomeration_vs_univariate_selection.py`:
   Example of dimensionality reduction with feature agglomeration based on
   Ward hierarchical clustering.

 * :ref:`example_cluster_plot_agglomerative_clustering.py`

.. warning:: **Connectivity constraints with average and complete linkage**

    Connectivity constraints and complete or average linkage can enhance
    the 'rich getting richer' aspect of agglomerative clustering,
    particularly so if they are built with
    :func:`sklearn.neighbors.kneighbors_graph`. In the limit of a small
    number of clusters, they tend to give a few macroscopically occupied
    clusters and almost empty ones. (see the discussion in
    :ref:`example_cluster_plot_agglomerative_clustering.py`).

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


Varying the metric
-------------------

Average and complete linkage can be used with a variety of distances (or
affinities), in particular Euclidean distance (*l2*), Manhattan distance
(or Cityblock, or *l1*), cosine distance, or any precomputed affinity
matrix.

* *l1* distance is often good for sparse features, or sparse noise: ie
  many of the features are zero, as in text mining using occurences of
  rare words.

* *cosine* distance is interesting because it is invariant to global
  scalings of the signal.

The guidelines for choosing a metric is to use one that maximizes the
distance between samples in different classes, and minimizes that within
each class.

.. image:: ../auto_examples/cluster/images/plot_agglomerative_clustering_metrics_005.png
    :target: ../auto_examples/cluster/plot_agglomerative_clustering_metrics.html
    :scale: 32

.. image:: ../auto_examples/cluster/images/plot_agglomerative_clustering_metrics_006.png
    :target: ../auto_examples/cluster/plot_agglomerative_clustering_metrics.html
    :scale: 32

.. image:: ../auto_examples/cluster/images/plot_agglomerative_clustering_metrics_007.png
    :target: ../auto_examples/cluster/plot_agglomerative_clustering_metrics.html
    :scale: 32

.. topic:: Examples:

 * :ref:`example_cluster_plot_agglomerative_clustering_metrics.py`


.. _dbscan:

DBSCAN
======

The :class:`DBSCAN` algorithm views clusters as areas of high density
separated by areas of low density. Due to this rather generic view, clusters
found by DBSCAN can be any shape, as opposed to k-means which assumes that
clusters are convex shaped. The central component to the DBSCAN is the concept
of *core samples*, which are samples that are in areas of high density. A
cluster is therefore a set of core samples, each close to each other
(measured by some distance measure)
and a set of non-core samples that are close to a core sample (but are not
themselves core samples). There are two parameters to the algorithm,
``min_samples`` and ``eps``,
which define formally what we mean when we say *dense*.
Higher ``min_samples`` or lower ``eps``
indicate higher density necessary to form a cluster.

More formally, we define a core sample as being a sample in the dataset such
that there exist ``min_samples`` other samples within a distance of
``eps``, which are defined as *neighbors* of the core sample. This tells
us that the core sample is in a dense area of the vector space. A cluster
is a set of core samples, that can be built by recursively by taking a core
sample, finding all of its neighbors that are core samples, finding all of
*their* neighbors that are core samples, and so on. A cluster also has a
set of non-core samples, which are samples that are neighbors of a core sample
in the cluster but are not themselves core samples. Intuitively, these samples
are on the fringes of a cluster.

Any core sample is part of a cluster, by definition. Further, any cluster has
at least ``min_samples`` points in it, following the definition of a core
sample. For any sample that is not a core sample, and does have a
distance higher than ``eps`` to any core sample, it is considered an outlier by
the algorithm.

In the figure below, the color indicates cluster membership, with large circles
indicating core samples found by the algorithm. Smaller circles are non-core
samples that are still part of a cluster. Moreover, the outliers are indicated
by black points below.

.. |dbscan_results| image:: ../auto_examples/cluster/images/plot_dbscan_001.png
        :target: ../auto_examples/cluster/plot_dbscan.html
        :scale: 50

.. centered:: |dbscan_results|

.. topic:: Examples:

    * :ref:`example_cluster_plot_dbscan.py`

.. topic:: Implementation

    The algorithm is non-deterministic, but the core samples will
    always belong to the same clusters (although the labels may be
    different). The non-determinism comes from deciding to which cluster a
    non-core sample belongs. A non-core sample can have a distance lower
    than ``eps`` to two core samples in different clusters. By the
    triangular inequality, those two core samples must be more distant than
    ``eps`` from each other, or they would be in the same cluster. The non-core
    sample is assigned to whichever cluster is generated first, where
    the order is determined randomly. Other than the ordering of
    the dataset, the algorithm is deterministic, making the results relatively
    stable between runs on the same data.

    The current implementation uses ball trees and kd-trees
    to determine the neighborhood of points,
    which avoids calculating the full distance matrix
    (as was done in scikit-learn versions before 0.14).
    The possibility to use custom metrics is retained;
    for details, see :class:`NearestNeighbors`.

.. topic:: References:

 * "A Density-Based Algorithm for Discovering Clusters in Large Spatial Databases
   with Noise"
   Ester, M., H. P. Kriegel, J. Sander, and X. Xu,
   In Proceedings of the 2nd International Conference on Knowledge Discovery
   and Data Mining, Portland, OR, AAAI Press, pp. 226–231. 1996

.. _clustering_evaluation:

Clustering performance evaluation
=================================

Evaluating the performance of a clustering algorithm is not as trivial as
counting the number of errors or the precision and recall of a supervised
classification algorithm. In particular any evaluation metric should not
take the absolute values of the cluster labels into account but rather
if this clustering define separations of the data similar to some ground
truth set of classes or satisfying some assumption such that members
belong to the same class are more similar that members of different
classes according to some similarity metric.

.. currentmodule:: sklearn.metrics


Adjusted Rand index
-------------------

Presentation and usage
~~~~~~~~~~~~~~~~~~~~~~

Given the knowledge of the ground truth class assignments ``labels_true``
and our clustering algorithm assignments of the same samples
``labels_pred``, the **adjusted Rand index** is a function that measures
the **similarity** of the two assignments, ignoring permutations and **with
chance normalization**::

  >>> from sklearn import metrics
  >>> labels_true = [0, 0, 0, 1, 1, 1]
  >>> labels_pred = [0, 0, 1, 1, 2, 2]

  >>> metrics.adjusted_rand_score(labels_true, labels_pred)  # doctest: +ELLIPSIS
  0.24...

One can permute 0 and 1 in the predicted labels, rename 2 to 3, and get
the same score::

  >>> labels_pred = [1, 1, 0, 0, 3, 3]
  >>> metrics.adjusted_rand_score(labels_true, labels_pred)  # doctest: +ELLIPSIS
  0.24...

Furthermore, :func:`adjusted_rand_score` is **symmetric**: swapping the argument
does not change the score. It can thus be used as a **consensus
measure**::

  >>> metrics.adjusted_rand_score(labels_pred, labels_true)  # doctest: +ELLIPSIS
  0.24...

Perfect labeling is scored 1.0::

  >>> labels_pred = labels_true[:]
  >>> metrics.adjusted_rand_score(labels_true, labels_pred)
  1.0

Bad (e.g. independent labelings) have negative or close to 0.0 scores::

  >>> labels_true = [0, 1, 2, 0, 3, 4, 5, 1]
  >>> labels_pred = [1, 1, 0, 0, 2, 2, 2, 2]
  >>> metrics.adjusted_rand_score(labels_true, labels_pred)  # doctest: +ELLIPSIS
  -0.12...


Advantages
~~~~~~~~~~

- **Random (uniform) label assignments have a ARI score close to 0.0**
  for any value of ``n_clusters`` and ``n_samples`` (which is not the
  case for raw Rand index or the V-measure for instance).

- **Bounded range [-1, 1]**: negative values are bad (independent
  labelings), similar clusterings have a positive ARI, 1.0 is the perfect
  match score.

- **No assumption is made on the cluster structure**: can be used
  to compare clustering algorithms such as k-means which assumes isotropic
  blob shapes with results of spectral clustering algorithms which can
  find cluster with "folded" shapes.


Drawbacks
~~~~~~~~~

- Contrary to inertia, **ARI requires knowledge of the ground truth
  classes** while is almost never available in practice or requires manual
  assignment by human annotators (as in the supervised learning setting).

  However ARI can also be useful in a purely unsupervised setting as a
  building block for a Consensus Index that can be used for clustering
  model selection (TODO).


.. topic:: Examples:

 * :ref:`example_cluster_plot_adjusted_for_chance_measures.py`: Analysis of
   the impact of the dataset size on the value of clustering measures
   for random assignments.


Mathematical formulation
~~~~~~~~~~~~~~~~~~~~~~~~

If C is a ground truth class assignment and K the clustering, let us
define :math:`a` and :math:`b` as:

- :math:`a`, the number of pairs of elements that are in the same set
  in C and in the same set in K

- :math:`b`, the number of pairs of elements that are in different sets
  in C and in different sets in K

The raw (unadjusted) Rand index is then given by:

.. math:: \text{RI} = \frac{a + b}{C_2^{n_{samples}}}

Where :math:`C_2^{n_{samples}}` is the total number of possible pairs
in the dataset (without ordering).

However the RI score does not guarantee that random label assignments
will get a value close to zero (esp. if the number of clusters is in
the same order of magnitude as the number of samples).

To counter this effect we can discount the expected RI :math:`E[\text{RI}]` of
random labelings by defining the adjusted Rand index as follows:

.. math:: \text{ARI} = \frac{\text{RI} - E[\text{RI}]}{\max(\text{RI}) - E[\text{RI}]}

.. topic:: References

 * `Comparing Partitions
   <http://www.springerlink.com/content/x64124718341j1j0/>`_
   L. Hubert and P. Arabie, Journal of Classification 1985

 * `Wikipedia entry for the adjusted Rand index
   <http://en.wikipedia.org/wiki/Rand_index#Adjusted_Rand_index>`_


Mutual Information based scores
-------------------------------

Presentation and usage
~~~~~~~~~~~~~~~~~~~~~~

Given the knowledge of the ground truth class assignments ``labels_true`` and
our clustering algorithm assignments of the same samples ``labels_pred``, the
**Mutual Information** is a function that measures the **agreement** of the two
assignments, ignoring permutations.  Two different normalized versions of this
measure are available, **Normalized Mutual Information(NMI)** and **Adjusted
Mutual Information(AMI)**. NMI is often used in the literature while AMI was
proposed more recently and is **normalized against chance**::

  >>> from sklearn import metrics
  >>> labels_true = [0, 0, 0, 1, 1, 1]
  >>> labels_pred = [0, 0, 1, 1, 2, 2]

  >>> metrics.adjusted_mutual_info_score(labels_true, labels_pred)  # doctest: +ELLIPSIS
  0.22504...

One can permute 0 and 1 in the predicted labels, rename 2 to 3 and get
the same score::

  >>> labels_pred = [1, 1, 0, 0, 3, 3]
  >>> metrics.adjusted_mutual_info_score(labels_true, labels_pred)  # doctest: +ELLIPSIS
  0.22504...

All, :func:`mutual_info_score`, :func:`adjusted_mutual_info_score` and
:func:`normalized_mutual_info_score` are symmetric: swapping the argument does
not change the score. Thus they can be used as a **consensus measure**::

  >>> metrics.adjusted_mutual_info_score(labels_pred, labels_true)  # doctest: +ELLIPSIS
  0.22504...

Perfect labeling is scored 1.0::

  >>> labels_pred = labels_true[:]
  >>> metrics.adjusted_mutual_info_score(labels_true, labels_pred)
  1.0

  >>> metrics.normalized_mutual_info_score(labels_true, labels_pred)
  1.0

This is not true for ``mutual_info_score``, which is therefore harder to judge::

  >>> metrics.mutual_info_score(labels_true, labels_pred)  # doctest: +ELLIPSIS
  0.69...

Bad (e.g. independent labelings) have non-positive scores::

  >>> labels_true = [0, 1, 2, 0, 3, 4, 5, 1]
  >>> labels_pred = [1, 1, 0, 0, 2, 2, 2, 2]
  >>> metrics.adjusted_mutual_info_score(labels_true, labels_pred)  # doctest: +ELLIPSIS
  -0.10526...


Advantages
~~~~~~~~~~

- **Random (uniform) label assignments have a AMI score close to 0.0**
  for any value of ``n_clusters`` and ``n_samples`` (which is not the
  case for raw Mutual Information or the V-measure for instance).

- **Bounded range [0, 1]**:  Values close to zero indicate two label
  assignments that are largely independent, while values close to one
  indicate significant agreement. Further, values of exactly 0 indicate
  **purely** independent label assignments and a AMI of exactly 1 indicates
  that the two label assignments are equal (with or without permutation).

- **No assumption is made on the cluster structure**: can be used
  to compare clustering algorithms such as k-means which assumes isotropic
  blob shapes with results of spectral clustering algorithms which can
  find cluster with "folded" shapes.


Drawbacks
~~~~~~~~~

- Contrary to inertia, **MI-based measures require the knowledge
  of the ground truth classes** while almost never available in practice or
  requires manual assignment by human annotators (as in the supervised learning
  setting).

  However MI-based measures can also be useful in purely unsupervised setting as a
  building block for a Consensus Index that can be used for clustering
  model selection.

- NMI and MI are not adjusted against chance.


.. topic:: Examples:

 * :ref:`example_cluster_plot_adjusted_for_chance_measures.py`: Analysis of
   the impact of the dataset size on the value of clustering measures
   for random assignments. This example also includes the Adjusted Rand
   Index.


Mathematical formulation
~~~~~~~~~~~~~~~~~~~~~~~~

Assume two label assignments (of the same N objects), :math:`U` and :math:`V`.
Their entropy is the amount of uncertainty for a partition set, defined by:

.. math:: H(U) = \sum_{i=1}^{|U|}P(i)\log(P(i))

where :math:`P(i) = |U_i| / N` is the probability that an object picked at
random from :math:`U` falls into class :math:`U_i`. Likewise for :math:`V`:

.. math:: H(V) = \sum_{j=1}^{|V|}P'(j)\log(P'(j))

With :math:`P'(j) = |V_j| / N`. The mutual information (MI) between :math:`U`
and :math:`V` is calculated by:

.. math:: \text{MI}(U, V) = \sum_{i=1}^{|U|}\sum_{j=1}^{|V|}P(i, j)\log\left(\frac{P(i,j)}{P(i)P'(j)}\right)

where :math:`P(i, j) = |U_i \cap V_j| / N` is the probability that an object
picked at random falls into both classes :math:`U_i` and :math:`V_j`.

The normalized mutual information is defined as

.. math:: \text{NMI}(U, V) = \frac{\text{MI}(U, V)}{\sqrt{H(U)H(V)}}

This value of the mutual information and also the normalized variant is not
adjusted for chance and will tend to increase as the number of different labels
(clusters) increases, regardless of the actual amount of "mutual information"
between the label assignments.

The expected value for the mutual information can be calculated using the
following equation, from Vinh, Epps, and Bailey, (2009). In this equation,
:math:`a_i = |U_i|` (the number of elements in :math:`U_i`) and
:math:`b_j = |V_j|` (the number of elements in :math:`V_j`).


.. math:: E[\text{MI}(U,V)]=\sum_{i=1}^|U| \sum_{j=1}^|V| \sum_{n_{ij}=(a_i+b_j-N)^+
   }^{\min(a_i, b_j)} \frac{n_{ij}}{N}\log \left( \frac{ N.n_{ij}}{a_i b_j}\right)
   \frac{a_i!b_j!(N-a_i)!(N-b_j)!}{N!n_{ij}!(a_i-n_{ij})!(b_j-n_{ij})!
   (N-a_i-b_j+n_{ij})!}

Using the expected value, the adjusted mutual information can then be
calculated using a similar form to that of the adjusted Rand index:

.. math:: \text{AMI} = \frac{\text{MI} - E[\text{MI}]}{\max(H(U), H(V)) - E[\text{MI}]}

.. topic:: References

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

Homogeneity, completeness and V-measure
---------------------------------------

Presentation and usage
~~~~~~~~~~~~~~~~~~~~~~

Given the knowledge of the ground truth class assignments of the samples,
it is possible to define some intuitive metric using conditional entropy
analysis.

In particular Rosenberg and Hirschberg (2007) define the following two
desirable objectives for any cluster assignment:

- **homogeneity**: each cluster contains only members of a single class.

- **completeness**: all members of a given class are assigned to the same
  cluster.

We can turn those concept as scores :func:`homogeneity_score` and
:func:`completeness_score`. Both are bounded below by 0.0 and above by
1.0 (higher is better)::

  >>> from sklearn import metrics
  >>> labels_true = [0, 0, 0, 1, 1, 1]
  >>> labels_pred = [0, 0, 1, 1, 2, 2]

  >>> metrics.homogeneity_score(labels_true, labels_pred)  # doctest: +ELLIPSIS
  0.66...

  >>> metrics.completeness_score(labels_true, labels_pred) # doctest: +ELLIPSIS
  0.42...

Their harmonic mean called **V-measure** is computed by
:func:`v_measure_score`::

  >>> metrics.v_measure_score(labels_true, labels_pred)    # doctest: +ELLIPSIS
  0.51...

The V-measure is actually equivalent to the mutual information (NMI)
discussed above normalized by the sum of the label entropies [B2011]_.

Homogeneity, completeness and V-measure can be computed at once using
:func:`homogeneity_completeness_v_measure` as follows::

  >>> metrics.homogeneity_completeness_v_measure(labels_true, labels_pred)
  ...                                                      # doctest: +ELLIPSIS
  (0.66..., 0.42..., 0.51...)

The following clustering assignment is slightly better, since it is
homogeneous but not complete::

  >>> labels_pred = [0, 0, 0, 1, 2, 2]
  >>> metrics.homogeneity_completeness_v_measure(labels_true, labels_pred)
  ...                                                      # doctest: +ELLIPSIS
  (1.0, 0.68..., 0.81...)

.. note::

  :func:`v_measure_score` is **symmetric**: it can be used to evaluate
  the **agreement** of two independent assignments on the same dataset.

  This is not the case for :func:`completeness_score` and
  :func:`homogeneity_score`: both are bound by the relationship::

    homogeneity_score(a, b) == completeness_score(b, a)


Advantages
~~~~~~~~~~

- **Bounded scores**: 0.0 is as bad as it can be, 1.0 is a perfect score

- Intuitive interpretation: clustering with bad V-measure can be
  **qualitatively analyzed in terms of homogeneity and completeness**
  to better feel what 'kind' of mistakes is done by the assignment.

- **No assumption is made on the cluster structure**: can be used
  to compare clustering algorithms such as k-means which assumes isotropic
  blob shapes with results of spectral clustering algorithms which can
  find cluster with "folded" shapes.


Drawbacks
~~~~~~~~~

- The previously introduced metrics are **not normalized with regards to
  random labeling**: this means that depending on the number of samples,
  clusters and ground truth classes, a completely random labeling will
  not always yield the same values for homogeneity, completeness and
  hence v-measure. In particular **random labeling won't yield zero
  scores especially when the number of clusters is large**.

  This problem can safely be ignored when the number of samples is more
  than a thousand and the number of clusters is less than 10. **For
  smaller sample sizes or larger number of clusters it is safer to use
  an adjusted index such as the Adjusted Rand Index (ARI)**.

.. figure:: ../auto_examples/cluster/images/plot_adjusted_for_chance_measures_001.png
   :target: ../auto_examples/cluster/plot_adjusted_for_chance_measures.html
   :align: center
   :scale: 100

- These metrics **require the knowledge of the ground truth classes** while
  almost never available in practice or requires manual assignment by
  human annotators (as in the supervised learning setting).


.. topic:: Examples:

 * :ref:`example_cluster_plot_adjusted_for_chance_measures.py`: Analysis of
   the impact of the dataset size on the value of clustering measures
   for random assignments.


Mathematical formulation
~~~~~~~~~~~~~~~~~~~~~~~~

Homogeneity and completeness scores are formally given by:

.. math:: h = 1 - \frac{H(C|K)}{H(C)}

.. math:: c = 1 - \frac{H(K|C)}{H(K)}

where :math:`H(C|K)` is the **conditional entropy of the classes given
the cluster assignments** and is given by:

.. math:: H(C|K) = - \sum_{c=1}^{|C|} \sum_{k=1}^{|K|} \frac{n_{c,k}}{n}
          \cdot \log\left(\frac{n_{c,k}}{n_k}\right)

and :math:`H(C)` is the **entropy of the classes** and is given by:

.. math:: H(C) = - \sum_{c=1}^{|C|} \frac{n_c}{n} \cdot \log\left(\frac{n_c}{n}\right)

with :math:`n` the total number of samples, :math:`n_c` and :math:`n_k`
the number of samples respectively belonging to class :math:`c` and
cluster :math:`k`, and finally :math:`n_{c,k}` the number of samples
from class :math:`c` assigned to cluster :math:`k`.

The **conditional entropy of clusters given class** :math:`H(K|C)` and the
**entropy of clusters** :math:`H(K)` are defined in a symmetric manner.

Rosenberg and Hirschberg further define **V-measure** as the **harmonic
mean of homogeneity and completeness**:

.. math:: v = 2 \cdot \frac{h \cdot c}{h + c}

.. topic:: References

 .. [RH2007] `V-Measure: A conditional entropy-based external cluster evaluation
   measure <http://acl.ldc.upenn.edu/D/D07/D07-1043.pdf>`_
   Andrew Rosenberg and Julia Hirschberg, 2007

 .. [B2011] `Identication and Characterization of Events in Social Media
   <http://www.cs.columbia.edu/~hila/hila-thesis-distributed.pdf>`_, Hila
   Becker, PhD Thesis.

.. _silhouette_coefficient:

Silhouette Coefficient
----------------------

Presentation and usage
~~~~~~~~~~~~~~~~~~~~~~

If the ground truth labels are not known, evaluation must be performed using
the model itself. The Silhouette Coefficient
(:func:`sklearn.metrics.silhouette_score`)
is an example of such an evaluation, where a
higher Silhouette Coefficient score relates to a model with better defined
clusters. The Silhouette Coefficient is defined for each sample and is composed
of two scores:

- **a**: The mean distance between a sample and all other points in the same
  class.

- **b**: The mean distance between a sample and all other points in the *next
  nearest cluster*.

The Silhoeutte Coefficient *s* for a single sample is then given as:

.. math:: s = \frac{b - a}{max(a, b)}

The Silhouette Coefficient for a set of samples is given as the mean of the
Silhouette Coefficient for each sample.


  >>> from sklearn import metrics
  >>> from sklearn.metrics import pairwise_distances
  >>> from sklearn import datasets
  >>> dataset = datasets.load_iris()
  >>> X = dataset.data
  >>> y = dataset.target

In normal usage, the Silhouette Coefficient is applied to the results of a
cluster analysis.

  >>> import numpy as np
  >>> from sklearn.cluster import KMeans
  >>> kmeans_model = KMeans(n_clusters=3, random_state=1).fit(X)
  >>> labels = kmeans_model.labels_
  >>> metrics.silhouette_score(X, labels, metric='euclidean')
  ...                                                      # doctest: +ELLIPSIS
  0.55...

.. topic:: References

 * Peter J. Rousseeuw (1987). "Silhouettes: a Graphical Aid to the
   Interpretation and Validation of Cluster Analysis". Computational
   and Applied Mathematics 20: 53–65. doi:10.1016/0377-0427(87)90125-7.


Advantages
~~~~~~~~~~

- The score is bounded between -1 for incorrect clustering and +1 for highly
  dense clustering. Scores around zero indicate overlapping clusters.

- The score is higher when clusters are dense and well separated, which relates
  to a standard concept of a cluster.


Drawbacks
~~~~~~~~~

- The Silhouette Coefficient is generally higher for convex clusters than other
  concepts of clusters, such as density based clusters like those obtained
  through DBSCAN.

