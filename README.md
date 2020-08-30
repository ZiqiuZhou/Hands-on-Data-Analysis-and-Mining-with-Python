# Hands-on-Data-Analysis-and-Mining-with-Python
### 数据挖掘的基本任务包括利用分类与预测、聚类分析、关联规则、时序模式、偏差检测、智能推荐等方法，帮助企业提取数据中蕴含的商业价值，提高企业的竞争力。
## 一般步骤:
### 数据取样
### 数据探索——EDA
- 通过检验数据集的数据质量、绘制图表、计算某些特征量等手段，对样本数据集的结构和规律进行分析的过程就是数据探索。数据探索有助于选择合适的数据预处理和建模方法，甚至可以完成一些通常由数据挖掘解决的问题
  - 数据质量分析
    - 缺失值(NaN或者特殊的字符)、异常值、不一致的值
  - 数据特征分析
    - 分布分析: 
      - 定量数据——对数据分桶，再绘制条形图，或者不分桶直接柱状图，相关性(特征之间及特征与因变量之间),数字特征分布可视化
      - 定性数据——饼状图、条形图, 特征分布：df.value_counts()和df.unique()/df.nunique()
      - 预测值的分布
    - 对比分析
    - 统计量分析(定量数据): df/series.sum(), cumsum(), mean(), var(), corr(), describe()
    - 周期性分析：随着时间的变化而呈现出某种周期变化趋势
    - 相关性分析
  - 统计绘图函数
    - plot(), pie(), hist()直方图(分布或series.plot.density()), bar()柱状图, box()箱线图
### 数据预处理
- 数据预处理的主要内容包括数据清洗、数据集成、数据变换和数据归约:
![Image text](https://github.com/ZiqiuZhou/Hands-on-Data-Analysis-and-Mining-with-Python/blob/master/IMG/%E6%95%B0%E6%8D%AE%E9%A2%84%E5%A4%84%E7%90%86.PNG)
  - 数据清洗
    - 缺失值(插补)：属性均值/中位数/固定值(df.fillna(), df.replace()), 回归插值，拉格朗日插值(from scipy.interpolate import lagrange)或者删除(df.dropna())
    - 异常值:平均修正，视为缺失值，删除或者不处理
  - 数据集成
  - 数据变换: 
    - 平方、开方、取对数
    - 标准化: 对于基于距离的挖掘算法很重要
      - min-max: (x - min) / max - min
      - (x -μ) / σ
    - 对类别模型做One-hot Encoder: pd.get_dummies() (树模型无须此步骤，反而需要对数值类别分桶)
    - 连续属性离散化: pd.cut()
    - 特征构造：利用已有的属性集构造出新的属性
    - 特征筛选：
      - 过滤式：移除低方差的特征；低相关系数特征(pearson和卡方检测)
      - 包裹式：从初始特征集合中不断的选择特征子集，训练学习器，根据学习器的性能来对子集进行评价，直到选择出最佳的子集
      - 嵌入式：嵌入式特征选择在学习器训练过程中自动地进行特征选择——L1和L2正则化或者树模型
  - 数据归约
    - 特征归约：合并特征(PCA)，或删除相关性低的特征
    - 数值归约：直方图分桶，聚类， 线性回归
  - 预处理主要函数：isnull()/notnull(), scipy.interpolate(), sklearn.decomposition PCA()
     
### 挖掘建模
- 模型调参(超参数)：
  - 贪心算法 https://www.jianshu.com/p/ab89df9759c8 (https://www.jianshu.com/p/ab89df9759c8)
  - 网格调参GridSearchCV https://blog.csdn.net/weixin_43172660/article/details/83032029
  - 贝叶斯调参 https://blog.csdn.net/linxid/article/details/81189154


### 模型融合
- 简单加权融合：
  - 回归（分类概率）：算术平均融合（Arithmetic mean），几何平均融合（Geometric mean）
  - 分类：投票（Voting)
  - 综合：排序融合(Rank averaging)，log融合
- Bagging
- Boosting
- Stacking
