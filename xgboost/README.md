与实现的RandomForest类似，XGBoost也是由多棵树构成，区别在于： 
1. XGBoost的第T棵数学习的是真实值与前面T-1棵树之和的均方误差；即RandomForest是取所有树中预测概率最大的值作为最终结果，而XGBoost是把所有树的预测值求和作为最终结果；
2. XGBoost在构建树是所采用的结点增益方法不同，RandomForest是采用信息熵增益最大的，XGBoost见下图的证明的结论部分；
3. XGBoost在构建树是计算阶段的阈值的方法不同，RandomForest是采用信息熵增益最大是所对应的特征值，XGBoost的计算方式见下图的结论部分

![image](https://github.com/XPping/machine-learning/raw/master/xgboost/algorithm/1.jpg) 
![image](https://github.com/XPping/machine-learning/raw/master/result/xgboost.jpg) 
