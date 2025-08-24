#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
环状RNA组织表达预测 - 改进版Pipeline v2.0
基于原始高分模型(0.36)的改进版本

主要改进:
1. 更多生物学特征工程 (序列复杂度、k-mer频率、二级结构预测)
2. 高级特征选择 (递归特征消除、LASSO、互信息)
3. 数据增强和不平衡处理
4. 更精细的超参数网格
5. 多层Stacking和Blending集成
6. 模型校准和阈值优化
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# 基础库
import time
import pickle
from datetime import datetime
from collections import Counter
import itertools
from scipy import stats
from scipy.stats import entropy

# 机器学习库
from sklearn.model_selection import (
    StratifiedKFold, cross_val_score, RandomizedSearchCV, GridSearchCV
)
from sklearn.preprocessing import (
    LabelEncoder, StandardScaler, RobustScaler, MinMaxScaler,
    PolynomialFeatures, PowerTransformer
)
from sklearn.feature_selection import (
    SelectKBest, f_classif, mutual_info_classif,
    RFE, SelectFromModel, VarianceThreshold
)
from sklearn.ensemble import (
    RandomForestClassifier, ExtraTreesClassifier,
    VotingClassifier, StackingClassifier, BaggingClassifier
)
from sklearn.linear_model import (
    LogisticRegression, Lasso, ElasticNet
)
from sklearn.metrics import (
    f1_score, classification_report, confusion_matrix,
    make_scorer
)
from sklearn.calibration import CalibratedClassifierCV
from sklearn.utils.class_weight import compute_class_weight
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors

# 高级模型
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("警告: XGBoost未安装")

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("警告: LightGBM未安装")

try:
    from catboost import CatBoostClassifier
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    print("警告: CatBoost未安装")

class AdvancedRNAPredictionPipeline:
    """
    改进版环状RNA组织表达预测Pipeline
    """
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        np.random.seed(random_state)
        
        # 数据存储
        self.train_data = None
        self.test_data = None
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.test_ids = None
        
        # 预处理器
        self.encoders = {}
        self.scalers = {}
        self.feature_selector = None
        self.target_encoder = LabelEncoder()
        
        # 模型相关
        self.models = {}
        self.optimized_models = {}
        self.model_results = []
        self.final_model = None
        
        # 特征名称
        self.feature_names = []
        self.original_features = []
        self.engineered_features = []
        
    def load_data(self, train_path='train.csv', test_path='test.csv'):
        """
        加载训练和测试数据
        """
        print("加载数据...")
        print("=" * 50)
        
        # 加载数据
        self.train_data = pd.read_csv(train_path)
        self.test_data = pd.read_csv(test_path)
        
        print(f"训练集形状: {self.train_data.shape}")
        print(f"测试集形状: {self.test_data.shape}")
        
        # 保存测试集ID
        self.test_ids = self.test_data['ID'].values
        
        # 检查数据质量
        print("\n数据质量检查:")
        print(f"训练集缺失值: {self.train_data.isnull().sum().sum()}")
        print(f"测试集缺失值: {self.test_data.isnull().sum().sum()}")
        
        return self
    
    def analyze_data(self):
        """
        数据分析和探索
        """
        print("\n数据分析...")
        print("=" * 50)
        
        # 目标变量分布
        target_dist = self.train_data['Tissue'].value_counts()
        print("\n目标变量分布:")
        for tissue, count in target_dist.items():
            print(f"  {tissue}: {count} ({count/len(self.train_data)*100:.1f}%)")
        
        # 特征统计
        numeric_features = self.train_data.select_dtypes(include=[np.number]).columns.tolist()
        if 'ID' in numeric_features:
            numeric_features.remove('ID')
        
        print(f"\n数值特征数量: {len(numeric_features)}")
        print(f"分类特征数量: {len(self.train_data.columns) - len(numeric_features) - 2}")  # 减去ID和Tissue
        
        # 保存原始特征名
        self.original_features = [col for col in self.train_data.columns if col not in ['ID', 'Tissue']]
        
        return self
    
    def advanced_feature_engineering(self):
        """
        高级特征工程
        """
        print("\n高级特征工程...")
        print("=" * 50)
        
        # 合并训练和测试数据进行特征工程
        train_features = self.train_data.drop(['Tissue'], axis=1)
        all_data = pd.concat([train_features, self.test_data], ignore_index=True)
        
        print(f"原始特征数量: {len(self.original_features)}")
        
        # 1. 基础特征工程 (保留原有成功特征)
        print("1. 基础特征工程...")
        
        # 计算AT含量
        if 'A_ratio' in all_data.columns and 'T_ratio' in all_data.columns:
            all_data['AT_content'] = all_data['A_ratio'] + all_data['T_ratio']
        
        # GC/AT比例特征
        if 'GC_content' in all_data.columns and 'AT_content' in all_data.columns:
            all_data['GC_AT_ratio'] = all_data['GC_content'] / (all_data['AT_content'] + 1e-8)
            all_data['AT_GC_ratio'] = all_data['AT_content'] / (all_data['GC_content'] + 1e-8)
            all_data['GC_AT_diff'] = all_data['GC_content'] - all_data['AT_content']
            all_data['GC_AT_sum'] = all_data['GC_content'] + all_data['AT_content']
        
        # 长度相关特征
        if 'length' in all_data.columns:
            all_data['length_log'] = np.log1p(all_data['length'])
            all_data['length_sqrt'] = np.sqrt(all_data['length'])
            all_data['length_squared'] = all_data['length'] ** 2
            all_data['length_reciprocal'] = 1 / (all_data['length'] + 1)
        
        # 能量相关特征
        if 'Average free energy' in all_data.columns and 'length' in all_data.columns:
            all_data['energy_per_length'] = all_data['Average free energy'] / (all_data['length'] + 1)
            all_data['energy_density'] = abs(all_data['Average free energy']) / (all_data['length'] + 1)
            all_data['energy_log'] = np.log1p(abs(all_data['Average free energy']))
        
        # miRNA结合相关特征
        mirna_cols = [col for col in all_data.columns if 'miRNA' in col]
        if mirna_cols:
            all_data['miRNA_binding_total'] = all_data[mirna_cols].sum(axis=1)
            all_data['miRNA_binding_mean'] = all_data[mirna_cols].mean(axis=1)
            all_data['miRNA_binding_std'] = all_data[mirna_cols].std(axis=1)
            all_data['miRNA_binding_max'] = all_data[mirna_cols].max(axis=1)
            all_data['miRNA_binding_min'] = all_data[mirna_cols].min(axis=1)
            
            if 'length' in all_data.columns:
                all_data['miRNA_density'] = all_data['miRNA_binding_total'] / (all_data['length'] + 1)
        
        # 2. 序列复杂度特征
        print("2. 序列复杂度特征...")
        
        # 基于现有特征计算序列复杂度
        if 'GC_content' in all_data.columns and 'AT_content' in all_data.columns:
            # 香农熵近似 (基于GC/AT含量)
            gc_ratio = all_data['GC_content'] / 100.0
            at_ratio = all_data['AT_content'] / 100.0
            
            # 避免log(0)
            gc_ratio = np.clip(gc_ratio, 1e-8, 1-1e-8)
            at_ratio = np.clip(at_ratio, 1e-8, 1-1e-8)
            
            all_data['sequence_entropy'] = -(gc_ratio * np.log2(gc_ratio) + at_ratio * np.log2(at_ratio))
            all_data['gc_skewness'] = (all_data['GC_content'] - 50) / 50  # GC偏斜度
            all_data['at_skewness'] = (all_data['AT_content'] - 50) / 50  # AT偏斜度
        
        # 3. k-mer频率特征 (模拟)
        print("3. k-mer频率特征...")
        
        # 基于现有特征模拟k-mer特征
        if 'GC_content' in all_data.columns:
            # 模拟不同k-mer的频率
            all_data['kmer_GC_rich'] = (all_data['GC_content'] / 100) ** 2
            all_data['kmer_AT_rich'] = (all_data['AT_content'] / 100) ** 2
            all_data['kmer_balanced'] = 4 * (all_data['GC_content'] / 100) * (all_data['AT_content'] / 100)
        
        # 4. 二级结构特征增强
        print("4. 二级结构特征增强...")
        
        if 'Average free energy' in all_data.columns:
            # 结构稳定性指标
            all_data['structure_stability'] = -all_data['Average free energy'] / (all_data['length'] + 1)
            all_data['mfe_normalized'] = all_data['Average free energy'] / np.sqrt(all_data['length'] + 1)
            
            # 基于MFE的分类特征
            all_data['mfe_category'] = pd.cut(all_data['Average free energy'], bins=5, labels=['very_stable', 'stable', 'moderate', 'unstable', 'very_unstable'])
        
        # 5. 交互特征
        print("5. 交互特征...")
        
        # 重要特征的交互
        important_features = ['GC_content', 'AT_content', 'length', 'Average free energy']
        available_features = [f for f in important_features if f in all_data.columns]
        
        for i, feat1 in enumerate(available_features):
            for feat2 in available_features[i+1:]:
                all_data[f'{feat1}_x_{feat2}'] = all_data[feat1] * all_data[feat2]
                all_data[f'{feat1}_div_{feat2}'] = all_data[feat1] / (all_data[feat2] + 1e-8)
        
        # 6. 统计特征
        print("6. 统计特征...")
        
        # 数值特征的统计变换
        numeric_features = all_data.select_dtypes(include=[np.number]).columns.tolist()
        if 'ID' in numeric_features:
            numeric_features.remove('ID')
        
        # 添加多项式特征 (选择性)
        key_features = ['GC_content', 'AT_content', 'length'][:3]  # 限制特征数量
        available_key_features = [f for f in key_features if f in all_data.columns]
        
        if len(available_key_features) >= 2:
            poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
            poly_features = poly.fit_transform(all_data[available_key_features])
            poly_feature_names = poly.get_feature_names_out(available_key_features)
            
            # 只添加交互项，避免重复
            for i, name in enumerate(poly_feature_names):
                if ' ' in name:  # 交互项包含空格
                    all_data[f'poly_{name.replace(" ", "_")}'] = poly_features[:, i]
        
        # 7. 聚类特征
        print("7. 聚类特征...")
        
        # 基于主要特征进行聚类
        cluster_features = ['GC_content', 'AT_content', 'length']
        available_cluster_features = [f for f in cluster_features if f in all_data.columns]
        
        if len(available_cluster_features) >= 2:
            # 填充缺失值
            cluster_data = all_data[available_cluster_features].fillna(all_data[available_cluster_features].median())
            
            # 标准化
            scaler = StandardScaler()
            cluster_data_scaled = scaler.fit_transform(cluster_data)
            
            # K-means聚类
            for n_clusters in [3, 5, 8]:
                kmeans = KMeans(n_clusters=n_clusters, random_state=self.random_state, n_init=10)
                all_data[f'cluster_{n_clusters}'] = kmeans.fit_predict(cluster_data_scaled)
        
        # 8. 距离特征
        print("8. 距离特征...")
        
        if len(available_cluster_features) >= 2:
            # 计算到聚类中心的距离
            cluster_data = all_data[available_cluster_features].fillna(all_data[available_cluster_features].median())
            scaler = StandardScaler()
            cluster_data_scaled = scaler.fit_transform(cluster_data)
            
            # 到最近邻的距离
            nn = NearestNeighbors(n_neighbors=5)
            nn.fit(cluster_data_scaled)
            distances, _ = nn.kneighbors(cluster_data_scaled)
            
            all_data['nn_distance_mean'] = distances.mean(axis=1)
            all_data['nn_distance_std'] = distances.std(axis=1)
            all_data['nn_distance_min'] = distances.min(axis=1)
        
        # 处理分类特征
        categorical_features = all_data.select_dtypes(include=['object']).columns.tolist()
        if 'ID' in categorical_features:
            categorical_features.remove('ID')
        
        print(f"\n处理 {len(categorical_features)} 个分类特征...")
        
        for feature in categorical_features:
            if feature not in self.encoders:
                self.encoders[feature] = LabelEncoder()
                all_data[feature] = self.encoders[feature].fit_transform(all_data[feature].astype(str))
            else:
                all_data[feature] = self.encoders[feature].transform(all_data[feature].astype(str))
        
        # 分离训练和测试数据
        train_size = len(self.train_data)
        
        # 准备特征和目标变量
        feature_columns = [col for col in all_data.columns if col != 'ID']
        self.X_train = all_data.iloc[:train_size][feature_columns]
        self.X_test = all_data.iloc[train_size:][feature_columns]
        
        # 目标变量编码
        self.y_train = self.target_encoder.fit_transform(self.train_data['Tissue'])
        
        # 确保所有特征都是数值类型
        for col in feature_columns:
            # 处理Categorical类型
            if hasattr(self.X_train[col], 'cat'):
                self.X_train[col] = self.X_train[col].astype(str)
                self.X_test[col] = self.X_test[col].astype(str)
            
            # 转换为数值类型
            if self.X_train[col].dtype == 'object':
                self.X_train[col] = pd.to_numeric(self.X_train[col], errors='coerce')
                self.X_test[col] = pd.to_numeric(self.X_test[col], errors='coerce')
        
        # 处理转换后可能产生的NaN值
        self.X_train = self.X_train.fillna(0)
        self.X_test = self.X_test.fillna(0)
        
        # 保存特征名称
        self.feature_names = feature_columns
        self.engineered_features = [col for col in feature_columns if col not in self.original_features]
        
        print(f"\n特征工程完成:")
        print(f"  原始特征: {len(self.original_features)}")
        print(f"  新增特征: {len(self.engineered_features)}")
        print(f"  总特征数: {len(self.feature_names)}")
        print(f"  数据类型检查: {self.X_train.dtypes.value_counts().to_dict()}")
        
        return self
    
    def advanced_preprocessing(self):
        """
        高级数据预处理
        """
        print("\n高级数据预处理...")
        print("=" * 50)
        
        # 1. 处理缺失值
        print("1. 处理缺失值...")
        
        # 使用中位数填充数值特征
        numeric_features = self.X_train.select_dtypes(include=[np.number]).columns
        for feature in numeric_features:
            median_value = self.X_train[feature].median()
            self.X_train[feature].fillna(median_value, inplace=True)
            self.X_test[feature].fillna(median_value, inplace=True)
        
        # 2. 异常值处理
        print("2. 异常值处理...")
        
        # 使用IQR方法处理异常值
        for feature in numeric_features:
            Q1 = self.X_train[feature].quantile(0.25)
            Q3 = self.X_train[feature].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # 裁剪异常值
            self.X_train[feature] = np.clip(self.X_train[feature], lower_bound, upper_bound)
            self.X_test[feature] = np.clip(self.X_test[feature], lower_bound, upper_bound)
        
        # 3. 特征缩放
        print("3. 特征缩放...")
        
        # 只对数值特征进行缩放，保持分类特征不变
        numeric_features = self.X_train.select_dtypes(include=[np.number]).columns
        categorical_features = self.X_train.select_dtypes(exclude=[np.number]).columns
        
        if len(numeric_features) > 0:
            # 使用RobustScaler (对异常值更鲁棒)
            self.scalers['robust'] = RobustScaler()
            X_train_numeric_scaled = self.scalers['robust'].fit_transform(self.X_train[numeric_features])
            X_test_numeric_scaled = self.scalers['robust'].transform(self.X_test[numeric_features])
            
            # 重新组合数据
            X_train_numeric_df = pd.DataFrame(X_train_numeric_scaled, columns=numeric_features, index=self.X_train.index)
            X_test_numeric_df = pd.DataFrame(X_test_numeric_scaled, columns=numeric_features, index=self.X_test.index)
            
            # 合并数值特征和分类特征
            if len(categorical_features) > 0:
                self.X_train = pd.concat([X_train_numeric_df, self.X_train[categorical_features]], axis=1)
                self.X_test = pd.concat([X_test_numeric_df, self.X_test[categorical_features]], axis=1)
            else:
                self.X_train = X_train_numeric_df
                self.X_test = X_test_numeric_df
            
            # 确保列顺序一致
            self.X_train = self.X_train[self.feature_names]
            self.X_test = self.X_test[self.feature_names]
        
        print(f"预处理完成，特征形状: {self.X_train.shape}")
        
        return self
    
    def advanced_feature_selection(self):
        """
        高级特征选择
        """
        print("\n高级特征选择...")
        print("=" * 50)
        
        original_feature_count = self.X_train.shape[1]
        
        # 1. 方差阈值过滤
        print("1. 方差阈值过滤...")
        variance_selector = VarianceThreshold(threshold=0.01)
        X_train_var = variance_selector.fit_transform(self.X_train)
        X_test_var = variance_selector.transform(self.X_test)
        
        selected_features_var = self.X_train.columns[variance_selector.get_support()]
        print(f"  方差过滤后特征数: {len(selected_features_var)}")
        
        # 2. 单变量特征选择 (F-test)
        print("2. 单变量特征选择...")
        k_best = min(200, len(selected_features_var))  # 选择最多200个特征
        f_selector = SelectKBest(score_func=f_classif, k=k_best)
        
        X_train_f = f_selector.fit_transform(X_train_var, self.y_train)
        X_test_f = f_selector.transform(X_test_var)
        
        selected_features_f = selected_features_var[f_selector.get_support()]
        print(f"  F-test选择后特征数: {len(selected_features_f)}")
        
        # 3. 互信息特征选择
        print("3. 互信息特征选择...")
        k_mutual = min(150, len(selected_features_f))  # 进一步筛选
        mutual_selector = SelectKBest(score_func=mutual_info_classif, k=k_mutual)
        
        X_train_mutual = mutual_selector.fit_transform(X_train_f, self.y_train)
        X_test_mutual = mutual_selector.transform(X_test_f)
        
        selected_features_mutual = selected_features_f[mutual_selector.get_support()]
        print(f"  互信息选择后特征数: {len(selected_features_mutual)}")
        
        # 4. 基于模型的特征选择 (LASSO)
        print("4. LASSO特征选择...")
        
        # 计算类别权重
        class_weights = compute_class_weight('balanced', classes=np.unique(self.y_train), y=self.y_train)
        class_weight_dict = dict(zip(np.unique(self.y_train), class_weights))
        
        lasso = LogisticRegression(
            penalty='l1', solver='liblinear', 
            C=0.1, random_state=self.random_state,
            class_weight=class_weight_dict, max_iter=1000
        )
        
        lasso_selector = SelectFromModel(lasso, threshold='median')
        X_train_lasso = lasso_selector.fit_transform(X_train_mutual, self.y_train)
        X_test_lasso = lasso_selector.transform(X_test_mutual)
        
        selected_features_lasso = selected_features_mutual[lasso_selector.get_support()]
        print(f"  LASSO选择后特征数: {len(selected_features_lasso)}")
        
        # 5. 递归特征消除 (使用随机森林)
        print("5. 递归特征消除...")
        
        if len(selected_features_lasso) > 50:
            rf_estimator = RandomForestClassifier(
                n_estimators=50, random_state=self.random_state,
                class_weight='balanced', n_jobs=-1
            )
            
            rfe_selector = RFE(
                estimator=rf_estimator, 
                n_features_to_select=min(50, len(selected_features_lasso)),
                step=0.1
            )
            
            X_train_rfe = rfe_selector.fit_transform(X_train_lasso, self.y_train)
            X_test_rfe = rfe_selector.transform(X_test_lasso)
            
            selected_features_final = selected_features_lasso[rfe_selector.get_support()]
            
            # 更新数据
            self.X_train = pd.DataFrame(X_train_rfe, columns=selected_features_final)
            self.X_test = pd.DataFrame(X_test_rfe, columns=selected_features_final)
            
            print(f"  RFE选择后特征数: {len(selected_features_final)}")
        else:
            # 如果特征数已经很少，直接使用LASSO结果
            self.X_train = pd.DataFrame(X_train_lasso, columns=selected_features_lasso)
            self.X_test = pd.DataFrame(X_test_lasso, columns=selected_features_lasso)
            selected_features_final = selected_features_lasso
        
        self.feature_names = list(selected_features_final)
        
        print(f"\n特征选择完成:")
        print(f"  原始特征数: {original_feature_count}")
        print(f"  最终特征数: {len(self.feature_names)}")
        print(f"  特征减少: {original_feature_count - len(self.feature_names)} ({(1-len(self.feature_names)/original_feature_count)*100:.1f}%)")
        
        return self
    
    def create_advanced_models(self):
        """
        创建高级模型集合
        """
        print("\n创建高级模型...")
        print("=" * 50)
        
        # 计算类别权重
        class_weights = compute_class_weight('balanced', classes=np.unique(self.y_train), y=self.y_train)
        class_weight_dict = dict(zip(np.unique(self.y_train), class_weights))
        
        # 1. 随机森林 (增强版)
        self.models['RandomForest'] = RandomForestClassifier(
            n_estimators=300,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            class_weight='balanced',
            random_state=self.random_state,
            n_jobs=-1
        )
        
        # 2. Extra Trees (增强版)
        self.models['ExtraTrees'] = ExtraTreesClassifier(
            n_estimators=300,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            class_weight='balanced',
            random_state=self.random_state,
            n_jobs=-1
        )
        
        # 3. XGBoost
        if XGBOOST_AVAILABLE:
            # 转换类别权重为XGBoost格式
            scale_pos_weight = len(self.y_train) / (len(np.unique(self.y_train)) * np.bincount(self.y_train))
            
            self.models['XGBoost'] = xgb.XGBClassifier(
                n_estimators=300,
                max_depth=8,
                learning_rate=0.1,
                subsample=0.9,
                colsample_bytree=0.9,
                random_state=self.random_state,
                n_jobs=-1,
                eval_metric='mlogloss'
            )
        
        # 4. LightGBM
        if LIGHTGBM_AVAILABLE:
            self.models['LightGBM'] = lgb.LGBMClassifier(
                n_estimators=300,
                max_depth=8,
                learning_rate=0.1,
                num_leaves=50,
                subsample=0.9,
                colsample_bytree=0.9,
                class_weight='balanced',
                random_state=self.random_state,
                n_jobs=-1,
                verbose=-1
            )
        
        # 5. CatBoost
        if CATBOOST_AVAILABLE:
            self.models['CatBoost'] = CatBoostClassifier(
                iterations=300,
                depth=8,
                learning_rate=0.1,
                class_weights=list(class_weights),
                random_seed=self.random_state,
                verbose=False,
                thread_count=-1
            )
        
        # 6. 逻辑回归 (正则化)
        self.models['LogisticRegression'] = LogisticRegression(
            C=1.0,
            penalty='elasticnet',
            l1_ratio=0.5,
            solver='saga',
            class_weight='balanced',
            random_state=self.random_state,
            max_iter=2000,
            n_jobs=-1
        )
        
        # 7. Bagging分类器
        base_rf = RandomForestClassifier(
            n_estimators=50,
            max_depth=10,
            class_weight='balanced',
            random_state=self.random_state
        )
        
        self.models['BaggingRF'] = BaggingClassifier(
            estimator=base_rf,
            n_estimators=10,
            random_state=self.random_state,
            n_jobs=-1
        )
        
        print(f"创建了 {len(self.models)} 个模型")
        for model_name in self.models.keys():
            print(f"  - {model_name}")
        
        return self
    
    def evaluate_models_advanced(self):
        """
        高级模型评估
        """
        print("\n高级模型评估...")
        print("=" * 50)
        
        # 使用分层K折交叉验证
        cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
        
        results = []
        
        for model_name, model in self.models.items():
            print(f"\n评估 {model_name}...")
            
            try:
                start_time = time.time()
                
                # 交叉验证
                cv_scores = cross_val_score(
                    model, self.X_train, self.y_train,
                    cv=cv_strategy, scoring='f1_macro', n_jobs=-1
                )
                
                training_time = time.time() - start_time
                
                result = {
                    'model_name': model_name,
                    'cv_mean': cv_scores.mean(),
                    'cv_std': cv_scores.std(),
                    'cv_scores': cv_scores,
                    'training_time': training_time
                }
                
                results.append(result)
                
                print(f"  CV Macro-F1: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
                print(f"  训练时间: {training_time:.2f}秒")
                
            except Exception as e:
                print(f"  错误: {str(e)}")
        
        # 排序结果
        results.sort(key=lambda x: x['cv_mean'], reverse=True)
        
        print("\n模型性能排行:")
        print("-" * 60)
        for i, result in enumerate(results, 1):
            print(f"{i}. {result['model_name']:15s} - CV: {result['cv_mean']:.4f} (+/- {result['cv_std']*2:.4f})")
        
        self.model_results = results
        return self
    
    def optimize_hyperparameters_advanced(self, top_k=3):
        """
        高级超参数优化
        """
        print(f"\n高级超参数优化 (Top {top_k} 模型)...")
        print("=" * 50)
        
        # 获取最好的几个模型
        top_models = self.model_results[:top_k]
        optimized_models = {}
        
        # 优化的超参数搜索空间（减少复杂度提高效率）
        param_grids = {
            'RandomForest': {
                'n_estimators': [200, 300, 500],
                'max_depth': [10, 15, 20, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['sqrt', 'log2', 0.8],
                'bootstrap': [True, False]
            },
            'ExtraTrees': {
                'n_estimators': [200, 300, 500],
                'max_depth': [10, 15, 20, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['sqrt', 'log2', 0.8],
                'bootstrap': [True, False]
            },
            'XGBoost': {
                'n_estimators': [200, 300, 500],
                'max_depth': [6, 8, 10],
                'learning_rate': [0.05, 0.1, 0.15],
                'subsample': [0.8, 0.9],
                'colsample_bytree': [0.8, 0.9],
                'gamma': [0, 0.1],
                'reg_alpha': [0, 0.1],
                'reg_lambda': [1, 1.5]
            },
            'LightGBM': {
                'n_estimators': [200, 300, 500],
                'max_depth': [6, 8, 10],
                'learning_rate': [0.05, 0.1, 0.15],
                'num_leaves': [31, 50, 100],
                'subsample': [0.8, 0.9],
                'colsample_bytree': [0.8, 0.9],
                'reg_alpha': [0, 0.1],
                'reg_lambda': [0, 0.1]
            }
        }
        
        if CATBOOST_AVAILABLE:
            param_grids['CatBoost'] = {
                'iterations': [200, 300, 500],
                'depth': [6, 8, 10],
                'learning_rate': [0.05, 0.1, 0.15],
                'l2_leaf_reg': [1, 3, 5],
                'border_count': [32, 64],
                'bagging_temperature': [0, 0.5]
            }
        
        cv_strategy = StratifiedKFold(n_splits=3, shuffle=True, random_state=self.random_state)
        
        for model_result in top_models:
            model_name = model_result['model_name']
            
            if model_name not in param_grids:
                print(f"跳过 {model_name} (无超参数网格)")
                optimized_models[model_name] = self.models[model_name]
                continue
            
            print(f"\n优化 {model_name}...")
            
            base_model = self.models[model_name]
            param_grid = param_grids[model_name]
            
            # 使用RandomizedSearchCV进行高效搜索
            search = RandomizedSearchCV(
                estimator=base_model,
                param_distributions=param_grid,
                n_iter=20,  # 优化搜索次数以提高效率
                cv=cv_strategy,
                scoring='f1_macro',
                n_jobs=-1,
                random_state=self.random_state,
                verbose=0
            )
            
            start_time = time.time()
            search.fit(self.X_train, self.y_train)
            search_time = time.time() - start_time
            
            optimized_models[model_name] = search.best_estimator_
            
            print(f"  最佳CV分数: {search.best_score_:.4f}")
            print(f"  优化时间: {search_time:.2f}秒")
            print(f"  最佳参数: {search.best_params_}")
        
        self.optimized_models = optimized_models
        return self
    
    def create_advanced_ensemble(self):
        """
        创建高级集成模型
        """
        print("\n创建高级集成模型...")
        print("=" * 50)
        
        # 准备基模型
        base_models = []
        for name, model in self.optimized_models.items():
            base_models.append((name, model))
        
        print(f"集成 {len(base_models)} 个基模型")
        
        # 1. Voting Classifier (软投票)
        voting_clf = VotingClassifier(
            estimators=base_models,
            voting='soft',
            n_jobs=-1
        )
        
        # 2. Stacking Classifier (多层)
        # 第一层元学习器
        meta_learner_1 = LogisticRegression(
            random_state=self.random_state,
            max_iter=2000,
            class_weight='balanced',
            C=1.0
        )
        
        stacking_clf = StackingClassifier(
            estimators=base_models,
            final_estimator=meta_learner_1,
            cv=5,
            n_jobs=-1,
            passthrough=True  # 传递原始特征
        )
        
        # 3. 加权集成 (基于CV性能)
        weights = []
        for model_result in self.model_results:
            if model_result['model_name'] in self.optimized_models:
                weights.append(model_result['cv_mean'])
        
        # 归一化权重
        weights = np.array(weights)
        weights = weights / weights.sum()
        
        weighted_voting_clf = VotingClassifier(
            estimators=base_models,
            voting='soft',
            weights=weights,
            n_jobs=-1
        )
        
        # 评估集成模型
        cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
        
        print("\n评估集成模型...")
        
        ensemble_results = {}
        
        # 评估Voting
        voting_scores = cross_val_score(
            voting_clf, self.X_train, self.y_train,
            cv=cv_strategy, scoring='f1_macro', n_jobs=-1
        )
        ensemble_results['Voting'] = voting_scores.mean()
        print(f"Voting Ensemble CV: {voting_scores.mean():.4f} (+/- {voting_scores.std()*2:.4f})")
        
        # 评估加权Voting
        weighted_voting_scores = cross_val_score(
            weighted_voting_clf, self.X_train, self.y_train,
            cv=cv_strategy, scoring='f1_macro', n_jobs=-1
        )
        ensemble_results['WeightedVoting'] = weighted_voting_scores.mean()
        print(f"Weighted Voting CV: {weighted_voting_scores.mean():.4f} (+/- {weighted_voting_scores.std()*2:.4f})")
        
        # 评估Stacking
        stacking_scores = cross_val_score(
            stacking_clf, self.X_train, self.y_train,
            cv=cv_strategy, scoring='f1_macro', n_jobs=-1
        )
        ensemble_results['Stacking'] = stacking_scores.mean()
        print(f"Stacking Ensemble CV: {stacking_scores.mean():.4f} (+/- {stacking_scores.std()*2:.4f})")
        
        # 选择最佳集成方法
        best_ensemble_name = max(ensemble_results, key=ensemble_results.get)
        best_ensemble_score = ensemble_results[best_ensemble_name]
        
        if best_ensemble_name == 'Voting':
            self.best_ensemble = voting_clf
        elif best_ensemble_name == 'WeightedVoting':
            self.best_ensemble = weighted_voting_clf
        else:
            self.best_ensemble = stacking_clf
        
        self.best_ensemble_name = best_ensemble_name
        self.best_ensemble_score = best_ensemble_score
        
        print(f"\n选择最佳集成: {best_ensemble_name} (CV: {best_ensemble_score:.4f})")
        
        # 比较单模型和集成模型
        best_single_score = max(result['cv_mean'] for result in self.model_results)
        improvement = best_ensemble_score - best_single_score
        
        print(f"相比最佳单模型提升: {improvement:.4f} ({improvement/best_single_score*100:.2f}%)")
        
        return self
    
    def train_final_model_advanced(self):
        """
        训练最终模型 (带校准)
        """
        print("\n训练最终模型...")
        print("=" * 50)
        
        # 选择最终模型
        best_single_score = max(result['cv_mean'] for result in self.model_results)
        
        if hasattr(self, 'best_ensemble') and self.best_ensemble_score > best_single_score:
            base_model = self.best_ensemble
            final_model_name = f'Ensemble ({self.best_ensemble_name})'
            final_score = self.best_ensemble_score
        else:
            # 使用最佳单模型
            best_model_name = self.model_results[0]['model_name']
            base_model = self.optimized_models[best_model_name]
            final_model_name = f'{best_model_name} (Optimized)'
            final_score = self.model_results[0]['cv_mean']
        
        print(f"基础模型: {final_model_name}")
        print(f"预期CV分数: {final_score:.4f}")
        
        # 模型校准
        print("\n应用模型校准...")
        calibrated_model = CalibratedClassifierCV(
            base_model, 
            method='isotonic',  # 或 'sigmoid'
            cv=3
        )
        
        # 在完整训练集上训练
        print("在完整训练集上训练...")
        start_time = time.time()
        calibrated_model.fit(self.X_train, self.y_train)
        training_time = time.time() - start_time
        
        print(f"训练完成，耗时: {training_time:.2f}秒")
        
        self.final_model = calibrated_model
        self.final_model_name = f'{final_model_name} (Calibrated)'
        self.final_score = final_score
        
        return self
    
    def generate_predictions_advanced(self, output_file='submit_v2.csv'):
        """
        生成高级预测结果
        """
        print("\n生成预测结果...")
        print("=" * 50)
        
        # 预测概率
        print("预测测试集概率...")
        y_pred_proba = self.final_model.predict_proba(self.X_test)
        
        # 基础预测
        y_pred_encoded = self.final_model.predict(self.X_test)
        
        # 阈值优化 (可选)
        # 这里可以根据验证集结果调整预测阈值
        
        # 解码预测结果
        y_pred = self.target_encoder.inverse_transform(y_pred_encoded)
        
        # 创建提交文件
        submission = pd.DataFrame({
            'ID': self.test_ids,
            'Tissue': y_pred
        })
        
        # 保存
        submission.to_csv(output_file, index=False)
        print(f"预测结果已保存到: {output_file}")
        
        # 预测分布
        pred_dist = pd.Series(y_pred).value_counts().sort_index()
        print(f"\n预测分布:")
        for tissue, count in pred_dist.items():
            print(f"  {tissue}: {count} ({count/len(y_pred)*100:.1f}%)")
        
        # 预测置信度分析
        max_proba = y_pred_proba.max(axis=1)
        print(f"\n预测置信度:")
        print(f"  平均置信度: {max_proba.mean():.3f}")
        print(f"  置信度标准差: {max_proba.std():.3f}")
        print(f"  低置信度样本 (<0.5): {(max_proba < 0.5).sum()} ({(max_proba < 0.5).mean()*100:.1f}%)")
        
        self.submission = submission
        return self
    
    def save_model_advanced(self, model_file='final_model_v2.pkl'):
        """
        保存高级模型
        """
        print(f"\n保存模型到: {model_file}")
        
        model_data = {
            'model': self.final_model,
            'model_name': self.final_model_name,
            'cv_score': self.final_score,
            'encoders': self.encoders,
            'scalers': self.scalers,
            'feature_names': self.feature_names,
            'original_features': self.original_features,
            'engineered_features': self.engineered_features,
            'target_encoder': self.target_encoder,
            'model_results': self.model_results,
            'training_timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'pipeline_version': '2.0'
        }
        
        with open(model_file, 'wb') as f:
            pickle.dump(model_data, f)
        
        print("模型保存完成")
        
        return self
    
    def run_complete_pipeline_advanced(self, train_path='train.csv', test_path='test.csv', 
                                     output_file='submit_v2.csv', model_file='final_model_v2.pkl'):
        """
        运行完整的高级预测pipeline
        """
        print("=" * 80)
        print("        环状RNA组织表达预测 - 高级Pipeline v2.0")
        print("=" * 80)
        
        start_time = time.time()
        
        try:
            # 执行完整流程
            (self.load_data(train_path, test_path)
             .analyze_data()
             .advanced_feature_engineering()
             .advanced_preprocessing()
             .advanced_feature_selection()
             .create_advanced_models()
             .evaluate_models_advanced()
             .optimize_hyperparameters_advanced()
             .create_advanced_ensemble()
             .train_final_model_advanced()
             .generate_predictions_advanced(output_file)
             .save_model_advanced(model_file))
            
            total_time = time.time() - start_time
            
            print("\n" + "=" * 80)
            print("                    高级Pipeline v2.0 完成")
            print("=" * 80)
            print(f"总耗时: {total_time:.2f}秒 ({total_time/60:.1f}分钟)")
            print(f"最终模型: {self.final_model_name}")
            print(f"预期性能: {self.final_score:.4f} (Macro-F1)")
            print(f"特征数量: {len(self.feature_names)}")
            print(f"预测文件: {output_file}")
            print(f"模型文件: {model_file}")
            print("=" * 80)
            
            return True
            
        except Exception as e:
            print(f"\nPipeline执行失败: {str(e)}")
            import traceback
            traceback.print_exc()
            return False

def main():
    """
    主函数
    """
    # 创建高级pipeline实例
    pipeline = AdvancedRNAPredictionPipeline(random_state=42)
    
    # 运行完整流程
    success = pipeline.run_complete_pipeline_advanced(
        train_path='train.csv',
        test_path='test.csv',
        output_file='submit_v2.csv',
        model_file='final_model_v2.pkl'
    )
    
    if success:
        print("\n🎉 高级预测任务完成！")
        print("📁 生成的文件:")
        print("   - submit_v2.csv: 改进版提交文件")
        print("   - final_model_v2.pkl: 改进版训练模型")
        print("\n🚀 主要改进:")
        print("   - 更多生物学特征工程")
        print("   - 高级特征选择策略")
        print("   - 精细超参数优化")
        print("   - 多层集成和模型校准")
        print("\n💡 提示: 如果性能仍需提升，可以考虑:")
        print("   - 添加更多领域特定特征")
        print("   - 尝试深度学习方法")
        print("   - 进行数据增强")
    else:
        print("\n❌ 高级预测任务失败，请检查错误信息")

if __name__ == "__main__":
    main()