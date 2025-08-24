#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
环状RNA组织表达预测 - 改进版本v3
基于原始高分模型(0.36)的另一种优化策略

改进策略:
1. 生物学意义的特征组合
2. 保守但稳定的特征选择
3. 贝叶斯优化超参数搜索
4. 伪标签技术和数据增强
5. 精细的类别平衡处理
6. 优化的模型权重和集成策略
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# 机器学习库
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, RobustScaler
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, classification_report
from sklearn.utils.class_weight import compute_class_weight
from sklearn.calibration import CalibratedClassifierCV

# XGBoost和LightGBM
import xgboost as xgb
import lightgbm as lgb

# CatBoost (可选)
try:
    import catboost as cb
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    print("CatBoost未安装，将跳过CatBoost模型")

# 贝叶斯优化
try:
    from skopt import BayesSearchCV
    from skopt.space import Real, Integer, Categorical
    BAYESIAN_OPT_AVAILABLE = True
except ImportError:
    from sklearn.model_selection import RandomizedSearchCV
    BAYESIAN_OPT_AVAILABLE = False
    print("scikit-optimize未安装，使用RandomizedSearchCV")

# 其他工具
import time
import pickle
from datetime import datetime
from collections import Counter
import math

class RNAPredictionPipelineV3:
    """
    环状RNA组织表达预测Pipeline - 改进版本v3
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
        self.target_encoder = None
        
        # 模型相关
        self.models = {}
        self.optimized_models = {}
        self.model_results = []
        self.final_model = None
        
        print("RNAPredictionPipelineV3 初始化完成")
        print(f"随机种子: {random_state}")
        print(f"贝叶斯优化可用: {BAYESIAN_OPT_AVAILABLE}")
        print(f"CatBoost可用: {CATBOOST_AVAILABLE}")
    
    def load_data(self, train_path='train.csv', test_path='test.csv'):
        """
        加载训练和测试数据
        """
        print("\n加载数据...")
        print("=" * 50)
        
        # 加载数据
        self.train_data = pd.read_csv(train_path)
        self.test_data = pd.read_csv(test_path)
        
        print(f"训练集形状: {self.train_data.shape}")
        print(f"测试集形状: {self.test_data.shape}")
        
        # 保存测试集ID
        self.test_ids = self.test_data['ID'].values
        
        # 检查数据质量
        print(f"\n训练集缺失值: {self.train_data.isnull().sum().sum()}")
        print(f"测试集缺失值: {self.test_data.isnull().sum().sum()}")
        
        # 目标变量分布
        target_dist = self.train_data['Tissue'].value_counts()
        print(f"\n目标变量分布:")
        for tissue, count in target_dist.items():
            print(f"  {tissue}: {count} ({count/len(self.train_data)*100:.1f}%)")
        
        return self
    
    def analyze_data(self):
        """
        数据分析和探索
        """
        print("\n数据分析...")
        print("=" * 50)
        
        # 基本统计信息
        numeric_cols = self.train_data.select_dtypes(include=[np.number]).columns
        print(f"数值特征数量: {len(numeric_cols)}")
        
        categorical_cols = self.train_data.select_dtypes(include=['object']).columns
        categorical_cols = [col for col in categorical_cols if col not in ['ID', 'Tissue']]
        print(f"分类特征数量: {len(categorical_cols)}")
        
        # 特征统计
        print("\n数值特征统计:")
        for col in numeric_cols[:5]:  # 显示前5个
            data = self.train_data[col]
            print(f"  {col}: 均值={data.mean():.3f}, 标准差={data.std():.3f}, 范围=[{data.min():.3f}, {data.max():.3f}]")
        
        if len(categorical_cols) > 0:
            print("\n分类特征统计:")
            for col in categorical_cols[:3]:  # 显示前3个
                unique_count = self.train_data[col].nunique()
                print(f"  {col}: {unique_count}个唯一值")
        
        return self
    
    def create_biological_features(self, data):
        """
        创建生物学意义的特征组合
        """
        print("创建生物学特征...")
        
        # 基础特征
        gc_content = data.get('GC_Content', 0)
        length = data.get('Length', 1)
        energy = data.get('Energy', 0)
        mirna_binding = data.get('miRNA_Binding_Sites', 0)
        
        # 生物学特征组合
        features = {}
        
        # 1. GC含量相关特征
        features['GC_Length_Interaction'] = gc_content * np.log1p(length)
        features['GC_Energy_Ratio'] = gc_content / (abs(energy) + 1e-6)
        features['GC_Stability'] = gc_content * (1 / (abs(energy) + 1e-6))
        
        # 2. 长度相关特征
        features['Length_Log'] = np.log1p(length)
        features['Length_Sqrt'] = np.sqrt(length)
        features['Length_Energy_Density'] = abs(energy) / (length + 1e-6)
        
        # 3. 能量相关特征
        features['Energy_Abs'] = abs(energy)
        features['Energy_Normalized'] = energy / (length + 1e-6)
        features['Energy_Stability_Score'] = -energy / (gc_content * length + 1e-6)
        
        # 4. miRNA结合相关特征
        features['miRNA_Density'] = mirna_binding / (length + 1e-6)
        features['miRNA_GC_Interaction'] = mirna_binding * gc_content
        features['miRNA_Energy_Ratio'] = mirna_binding / (abs(energy) + 1e-6)
        
        # 5. 复合生物学特征
        features['Structural_Complexity'] = (gc_content * length) / (abs(energy) + 1e-6)
        features['Binding_Efficiency'] = (mirna_binding * gc_content) / (length + 1e-6)
        features['Thermodynamic_Score'] = (gc_content * length) / (abs(energy) + mirna_binding + 1e-6)
        
        return features
    
    def feature_engineering(self):
        """
        特征工程
        """
        print("\n特征工程...")
        print("=" * 50)
        
        # 合并训练和测试数据进行一致的特征工程
        train_features = self.train_data.drop(['ID', 'Tissue'], axis=1)
        test_features = self.test_data.drop(['ID'], axis=1)
        
        # 确保列顺序一致
        common_cols = list(set(train_features.columns) & set(test_features.columns))
        train_features = train_features[common_cols]
        test_features = test_features[common_cols]
        
        all_features = pd.concat([train_features, test_features], axis=0, ignore_index=True)
        
        print(f"原始特征数量: {len(common_cols)}")
        
        # 创建生物学特征
        bio_features_list = []
        for idx in range(len(all_features)):
            row_data = all_features.iloc[idx].to_dict()
            bio_features = self.create_biological_features(row_data)
            bio_features_list.append(bio_features)
        
        bio_features_df = pd.DataFrame(bio_features_list)
        print(f"生物学特征数量: {len(bio_features_df.columns)}")
        
        # 合并所有特征
        all_features_combined = pd.concat([all_features, bio_features_df], axis=1)
        
        # 处理缺失值
        numeric_cols = all_features_combined.select_dtypes(include=[np.number]).columns
        all_features_combined[numeric_cols] = all_features_combined[numeric_cols].fillna(
            all_features_combined[numeric_cols].median()
        )
        
        # 处理分类特征
        categorical_cols = all_features_combined.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if col not in self.encoders:
                self.encoders[col] = LabelEncoder()
                all_features_combined[col] = all_features_combined[col].fillna('Unknown')
                self.encoders[col].fit(all_features_combined[col])
            
            all_features_combined[col] = all_features_combined[col].fillna('Unknown')
            all_features_combined[col] = self.encoders[col].transform(all_features_combined[col])
        
        # 分离训练和测试集
        n_train = len(self.train_data)
        self.X_train = all_features_combined.iloc[:n_train].copy()
        self.X_test = all_features_combined.iloc[n_train:].copy()
        
        # 处理目标变量
        self.target_encoder = LabelEncoder()
        self.y_train = self.target_encoder.fit_transform(self.train_data['Tissue'])
        
        print(f"最终特征数量: {self.X_train.shape[1]}")
        print(f"训练集形状: {self.X_train.shape}")
        print(f"测试集形状: {self.X_test.shape}")
        
        # 保存特征名称
        self.feature_names = list(self.X_train.columns)
        
        return self
    
    def conservative_feature_selection(self):
        """
        保守但稳定的特征选择
        """
        print("\n保守特征选择...")
        print("=" * 50)
        
        # 1. 移除低方差特征
        from sklearn.feature_selection import VarianceThreshold
        var_selector = VarianceThreshold(threshold=0.01)
        X_var = var_selector.fit_transform(self.X_train)
        selected_features = np.array(self.feature_names)[var_selector.get_support()]
        
        print(f"方差筛选后特征数量: {len(selected_features)}")
        
        # 2. 相关性筛选
        X_var_df = pd.DataFrame(X_var, columns=selected_features)
        corr_matrix = X_var_df.corr().abs()
        
        # 移除高相关性特征
        upper_tri = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        
        to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > 0.95)]
        X_corr = X_var_df.drop(columns=to_drop)
        
        print(f"相关性筛选后特征数量: {X_corr.shape[1]}")
        
        # 3. 统计显著性筛选
        k_best = min(200, X_corr.shape[1])  # 保守选择
        selector = SelectKBest(score_func=f_classif, k=k_best)
        X_selected = selector.fit_transform(X_corr, self.y_train)
        
        final_features = X_corr.columns[selector.get_support()]
        print(f"最终选择特征数量: {len(final_features)}")
        
        # 更新数据
        self.X_train = pd.DataFrame(X_selected, columns=final_features)
        
        # 对测试集应用相同的变换
        X_test_var = var_selector.transform(self.X_test)
        X_test_var_df = pd.DataFrame(X_test_var, columns=selected_features)
        X_test_corr = X_test_var_df.drop(columns=to_drop)
        X_test_selected = selector.transform(X_test_corr)
        self.X_test = pd.DataFrame(X_test_selected, columns=final_features)
        
        # 保存选择器
        self.feature_selector = {
            'var_selector': var_selector,
            'corr_features': list(X_corr.columns),
            'kbest_selector': selector,
            'final_features': list(final_features)
        }
        
        return self
    
    def create_models(self):
        """
        创建基础模型
        """
        print("\n创建模型...")
        print("=" * 50)
        
        # 计算类别权重
        class_weights = compute_class_weight(
            'balanced', 
            classes=np.unique(self.y_train), 
            y=self.y_train
        )
        class_weight_dict = dict(zip(np.unique(self.y_train), class_weights))
        
        print(f"类别权重: {class_weight_dict}")
        
        # 基础模型配置
        self.models = {
            'RandomForest': RandomForestClassifier(
                n_estimators=200,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                max_features='sqrt',
                class_weight='balanced',
                random_state=self.random_state,
                n_jobs=-1
            ),
            
            'ExtraTrees': ExtraTreesClassifier(
                n_estimators=200,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                max_features='sqrt',
                class_weight='balanced',
                random_state=self.random_state,
                n_jobs=-1
            ),
            
            'XGBoost': xgb.XGBClassifier(
                n_estimators=200,
                max_depth=8,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=self.random_state,
                n_jobs=-1,
                eval_metric='mlogloss'
            ),
            
            'LightGBM': lgb.LGBMClassifier(
                n_estimators=200,
                max_depth=8,
                learning_rate=0.1,
                num_leaves=50,
                subsample=0.8,
                colsample_bytree=0.8,
                class_weight='balanced',
                random_state=self.random_state,
                n_jobs=-1,
                verbose=-1
            )
        }
        
        if CATBOOST_AVAILABLE:
            self.models['CatBoost'] = cb.CatBoostClassifier(
                iterations=200,
                depth=8,
                learning_rate=0.1,
                class_weights=list(class_weights),
                random_seed=self.random_state,
                verbose=False
            )
        
        print(f"创建了 {len(self.models)} 个基础模型")
        
        return self
    
    def evaluate_models(self):
        """
        评估基础模型性能
        """
        print("\n评估基础模型...")
        print("=" * 50)
        
        cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
        results = []
        
        for name, model in self.models.items():
            print(f"\n评估 {name}...")
            
            try:
                start_time = time.time()
                cv_scores = cross_val_score(
                    model, self.X_train, self.y_train,
                    cv=cv_strategy, scoring='f1_macro', n_jobs=-1
                )
                training_time = time.time() - start_time
                
                result = {
                    'model_name': name,
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
    
    def bayesian_optimization(self, top_k=3):
        """
        贝叶斯优化超参数
        """
        print(f"\n贝叶斯优化 (Top {top_k} 模型)...")
        print("=" * 50)
        
        top_models = self.model_results[:top_k]
        optimized_models = {}
        
        # 定义搜索空间
        if BAYESIAN_OPT_AVAILABLE:
            search_spaces = {
                'RandomForest': {
                    'n_estimators': Integer(100, 500),
                    'max_depth': Integer(10, 30),
                    'min_samples_split': Integer(2, 20),
                    'min_samples_leaf': Integer(1, 10),
                    'max_features': Categorical(['sqrt', 'log2', 0.8])
                },
                'LightGBM': {
                    'n_estimators': Integer(100, 500),
                    'max_depth': Integer(6, 15),
                    'learning_rate': Real(0.01, 0.3, prior='log-uniform'),
                    'num_leaves': Integer(20, 100),
                    'subsample': Real(0.6, 1.0)
                },
                'XGBoost': {
                    'n_estimators': Integer(100, 500),
                    'max_depth': Integer(6, 15),
                    'learning_rate': Real(0.01, 0.3, prior='log-uniform'),
                    'subsample': Real(0.6, 1.0),
                    'colsample_bytree': Real(0.6, 1.0)
                }
            }
        else:
            search_spaces = {
                'RandomForest': {
                    'n_estimators': [100, 200, 300, 400, 500],
                    'max_depth': [10, 15, 20, 25, 30],
                    'min_samples_split': [2, 5, 10, 15, 20],
                    'min_samples_leaf': [1, 2, 5, 10],
                    'max_features': ['sqrt', 'log2', 0.8]
                },
                'LightGBM': {
                    'n_estimators': [100, 200, 300, 400, 500],
                    'max_depth': [6, 8, 10, 12, 15],
                    'learning_rate': [0.01, 0.05, 0.1, 0.2, 0.3],
                    'num_leaves': [20, 40, 60, 80, 100],
                    'subsample': [0.6, 0.7, 0.8, 0.9, 1.0]
                },
                'XGBoost': {
                    'n_estimators': [100, 200, 300, 400, 500],
                    'max_depth': [6, 8, 10, 12, 15],
                    'learning_rate': [0.01, 0.05, 0.1, 0.2, 0.3],
                    'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
                    'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0]
                }
            }
        
        cv_strategy = StratifiedKFold(n_splits=3, shuffle=True, random_state=self.random_state)
        
        for model_result in top_models:
            model_name = model_result['model_name']
            
            if model_name not in search_spaces:
                print(f"跳过 {model_name} (无搜索空间定义)")
                optimized_models[model_name] = self.models[model_name]
                continue
            
            print(f"\n优化 {model_name}...")
            
            base_model = self.models[model_name]
            search_space = search_spaces[model_name]
            
            if BAYESIAN_OPT_AVAILABLE:
                search = BayesSearchCV(
                    estimator=base_model,
                    search_spaces=search_space,
                    n_iter=20,
                    cv=cv_strategy,
                    scoring='f1_macro',
                    n_jobs=-1,
                    random_state=self.random_state
                )
            else:
                # 使用预定义的搜索空间（已经是list格式）
                param_dist = search_space
                
                search = RandomizedSearchCV(
                    estimator=base_model,
                    param_distributions=param_dist,
                    n_iter=20,
                    cv=cv_strategy,
                    scoring='f1_macro',
                    n_jobs=-1,
                    random_state=self.random_state
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
    
    def create_ensemble_with_weights(self):
        """
        创建加权集成模型
        """
        print("\n创建加权集成...")
        print("=" * 50)
        
        # 基于CV性能计算权重
        model_weights = {}
        total_score = 0
        
        for result in self.model_results:
            if result['model_name'] in self.optimized_models:
                model_weights[result['model_name']] = result['cv_mean']
                total_score += result['cv_mean']
        
        # 归一化权重
        for name in model_weights:
            model_weights[name] = model_weights[name] / total_score
        
        print(f"模型权重: {model_weights}")
        
        # 准备基模型
        base_models = []
        for name, model in self.optimized_models.items():
            # 使用校准的分类器
            calibrated_model = CalibratedClassifierCV(model, cv=3, method='isotonic')
            base_models.append((name, calibrated_model))
        
        # 创建加权投票分类器
        weights = [model_weights.get(name, 1.0) for name, _ in base_models]
        
        self.ensemble_model = VotingClassifier(
            estimators=base_models,
            voting='soft',
            weights=weights,
            n_jobs=-1
        )
        
        # 评估集成模型
        cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
        ensemble_scores = cross_val_score(
            self.ensemble_model, self.X_train, self.y_train,
            cv=cv_strategy, scoring='f1_macro', n_jobs=-1
        )
        
        self.ensemble_score = ensemble_scores.mean()
        
        print(f"\n加权集成CV分数: {ensemble_scores.mean():.4f} (+/- {ensemble_scores.std()*2:.4f})")
        
        # 比较最佳单模型
        best_single_score = max(result['cv_mean'] for result in self.model_results)
        improvement = self.ensemble_score - best_single_score
        
        print(f"相比最佳单模型提升: {improvement:.4f} ({improvement/best_single_score*100:.2f}%)")
        
        return self
    
    def pseudo_labeling(self, confidence_threshold=0.9):
        """
        伪标签技术
        """
        print(f"\n伪标签技术 (置信度阈值: {confidence_threshold})...")
        print("=" * 50)
        
        # 训练初始模型
        best_model_name = self.model_results[0]['model_name']
        initial_model = self.optimized_models[best_model_name]
        initial_model.fit(self.X_train, self.y_train)
        
        # 预测测试集概率
        test_probs = initial_model.predict_proba(self.X_test)
        max_probs = np.max(test_probs, axis=1)
        
        # 选择高置信度样本
        high_conf_mask = max_probs >= confidence_threshold
        high_conf_indices = np.where(high_conf_mask)[0]
        
        if len(high_conf_indices) > 0:
            # 获取伪标签
            pseudo_labels = initial_model.predict(self.X_test[high_conf_mask])
            
            # 扩展训练集
            X_extended = pd.concat([
                self.X_train,
                self.X_test.iloc[high_conf_indices]
            ], axis=0, ignore_index=True)
            
            y_extended = np.concatenate([
                self.y_train,
                pseudo_labels
            ])
            
            print(f"添加了 {len(high_conf_indices)} 个伪标签样本")
            print(f"扩展后训练集大小: {len(X_extended)}")
            
            # 重新评估模型
            cv_strategy = StratifiedKFold(n_splits=3, shuffle=True, random_state=self.random_state)
            extended_scores = cross_val_score(
                initial_model, X_extended, y_extended,
                cv=cv_strategy, scoring='f1_macro', n_jobs=-1
            )
            
            print(f"伪标签后CV分数: {extended_scores.mean():.4f} (+/- {extended_scores.std()*2:.4f})")
            
            # 如果性能提升，使用扩展数据集
            original_score = self.model_results[0]['cv_mean']
            if extended_scores.mean() > original_score:
                self.X_train_extended = X_extended
                self.y_train_extended = y_extended
                self.use_pseudo_labels = True
                print("伪标签提升了性能，将使用扩展数据集")
            else:
                self.use_pseudo_labels = False
                print("伪标签未提升性能，使用原始数据集")
        else:
            print("没有找到高置信度的伪标签样本")
            self.use_pseudo_labels = False
        
        return self
    
    def train_final_model(self):
        """
        训练最终模型
        """
        print("\n训练最终模型...")
        print("=" * 50)
        
        # 选择训练数据
        if hasattr(self, 'use_pseudo_labels') and self.use_pseudo_labels:
            X_final = self.X_train_extended
            y_final = self.y_train_extended
            print("使用伪标签扩展的训练集")
        else:
            X_final = self.X_train
            y_final = self.y_train
            print("使用原始训练集")
        
        # 选择最终模型
        if hasattr(self, 'ensemble_model') and self.ensemble_score > self.model_results[0]['cv_mean']:
            final_model = self.ensemble_model
            final_model_name = 'Weighted Ensemble'
            final_score = self.ensemble_score
        else:
            best_model_name = self.model_results[0]['model_name']
            final_model = self.optimized_models[best_model_name]
            final_model_name = f'{best_model_name} (Optimized)'
            final_score = self.model_results[0]['cv_mean']
        
        print(f"最终模型: {final_model_name}")
        print(f"预期CV分数: {final_score:.4f}")
        
        # 训练最终模型
        print("在完整训练集上训练...")
        start_time = time.time()
        final_model.fit(X_final, y_final)
        training_time = time.time() - start_time
        
        print(f"训练完成，耗时: {training_time:.2f}秒")
        
        self.final_model = final_model
        self.final_model_name = final_model_name
        self.final_score = final_score
        
        return self
    
    def generate_predictions(self, output_file='submit_v3.csv'):
        """
        生成预测结果
        """
        print("\n生成预测结果...")
        print("=" * 50)
        
        # 预测
        print("预测测试集...")
        y_pred_encoded = self.final_model.predict(self.X_test)
        
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
        
        self.submission = submission
        return self
    
    def run_complete_pipeline(self, train_path='train.csv', test_path='test.csv', 
                            output_file='submit_v3.csv'):
        """
        运行完整的预测pipeline
        """
        print("=" * 80)
        print("           环状RNA组织表达预测 - Pipeline V3")
        print("=" * 80)
        
        start_time = time.time()
        
        try:
            # 执行完整流程
            (self.load_data(train_path, test_path)
             .analyze_data()
             .feature_engineering()
             .conservative_feature_selection()
             .create_models()
             .evaluate_models()
             .bayesian_optimization()
             .create_ensemble_with_weights()
             .pseudo_labeling()
             .train_final_model()
             .generate_predictions(output_file))
            
            total_time = time.time() - start_time
            
            print("\n" + "=" * 80)
            print("                        Pipeline V3 完成")
            print("=" * 80)
            print(f"总耗时: {total_time:.2f}秒 ({total_time/60:.1f}分钟)")
            print(f"最终模型: {self.final_model_name}")
            print(f"预期性能: {self.final_score:.4f} (Macro-F1)")
            print(f"预测文件: {output_file}")
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
    # 创建pipeline实例
    pipeline = RNAPredictionPipelineV3(random_state=42)
    
    # 运行完整流程
    success = pipeline.run_complete_pipeline(
        train_path='train.csv',
        test_path='test.csv',
        output_file='submit_v3.csv'
    )
    
    if success:
        print("\n🎉 预测任务完成！")
        print("📁 生成的文件:")
        print("   - submit_v3.csv: 提交文件")
        print("\n💡 改进策略:")
        print("   - 生物学特征组合")
        print("   - 保守特征选择")
        print("   - 贝叶斯优化")
        print("   - 伪标签技术")
        print("   - 加权集成")
    else:
        print("\n❌ 预测任务失败，请检查错误信息")

if __name__ == "__main__":
    main()