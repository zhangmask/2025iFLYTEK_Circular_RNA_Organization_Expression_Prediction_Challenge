#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
环状RNA组织表达预测 Pipeline v4
基于原始高分模型(0.36)的优化版本

主要改进:
1. 保留原始模型的成功架构
2. 优化特征选择策略和阈值
3. 精细化超参数网格
4. 改进集成权重策略
5. 增强数据预处理
"""

import pandas as pd
import numpy as np
import warnings
from datetime import datetime
import time
import pickle
from collections import Counter

# 机器学习库
from sklearn.model_selection import (
    StratifiedKFold, cross_val_score, RandomizedSearchCV
)
from sklearn.preprocessing import LabelEncoder, RobustScaler
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.ensemble import (
    RandomForestClassifier, ExtraTreesClassifier, VotingClassifier, StackingClassifier
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

# XGBoost和LightGBM
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

# CatBoost
try:
    from catboost import CatBoostClassifier
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    print("警告: CatBoost未安装")

warnings.filterwarnings('ignore')

class RNAPredictionPipelineV4:
    """
    环状RNA组织表达预测Pipeline v4
    基于原始高分模型的优化版本
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
        self.feature_names = None
        
        # 模型相关
        self.models = {}
        self.model_results = []
        self.optimized_models = {}
        self.final_model = None
        self.final_model_name = None
        self.final_score = None
        
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
        
        return self
    
    def analyze_data(self):
        """
        数据分析
        """
        print("\n数据分析...")
        print("=" * 50)
        
        # 基本信息
        print("训练集基本信息:")
        print(f"  样本数: {len(self.train_data)}")
        print(f"  特征数: {self.train_data.shape[1] - 2}")  # 减去ID和Tissue
        
        # 目标变量分布
        tissue_dist = self.train_data['Tissue'].value_counts().sort_index()
        print(f"\n目标变量分布:")
        for tissue, count in tissue_dist.items():
            print(f"  {tissue}: {count} ({count/len(self.train_data)*100:.1f}%)")
        
        # 缺失值检查
        train_missing = self.train_data.isnull().sum().sum()
        test_missing = self.test_data.isnull().sum().sum()
        print(f"\n缺失值:")
        print(f"  训练集: {train_missing}")
        print(f"  测试集: {test_missing}")
        
        return self
    
    def feature_engineering(self):
        """
        特征工程 - 保留原始模型的成功策略
        """
        print("\n特征工程...")
        print("=" * 50)
        
        # 提取特征和目标变量
        feature_cols = [col for col in self.train_data.columns if col not in ['ID', 'Tissue']]
        
        X_train = self.train_data[feature_cols].copy()
        y_train = self.train_data['Tissue'].copy()
        X_test = self.test_data[feature_cols].copy()
        
        print(f"原始特征数: {len(feature_cols)}")
        
        # 1. 处理分类特征编码
        print("处理分类特征...")
        categorical_features = ['Strand', 'Circtype', 'has_N']
        
        for col in categorical_features:
            if col in feature_cols:
                le = LabelEncoder()
                # 合并训练和测试数据进行编码，确保一致性
                combined_data = pd.concat([X_train[col], X_test[col]], axis=0)
                le.fit(combined_data)
                X_train[col] = le.transform(X_train[col])
                X_test[col] = le.transform(X_test[col])
                self.encoders[col] = le
        
        # 2. 处理缺失值 - 使用中位数填充（保留原始策略）
        print("处理缺失值...")
        for col in feature_cols:
            if X_train[col].isnull().sum() > 0:
                median_val = X_train[col].median()
                X_train[col].fillna(median_val, inplace=True)
                X_test[col].fillna(median_val, inplace=True)
        
        # 2. 目标变量编码
        print("编码目标变量...")
        self.target_encoder = LabelEncoder()
        y_train_encoded = self.target_encoder.fit_transform(y_train)
        
        # 3. 特征标准化 - 使用RobustScaler（保留原始策略）
        print("标准化特征...")
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # 转换为DataFrame保持列名
        X_train_scaled = pd.DataFrame(X_train_scaled, columns=feature_cols)
        X_test_scaled = pd.DataFrame(X_test_scaled, columns=feature_cols)
        
        # 4. 特征选择 - 双重策略（保留原始策略但优化参数）
        print("特征选择...")
        
        # 基于F统计量的特征选择
        k_features_f = min(150, len(feature_cols))  # 稍微增加特征数
        selector_f = SelectKBest(score_func=f_classif, k=k_features_f)
        X_train_f = selector_f.fit_transform(X_train_scaled, y_train_encoded)
        selected_features_f = X_train_scaled.columns[selector_f.get_support()].tolist()
        
        # 基于互信息的特征选择
        k_features_mi = min(150, len(feature_cols))  # 稍微增加特征数
        selector_mi = SelectKBest(score_func=mutual_info_classif, k=k_features_mi)
        X_train_mi = selector_mi.fit_transform(X_train_scaled, y_train_encoded)
        selected_features_mi = X_train_scaled.columns[selector_mi.get_support()].tolist()
        
        # 合并特征选择结果
        selected_features = list(set(selected_features_f + selected_features_mi))
        
        # 应用特征选择
        X_train_final = X_train_scaled[selected_features]
        X_test_final = X_test_scaled[selected_features]
        
        print(f"F统计量选择特征数: {len(selected_features_f)}")
        print(f"互信息选择特征数: {len(selected_features_mi)}")
        print(f"最终特征数: {len(selected_features)}")
        
        # 保存结果
        self.X_train = X_train_final.values
        self.y_train = y_train_encoded
        self.X_test = X_test_final.values
        self.feature_names = selected_features
        self.scalers['robust'] = scaler
        self.feature_selector = {'f_selector': selector_f, 'mi_selector': selector_mi}
        
        return self
    
    def create_models(self):
        """
        创建基础模型 - 保留原始模型组合
        """
        print("\n创建基础模型...")
        print("=" * 50)
        
        # 1. Random Forest
        self.models['RandomForest'] = RandomForestClassifier(
            n_estimators=100,
            random_state=self.random_state,
            n_jobs=-1,
            class_weight='balanced'
        )
        
        # 2. Extra Trees
        self.models['ExtraTrees'] = ExtraTreesClassifier(
            n_estimators=100,
            random_state=self.random_state,
            n_jobs=-1,
            class_weight='balanced'
        )
        
        # 3. XGBoost
        if XGBOOST_AVAILABLE:
            self.models['XGBoost'] = xgb.XGBClassifier(
                random_state=self.random_state,
                n_jobs=-1,
                eval_metric='mlogloss'
            )
        
        # 4. LightGBM
        if LIGHTGBM_AVAILABLE:
            self.models['LightGBM'] = lgb.LGBMClassifier(
                random_state=self.random_state,
                n_jobs=-1,
                verbose=-1
            )
        
        # 5. CatBoost
        if CATBOOST_AVAILABLE:
            self.models['CatBoost'] = CatBoostClassifier(
                random_state=self.random_state,
                verbose=False
            )
        
        # 6. Logistic Regression
        self.models['LogisticRegression'] = LogisticRegression(
            random_state=self.random_state,
            max_iter=1000,
            class_weight='balanced'
        )
        
        print(f"创建了 {len(self.models)} 个基础模型")
        for name in self.models.keys():
            print(f"  - {name}")
        
        return self
    
    def evaluate_models(self):
        """
        评估基础模型
        """
        print("\n评估基础模型...")
        print("=" * 50)
        
        cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
        
        results = []
        
        for name, model in self.models.items():
            print(f"\n评估 {name}...")
            
            start_time = time.time()
            cv_scores = cross_val_score(
                model, self.X_train, self.y_train,
                cv=cv_strategy, scoring='f1_macro', n_jobs=-1
            )
            eval_time = time.time() - start_time
            
            result = {
                'model_name': name,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'cv_scores': cv_scores,
                'eval_time': eval_time
            }
            
            results.append(result)
            
            print(f"  CV分数: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")
            print(f"  评估时间: {eval_time:.2f}秒")
        
        # 按性能排序
        results.sort(key=lambda x: x['cv_mean'], reverse=True)
        self.model_results = results
        
        print("\n模型性能排序:")
        for i, result in enumerate(results, 1):
            print(f"  {i}. {result['model_name']}: {result['cv_mean']:.4f}")
        
        return self
    
    def optimize_best_models(self, top_k=4):
        """
        优化表现最佳的K个模型 - 精细化超参数网格
        """
        print(f"\n优化Top {top_k}模型...")
        print("=" * 50)
        
        top_models = self.model_results[:top_k]
        optimized_models = {}
        
        # 精细化的超参数网格
        param_grids = {
            'RandomForest': {
                'n_estimators': [100, 200, 300, 500],
                'max_depth': [6, 8, 10, 12, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['sqrt', 'log2', 0.8]
            },
            'ExtraTrees': {
                'n_estimators': [100, 200, 300, 500],
                'max_depth': [6, 8, 10, 12, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['sqrt', 'log2', 0.8]
            },
            'LogisticRegression': {
                'C': [0.01, 0.1, 1, 10, 100],
                'penalty': ['l1', 'l2', 'elasticnet'],
                'solver': ['liblinear', 'saga'],
                'l1_ratio': [0.1, 0.5, 0.7, 0.9]
            }
        }
        
        # XGBoost参数
        if XGBOOST_AVAILABLE:
            param_grids['XGBoost'] = {
                'n_estimators': [100, 200, 300, 500],
                'max_depth': [4, 6, 8, 10],
                'learning_rate': [0.01, 0.05, 0.1, 0.15, 0.2],
                'subsample': [0.7, 0.8, 0.9, 1.0],
                'colsample_bytree': [0.7, 0.8, 0.9, 1.0],
                'reg_alpha': [0, 0.1, 0.5, 1],
                'reg_lambda': [0, 0.1, 0.5, 1]
            }
        
        # LightGBM参数
        if LIGHTGBM_AVAILABLE:
            param_grids['LightGBM'] = {
                'n_estimators': [100, 200, 300, 500],
                'max_depth': [4, 6, 8, 10],
                'learning_rate': [0.01, 0.05, 0.1, 0.15, 0.2],
                'num_leaves': [20, 31, 50, 100],
                'subsample': [0.7, 0.8, 0.9, 1.0],
                'colsample_bytree': [0.7, 0.8, 0.9, 1.0],
                'reg_alpha': [0, 0.1, 0.5, 1],
                'reg_lambda': [0, 0.1, 0.5, 1]
            }
        
        # CatBoost参数
        if CATBOOST_AVAILABLE:
            param_grids['CatBoost'] = {
                'iterations': [100, 200, 300, 500],
                'depth': [4, 6, 8, 10],
                'learning_rate': [0.01, 0.05, 0.1, 0.15, 0.2],
                'l2_leaf_reg': [1, 3, 5, 7, 9],
                'border_count': [32, 64, 128, 255]
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
            
            # 使用RandomizedSearchCV，增加搜索次数
            search = RandomizedSearchCV(
                estimator=base_model,
                param_distributions=param_grid,
                n_iter=50,  # 增加搜索次数
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
    
    def create_ensemble(self):
        """
        创建集成模型 - 改进权重策略
        """
        print("\n创建集成模型...")
        print("=" * 50)
        
        # 准备基模型
        base_models = []
        model_weights = []
        
        # 根据CV性能计算权重
        for name, model in self.optimized_models.items():
            base_models.append((name, model))
            # 找到对应的CV分数
            cv_score = 0.0
            for result in self.model_results:
                if result['model_name'] == name:
                    cv_score = result['cv_mean']
                    break
            model_weights.append(cv_score)
        
        # 归一化权重
        total_weight = sum(model_weights)
        model_weights = [w/total_weight for w in model_weights]
        
        print(f"集成 {len(base_models)} 个基模型")
        for i, (name, _) in enumerate(base_models):
            print(f"  {name}: 权重 {model_weights[i]:.3f}")
        
        # 1. 加权软投票
        voting_clf = VotingClassifier(
            estimators=base_models,
            voting='soft',
            weights=model_weights,  # 使用计算的权重
            n_jobs=-1
        )
        
        # 2. Stacking Classifier
        # 使用更强的元学习器
        meta_learner = LogisticRegression(
            random_state=self.random_state,
            max_iter=2000,
            class_weight='balanced',
            C=1.0
        )
        
        stacking_clf = StackingClassifier(
            estimators=base_models,
            final_estimator=meta_learner,
            cv=5,  # 增加CV折数
            n_jobs=-1
        )
        
        # 评估集成模型
        cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
        
        print("\n评估集成模型...")
        
        # 评估加权Voting
        voting_scores = cross_val_score(
            voting_clf, self.X_train, self.y_train,
            cv=cv_strategy, scoring='f1_macro', n_jobs=-1
        )
        
        print(f"加权Voting CV: {voting_scores.mean():.4f} (+/- {voting_scores.std()*2:.4f})")
        
        # 评估Stacking
        stacking_scores = cross_val_score(
            stacking_clf, self.X_train, self.y_train,
            cv=cv_strategy, scoring='f1_macro', n_jobs=-1
        )
        
        print(f"Stacking CV: {stacking_scores.mean():.4f} (+/- {stacking_scores.std()*2:.4f})")
        
        # 选择最佳集成方法
        if stacking_scores.mean() > voting_scores.mean():
            self.best_ensemble = stacking_clf
            self.best_ensemble_name = 'Stacking'
            self.best_ensemble_score = stacking_scores.mean()
        else:
            self.best_ensemble = voting_clf
            self.best_ensemble_name = 'Weighted Voting'
            self.best_ensemble_score = voting_scores.mean()
        
        print(f"\n选择最佳集成: {self.best_ensemble_name} (CV: {self.best_ensemble_score:.4f})")
        
        # 比较单模型和集成模型
        best_single_score = max(result['cv_mean'] for result in self.model_results)
        improvement = self.best_ensemble_score - best_single_score
        
        print(f"相比最佳单模型提升: {improvement:.4f} ({improvement/best_single_score*100:.2f}%)")
        
        return self
    
    def train_final_model(self):
        """
        训练最终模型
        """
        print("\n训练最终模型...")
        print("=" * 50)
        
        # 选择最终模型
        best_single_score = max(result['cv_mean'] for result in self.model_results)
        
        if hasattr(self, 'best_ensemble') and self.best_ensemble_score > best_single_score:
            final_model = self.best_ensemble
            final_model_name = f'Ensemble ({self.best_ensemble_name})'
            final_score = self.best_ensemble_score
        else:
            # 使用最佳单模型
            best_model_name = self.model_results[0]['model_name']
            final_model = self.optimized_models[best_model_name]
            final_model_name = f'{best_model_name} (Optimized)'
            final_score = self.model_results[0]['cv_mean']
        
        print(f"最终模型: {final_model_name}")
        print(f"预期CV分数: {final_score:.4f}")
        
        # 在完整训练集上训练
        print("在完整训练集上训练...")
        start_time = time.time()
        final_model.fit(self.X_train, self.y_train)
        training_time = time.time() - start_time
        
        print(f"训练完成，耗时: {training_time:.2f}秒")
        
        self.final_model = final_model
        self.final_model_name = final_model_name
        self.final_score = final_score
        
        return self
    
    def generate_predictions(self, output_file='submit_v4.csv'):
        """
        生成测试集预测结果
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
    
    def save_model(self, model_file='final_model_v4.pkl'):
        """
        保存训练好的模型和预处理器
        """
        print(f"\n保存模型到: {model_file}")
        
        model_data = {
            'model': self.final_model,
            'model_name': self.final_model_name,
            'cv_score': self.final_score,
            'scalers': self.scalers,
            'feature_selector': self.feature_selector,
            'feature_names': self.feature_names,
            'target_encoder': self.target_encoder,
            'training_timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'version': 'v4'
        }
        
        with open(model_file, 'wb') as f:
            pickle.dump(model_data, f)
        
        print("模型保存完成")
        
        return self
    
    def run_complete_pipeline(self, train_path='train.csv', test_path='test.csv', 
                            output_file='submit_v4.csv', model_file='final_model_v4.pkl'):
        """
        运行完整的预测pipeline
        """
        print("=" * 80)
        print("           环状RNA组织表达预测 - Pipeline v4")
        print("=" * 80)
        
        start_time = time.time()
        
        try:
            # 执行完整流程
            (self.load_data(train_path, test_path)
             .analyze_data()
             .feature_engineering()
             .create_models()
             .evaluate_models()
             .optimize_best_models()
             .create_ensemble()
             .train_final_model()
             .generate_predictions(output_file)
             .save_model(model_file))
            
            total_time = time.time() - start_time
            
            print("\n" + "=" * 80)
            print("                        Pipeline v4 完成")
            print("=" * 80)
            print(f"总耗时: {total_time:.2f}秒 ({total_time/60:.1f}分钟)")
            print(f"最终模型: {self.final_model_name}")
            print(f"预期性能: {self.final_score:.4f} (Macro-F1)")
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
    # 创建pipeline实例
    pipeline = RNAPredictionPipelineV4(random_state=42)
    
    # 运行完整流程
    success = pipeline.run_complete_pipeline(
        train_path='train.csv',
        test_path='test.csv',
        output_file='submit_v4.csv',
        model_file='final_model_v4.pkl'
    )
    
    if success:
        print("\n🎉 预测任务完成！")
        print("📁 生成的文件:")
        print("   - submit_v4.csv: 提交文件")
        print("   - final_model_v4.pkl: 训练好的模型")
        print("\n💡 v4版本改进:")
        print("   - 保留原始模型成功架构")
        print("   - 优化特征选择策略")
        print("   - 精细化超参数网格")
        print("   - 改进集成权重策略")
    else:
        print("\n❌ 预测任务失败，请检查错误信息")

if __name__ == "__main__":
    main()