#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
环状RNA组织表达预测挑战赛 - 完整解决方案
目标：获得尽可能高的Macro-F1分数

作者：AI Assistant
日期：2025年
"""

import pandas as pd
import numpy as np
import warnings
import pickle
import time
from datetime import datetime
import os
from typing import Dict, List, Tuple, Any

# 机器学习库
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, RobustScaler
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score, classification_report, confusion_matrix
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline

# 梯度提升库
import xgboost as xgb
import lightgbm as lgb
try:
    import catboost as cb
    CATBOOST_AVAILABLE = True
except ImportError:
    print("CatBoost not available, will skip CatBoost models")
    CATBOOST_AVAILABLE = False

# 超参数优化
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from scipy.stats import randint, uniform

# 模型集成
from sklearn.ensemble import StackingClassifier

# 设置
warnings.filterwarnings('ignore')
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

class RNAPredictionPipeline:
    """
    环状RNA组织表达预测完整pipeline
    """
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.encoders = {}
        self.scalers = {}
        self.feature_selector = None
        self.models = {}
        self.best_model = None
        self.feature_names = []
        self.target_encoder = None
        
    def load_data(self, train_path='train.csv', test_path='test.csv'):
        """
        加载训练和测试数据
        """
        print("加载数据...")
        self.train_df = pd.read_csv(train_path)
        self.test_df = pd.read_csv(test_path)
        
        print(f"训练集形状: {self.train_df.shape}")
        print(f"测试集形状: {self.test_df.shape}")
        
        # 保存测试集ID
        self.test_ids = self.test_df['ID'].copy()
        
        return self
    
    def analyze_data(self):
        """
        数据分析和质量检查
        """
        print("\n数据分析...")
        print("=" * 50)
        
        # 基本信息
        print("训练集基本信息:")
        print(self.train_df.info())
        
        # 缺失值检查
        missing_train = self.train_df.isnull().sum()
        missing_test = self.test_df.isnull().sum()
        
        if missing_train.sum() > 0:
            print(f"\n训练集缺失值:\n{missing_train[missing_train > 0]}")
        else:
            print("\n训练集无缺失值")
            
        if missing_test.sum() > 0:
            print(f"\n测试集缺失值:\n{missing_test[missing_test > 0]}")
        else:
            print("测试集无缺失值")
        
        # 目标变量分布
        target_dist = self.train_df['Tissue'].value_counts().sort_index()
        print(f"\n目标变量分布:")
        for tissue, count in target_dist.items():
            print(f"  {tissue}: {count} ({count/len(self.train_df)*100:.1f}%)")
        
        # 特征类型分析
        numeric_features = self.train_df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_features = self.train_df.select_dtypes(include=['object']).columns.tolist()
        
        # 移除ID和目标变量
        if 'ID' in numeric_features:
            numeric_features.remove('ID')
        if 'Tissue' in categorical_features:
            categorical_features.remove('Tissue')
            
        print(f"\n数值特征 ({len(numeric_features)}): {numeric_features}")
        print(f"分类特征 ({len(categorical_features)}): {categorical_features}")
        
        self.numeric_features = numeric_features
        self.categorical_features = categorical_features
        
        return self
    
    def feature_engineering(self):
        """
        特征工程和数据预处理
        """
        print("\n特征工程...")
        print("=" * 50)
        
        # 复制数据
        train_processed = self.train_df.copy()
        test_processed = self.test_df.copy()
        
        # 1. 处理分类特征
        for feature in self.categorical_features:
            if feature in train_processed.columns:
                # 使用LabelEncoder
                le = LabelEncoder()
                
                # 合并训练和测试数据的唯一值
                all_values = pd.concat([
                    train_processed[feature], 
                    test_processed[feature]
                ]).unique()
                
                le.fit(all_values)
                
                train_processed[feature] = le.transform(train_processed[feature])
                test_processed[feature] = le.transform(test_processed[feature])
                
                self.encoders[feature] = le
                print(f"编码特征 {feature}: {len(le.classes_)} 个类别")
        
        # 2. 创建新特征
        print("\n创建新特征...")
        
        # 核苷酸比例相关特征
        for df in [train_processed, test_processed]:
            # GC含量相关
            df['GC_AT_ratio'] = df['GC_content'] / (df['A_ratio'] + df['T_ratio'] + 1e-8)
            df['GC_content_squared'] = df['GC_content'] ** 2
            
            # 长度相关
            df['length_log'] = np.log1p(df['length'])
            df['length_sqrt'] = np.sqrt(df['length'])
            
            # 能量相关
            df['energy_per_length'] = df['Average free energy'] / (df['length'] + 1e-8)
            df['energy_squared'] = df['Average free energy'] ** 2
            
            # miRNA结合相关
            df['mirna_per_length'] = df['miRNA Binding count'] / (df['length'] + 1e-8)
            df['mirna_log'] = np.log1p(df['miRNA Binding count'])
            
            # 核苷酸比例交互
            df['AT_content'] = df['A_ratio'] + df['T_ratio']
            df['purine_content'] = df['A_ratio'] + df['G_ratio']  # 嘌呤
            df['pyrimidine_content'] = df['T_ratio'] + df['C_ratio']  # 嘧啶
            df['purine_pyrimidine_ratio'] = df['purine_content'] / (df['pyrimidine_content'] + 1e-8)
            
            # 复合特征
            df['complexity_score'] = (df['GC_content'] * df['length'] * df['miRNA Binding count']) / 1000
            df['stability_score'] = df['Average free energy'] * df['GC_content']
            
        print(f"创建了 {len([c for c in train_processed.columns if c not in self.train_df.columns])} 个新特征")
        
        # 3. 准备特征和目标
        # 移除ID和目标变量
        feature_columns = [col for col in train_processed.columns if col not in ['ID', 'Tissue']]
        
        X_train = train_processed[feature_columns]
        y_train = train_processed['Tissue']
        X_test = test_processed[feature_columns]
        
        # 4. 处理缺失值
        print("\n处理缺失值...")
        from sklearn.impute import SimpleImputer
        
        # 检查缺失值
        missing_train = X_train.isnull().sum()
        missing_test = X_test.isnull().sum()
        
        if missing_train.sum() > 0 or missing_test.sum() > 0:
            print(f"训练集缺失值: {missing_train.sum()}")
            print(f"测试集缺失值: {missing_test.sum()}")
            
            # 使用中位数填充数值特征
            imputer = SimpleImputer(strategy='median')
            X_train_imputed = imputer.fit_transform(X_train)
            X_test_imputed = imputer.transform(X_test)
            
            # 转换回DataFrame
            X_train = pd.DataFrame(X_train_imputed, columns=feature_columns)
            X_test = pd.DataFrame(X_test_imputed, columns=feature_columns)
            
            self.scalers['imputer'] = imputer
            print("缺失值处理完成")
        else:
            print("无缺失值")
        
        # 5. 目标变量编码
        self.target_encoder = LabelEncoder()
        y_train_encoded = self.target_encoder.fit_transform(y_train)
        
        print(f"目标类别: {self.target_encoder.classes_}")
        
        # 6. 特征标准化
        print("\n特征标准化...")
        scaler = RobustScaler()  # 对异常值更鲁棒
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # 转换回DataFrame
        X_train_scaled = pd.DataFrame(X_train_scaled, columns=feature_columns)
        X_test_scaled = pd.DataFrame(X_test_scaled, columns=feature_columns)
        
        self.scalers['features'] = scaler
        
        # 7. 特征选择
        print("\n特征选择...")
        
        # 使用多种特征选择方法
        # 方法1: 基于F统计量
        selector_f = SelectKBest(score_func=f_classif, k=min(50, len(feature_columns)))
        X_train_selected_f = selector_f.fit_transform(X_train_scaled, y_train_encoded)
        
        # 方法2: 基于互信息
        selector_mi = SelectKBest(score_func=mutual_info_classif, k=min(50, len(feature_columns)))
        X_train_selected_mi = selector_mi.fit_transform(X_train_scaled, y_train_encoded)
        
        # 合并选择的特征
        selected_features_f = set(np.array(feature_columns)[selector_f.get_support()])
        selected_features_mi = set(np.array(feature_columns)[selector_mi.get_support()])
        selected_features = list(selected_features_f.union(selected_features_mi))
        
        print(f"F统计量选择: {len(selected_features_f)} 个特征")
        print(f"互信息选择: {len(selected_features_mi)} 个特征")
        print(f"合并后: {len(selected_features)} 个特征")
        
        # 应用特征选择
        X_train_final = X_train_scaled[selected_features]
        X_test_final = X_test_scaled[selected_features]
        
        self.feature_names = selected_features
        self.feature_selector = {'f_selector': selector_f, 'mi_selector': selector_mi}
        
        # 保存处理后的数据
        self.X_train = X_train_final
        self.y_train = y_train_encoded
        self.X_test = X_test_final
        
        print(f"\n最终特征维度: {self.X_train.shape}")
        print(f"测试集特征维度: {self.X_test.shape}")
        
        return self
    
    def create_models(self):
        """
        创建多种机器学习模型
        """
        print("\n创建模型...")
        print("=" * 50)
        
        models = {}
        
        # 1. Random Forest
        models['RandomForest'] = RandomForestClassifier(
            n_estimators=300,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            random_state=self.random_state,
            n_jobs=-1,
            class_weight='balanced'
        )
        
        # 2. Extra Trees
        models['ExtraTrees'] = ExtraTreesClassifier(
            n_estimators=300,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            random_state=self.random_state,
            n_jobs=-1,
            class_weight='balanced'
        )
        
        # 3. XGBoost
        models['XGBoost'] = xgb.XGBClassifier(
            n_estimators=300,
            max_depth=8,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=self.random_state,
            eval_metric='mlogloss',
            verbosity=0,
            n_jobs=-1
        )
        
        # 4. LightGBM
        models['LightGBM'] = lgb.LGBMClassifier(
            n_estimators=300,
            max_depth=8,
            learning_rate=0.1,
            num_leaves=50,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=self.random_state,
            verbosity=-1,
            n_jobs=-1,
            class_weight='balanced'
        )
        
        # 5. CatBoost (如果可用)
        if CATBOOST_AVAILABLE:
            models['CatBoost'] = cb.CatBoostClassifier(
                iterations=300,
                depth=8,
                learning_rate=0.1,
                random_state=self.random_state,
                verbose=False,
                thread_count=-1,
                class_weights='Balanced'
            )
        
        # 6. Logistic Regression (作为基线)
        models['LogisticRegression'] = LogisticRegression(
            random_state=self.random_state,
            max_iter=1000,
            class_weight='balanced',
            n_jobs=-1
        )
        
        self.models = models
        print(f"创建了 {len(models)} 个模型")
        
        return self
    
    def evaluate_models(self, cv_folds=5):
        """
        使用交叉验证评估所有模型
        """
        print("\n模型评估...")
        print("=" * 50)
        
        results = []
        cv_strategy = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
        
        for name, model in self.models.items():
            print(f"\n评估 {name}...")
            start_time = time.time()
            
            try:
                # 交叉验证
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
    
    def optimize_best_models(self, top_k=3):
        """
        对表现最好的几个模型进行超参数优化
        """
        print(f"\n超参数优化 (Top {top_k} 模型)...")
        print("=" * 50)
        
        # 获取最好的几个模型
        top_models = self.model_results[:top_k]
        optimized_models = {}
        
        # 定义超参数搜索空间
        param_grids = {
            'RandomForest': {
                'n_estimators': [200, 300, 500],
                'max_depth': [10, 15, 20, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['sqrt', 'log2', 0.8]
            },
            'ExtraTrees': {
                'n_estimators': [200, 300, 500],
                'max_depth': [10, 15, 20, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['sqrt', 'log2', 0.8]
            },
            'XGBoost': {
                'n_estimators': [200, 300, 500],
                'max_depth': [6, 8, 10],
                'learning_rate': [0.05, 0.1, 0.15],
                'subsample': [0.8, 0.9, 1.0],
                'colsample_bytree': [0.8, 0.9, 1.0]
            },
            'LightGBM': {
                'n_estimators': [200, 300, 500],
                'max_depth': [6, 8, 10],
                'learning_rate': [0.05, 0.1, 0.15],
                'num_leaves': [31, 50, 100],
                'subsample': [0.8, 0.9, 1.0]
            }
        }
        
        if CATBOOST_AVAILABLE:
            param_grids['CatBoost'] = {
                'iterations': [200, 300, 500],
                'depth': [6, 8, 10],
                'learning_rate': [0.05, 0.1, 0.15],
                'l2_leaf_reg': [1, 3, 5]
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
            
            # 使用RandomizedSearchCV
            search = RandomizedSearchCV(
                estimator=base_model,
                param_distributions=param_grid,
                n_iter=30,  # 减少搜索次数以节省时间
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
        创建集成模型
        """
        print("\n创建集成模型...")
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
        
        # 2. Stacking Classifier
        # 使用逻辑回归作为元学习器
        meta_learner = LogisticRegression(
            random_state=self.random_state,
            max_iter=1000,
            class_weight='balanced'
        )
        
        stacking_clf = StackingClassifier(
            estimators=base_models,
            final_estimator=meta_learner,
            cv=3,
            n_jobs=-1
        )
        
        # 评估集成模型
        cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
        
        print("\n评估集成模型...")
        
        # 评估Voting
        voting_scores = cross_val_score(
            voting_clf, self.X_train, self.y_train,
            cv=cv_strategy, scoring='f1_macro', n_jobs=-1
        )
        
        print(f"Voting Ensemble CV: {voting_scores.mean():.4f} (+/- {voting_scores.std()*2:.4f})")
        
        # 评估Stacking
        stacking_scores = cross_val_score(
            stacking_clf, self.X_train, self.y_train,
            cv=cv_strategy, scoring='f1_macro', n_jobs=-1
        )
        
        print(f"Stacking Ensemble CV: {stacking_scores.mean():.4f} (+/- {stacking_scores.std()*2:.4f})")
        
        # 选择最佳集成方法
        if stacking_scores.mean() > voting_scores.mean():
            self.best_ensemble = stacking_clf
            self.best_ensemble_name = 'Stacking'
            self.best_ensemble_score = stacking_scores.mean()
        else:
            self.best_ensemble = voting_clf
            self.best_ensemble_name = 'Voting'
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
    
    def generate_predictions(self, output_file='submit.csv'):
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
    
    def save_model(self, model_file='final_model.pkl'):
        """
        保存训练好的模型和预处理器
        """
        print(f"\n保存模型到: {model_file}")
        
        model_data = {
            'model': self.final_model,
            'model_name': self.final_model_name,
            'cv_score': self.final_score,
            'encoders': self.encoders,
            'scalers': self.scalers,
            'feature_selector': self.feature_selector,
            'feature_names': self.feature_names,
            'target_encoder': self.target_encoder,
            'training_timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        with open(model_file, 'wb') as f:
            pickle.dump(model_data, f)
        
        print("模型保存完成")
        
        return self
    
    def run_complete_pipeline(self, train_path='train.csv', test_path='test.csv', 
                            output_file='submit.csv', model_file='final_model.pkl'):
        """
        运行完整的预测pipeline
        """
        print("=" * 80)
        print("           环状RNA组织表达预测 - 完整Pipeline")
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
            print("                        Pipeline 完成")
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
    pipeline = RNAPredictionPipeline(random_state=42)
    
    # 运行完整流程
    success = pipeline.run_complete_pipeline(
        train_path='train.csv',
        test_path='test.csv',
        output_file='submit.csv',
        model_file='final_model.pkl'
    )
    
    if success:
        print("\n🎉 预测任务完成！")
        print("📁 生成的文件:")
        print("   - submit.csv: 提交文件")
        print("   - final_model.pkl: 训练好的模型")
        print("\n💡 提示: 可以尝试调整超参数或添加更多特征来进一步提升性能")
    else:
        print("\n❌ 预测任务失败，请检查错误信息")

if __name__ == "__main__":
    main()