#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
高级特征工程脚本 - 环状RNA组织表达预测
目标：通过创建高质量特征提升Macro-F1分数
当前基线：0.36485
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, PolynomialFeatures
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif, RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LassoCV
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import VotingClassifier, StackingClassifier
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
import warnings
warnings.filterwarnings('ignore')

class AdvancedFeatureEngineer:
    def __init__(self):
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.feature_names = []
        
    def create_biological_features(self, df):
        """创建生物学相关的特征组合"""
        df_new = df.copy()
        
        # 1. GC含量相关特征
        df_new['GC_length_interaction'] = df_new['GC_content'] * df_new['length']
        df_new['GC_content_squared'] = df_new['GC_content'] ** 2
        df_new['GC_content_log'] = np.log1p(df_new['GC_content'])
        
        # 2. 碱基比例交互特征
        df_new['AT_ratio'] = df_new['A_ratio'] + df_new['T_ratio']
        df_new['GC_ratio'] = df_new['G_ratio'] + df_new['C_ratio']
        df_new['AT_GC_ratio'] = df_new['AT_ratio'] / (df_new['GC_ratio'] + 1e-8)
        df_new['purine_ratio'] = df_new['A_ratio'] + df_new['G_ratio']  # 嘌呤
        df_new['pyrimidine_ratio'] = df_new['T_ratio'] + df_new['C_ratio']  # 嘧啶
        df_new['purine_pyrimidine_ratio'] = df_new['purine_ratio'] / (df_new['pyrimidine_ratio'] + 1e-8)
        
        # 3. 碱基比例复杂组合
        df_new['base_diversity'] = -(df_new['A_ratio'] * np.log(df_new['A_ratio'] + 1e-8) +
                                    df_new['T_ratio'] * np.log(df_new['T_ratio'] + 1e-8) +
                                    df_new['G_ratio'] * np.log(df_new['G_ratio'] + 1e-8) +
                                    df_new['C_ratio'] * np.log(df_new['C_ratio'] + 1e-8))  # Shannon熵
        
        # 4. miRNA结合相关特征
        df_new['miRNA_binding_density'] = df_new['miRNA Binding count'] / (df_new['length'] + 1)
        df_new['miRNA_energy_interaction'] = df_new['miRNA Binding count'] * df_new['Average free energy']
        df_new['energy_per_length'] = df_new['Average free energy'] / (df_new['length'] + 1)
        df_new['miRNA_binding_log'] = np.log1p(df_new['miRNA Binding count'])
        
        # 5. 长度相关特征
        df_new['length_log'] = np.log1p(df_new['length'])
        df_new['length_squared'] = df_new['length'] ** 2
        df_new['length_sqrt'] = np.sqrt(df_new['length'])
        
        # 6. 自由能相关特征
        df_new['free_energy_abs'] = np.abs(df_new['Average free energy'])
        df_new['free_energy_squared'] = df_new['Average free energy'] ** 2
        df_new['free_energy_log'] = np.log1p(np.abs(df_new['Average free energy']))
        
        return df_new
    
    def create_statistical_features(self, df):
        """创建统计特征"""
        df_new = df.copy()
        
        # 数值特征列表
        numeric_cols = ['miRNA Binding count', 'Average free energy', 'length', 
                       'GC_content', 'A_ratio', 'T_ratio', 'G_ratio', 'C_ratio']
        
        # 1. 特征比值和差值
        for i, col1 in enumerate(numeric_cols):
            for col2 in numeric_cols[i+1:]:
                df_new[f'{col1}_{col2}_ratio'] = df_new[col1] / (df_new[col2] + 1e-8)
                df_new[f'{col1}_{col2}_diff'] = df_new[col1] - df_new[col2]
                df_new[f'{col1}_{col2}_product'] = df_new[col1] * df_new[col2]
        
        # 2. 多项式特征（选择重要特征）
        important_cols = ['GC_content', 'length', 'miRNA Binding count', 'Average free energy']
        for col in important_cols:
            df_new[f'{col}_poly2'] = df_new[col] ** 2
            df_new[f'{col}_poly3'] = df_new[col] ** 3
            df_new[f'{col}_sqrt'] = np.sqrt(np.abs(df_new[col]))
        
        # 3. 对数变换
        for col in numeric_cols:
            if (df_new[col] >= 0).all():
                df_new[f'{col}_log1p'] = np.log1p(df_new[col])
            else:
                df_new[f'{col}_log_abs'] = np.log1p(np.abs(df_new[col]))
        
        # 4. 标准化后的组合特征
        scaler_temp = StandardScaler()
        scaled_features = scaler_temp.fit_transform(df_new[numeric_cols])
        scaled_df = pd.DataFrame(scaled_features, columns=[f'{col}_scaled' for col in numeric_cols])
        
        # 标准化特征的组合
        scaled_df['scaled_sum'] = scaled_df.sum(axis=1)
        scaled_df['scaled_mean'] = scaled_df.mean(axis=1)
        scaled_df['scaled_std'] = scaled_df.std(axis=1)
        scaled_df['scaled_max'] = scaled_df.max(axis=1)
        scaled_df['scaled_min'] = scaled_df.min(axis=1)
        
        df_new = pd.concat([df_new, scaled_df], axis=1)
        
        return df_new
    
    def create_sequence_complexity_features(self, df):
        """创建序列复杂度相关特征"""
        df_new = df.copy()
        
        # 1. 基于自由能的复杂度
        df_new['energy_stability'] = -df_new['Average free energy']  # 负值表示更稳定
        df_new['energy_normalized'] = df_new['Average free energy'] / (df_new['length'] + 1)
        
        # 2. miRNA结合复杂度
        df_new['binding_efficiency'] = df_new['miRNA Binding count'] / (np.abs(df_new['Average free energy']) + 1)
        df_new['binding_strength'] = df_new['miRNA Binding count'] * np.abs(df_new['Average free energy'])
        
        # 3. 序列组成复杂度
        df_new['composition_balance'] = 1 - np.abs(df_new['AT_ratio'] - df_new['GC_ratio'])
        df_new['base_uniformity'] = 1 - np.std([df_new['A_ratio'], df_new['T_ratio'], 
                                               df_new['G_ratio'], df_new['C_ratio']], axis=0)
        
        # 4. 长度相关复杂度
        df_new['length_category'] = pd.cut(df_new['length'], bins=5, labels=False)
        df_new['length_percentile'] = df_new['length'].rank(pct=True)
        
        # 5. 综合复杂度指标
        df_new['complexity_score'] = (df_new['base_diversity'] * df_new['binding_efficiency'] * 
                                     df_new['composition_balance'])
        
        return df_new
    
    def encode_categorical_features(self, df, fit=True):
        """编码分类特征"""
        df_encoded = df.copy()
        categorical_cols = ['Strand', 'Circtype']
        
        for col in categorical_cols:
            if fit:
                le = LabelEncoder()
                df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
                self.label_encoders[col] = le
            else:
                if col in self.label_encoders:
                    # 处理测试集中可能出现的新类别
                    le = self.label_encoders[col]
                    unique_values = set(le.classes_)
                    df_encoded[col] = df_encoded[col].astype(str)
                    df_encoded[col] = df_encoded[col].apply(
                        lambda x: x if x in unique_values else le.classes_[0]
                    )
                    df_encoded[col] = le.transform(df_encoded[col])
        
        return df_encoded
    
    def select_features(self, X, y, method='all', k=100):
        """特征选择"""
        selected_features = set()
        
        if method in ['univariate', 'all']:
            # 单变量特征选择
            selector = SelectKBest(score_func=f_classif, k=min(k, X.shape[1]))
            selector.fit(X, y)
            selected_features.update(X.columns[selector.get_support()])
        
        if method in ['mutual_info', 'all']:
            # 互信息特征选择
            selector = SelectKBest(score_func=mutual_info_classif, k=min(k, X.shape[1]))
            selector.fit(X, y)
            selected_features.update(X.columns[selector.get_support()])
        
        if method in ['rfe', 'all']:
            # 递归特征消除
            estimator = RandomForestClassifier(n_estimators=50, random_state=42)
            selector = RFE(estimator, n_features_to_select=min(k, X.shape[1]))
            selector.fit(X, y)
            selected_features.update(X.columns[selector.get_support()])
        
        if method in ['lasso', 'all']:
            # LASSO特征选择
            lasso = LassoCV(cv=5, random_state=42, max_iter=1000)
            lasso.fit(X, y)
            selected_features.update(X.columns[np.abs(lasso.coef_) > 1e-5])
        
        return list(selected_features)
    
    def fit_transform(self, train_df, target_col='Tissue'):
        """训练集特征工程"""
        # 分离特征和目标
        X = train_df.drop(columns=[target_col, 'ID'])
        y = train_df[target_col]
        
        # 编码目标变量
        le_target = LabelEncoder()
        y_encoded = le_target.fit_transform(y)
        self.target_encoder = le_target
        
        # 编码分类特征
        X_encoded = self.encode_categorical_features(X, fit=True)
        
        # 创建生物学特征
        X_bio = self.create_biological_features(X_encoded)
        
        # 创建统计特征
        X_stat = self.create_statistical_features(X_bio)
        
        # 创建序列复杂度特征
        X_complex = self.create_sequence_complexity_features(X_stat)
        
        # 处理无穷大和NaN值
        X_complex = X_complex.replace([np.inf, -np.inf], np.nan)
        X_complex = X_complex.fillna(X_complex.median())
        
        # 保存特征名称
        self.feature_names = X_complex.columns.tolist()
        
        # 标准化特征
        X_scaled = self.scaler.fit_transform(X_complex)
        X_scaled = pd.DataFrame(X_scaled, columns=self.feature_names)
        
        return X_scaled, y_encoded
    
    def transform(self, test_df):
        """测试集特征工程"""
        # 移除ID列
        X = test_df.drop(columns=['ID'])
        
        # 编码分类特征
        X_encoded = self.encode_categorical_features(X, fit=False)
        
        # 创建生物学特征
        X_bio = self.create_biological_features(X_encoded)
        
        # 创建统计特征
        X_stat = self.create_statistical_features(X_bio)
        
        # 创建序列复杂度特征
        X_complex = self.create_sequence_complexity_features(X_stat)
        
        # 处理无穷大和NaN值
        X_complex = X_complex.replace([np.inf, -np.inf], np.nan)
        X_complex = X_complex.fillna(X_complex.median())
        
        # 确保特征顺序一致
        missing_cols = set(self.feature_names) - set(X_complex.columns)
        for col in missing_cols:
            X_complex[col] = 0
        X_complex = X_complex[self.feature_names]
        
        # 标准化特征
        X_scaled = self.scaler.transform(X_complex)
        X_scaled = pd.DataFrame(X_scaled, columns=self.feature_names)
        
        return X_scaled

def create_advanced_models():
    """创建高级模型"""
    models = {
        'rf_tuned': RandomForestClassifier(
            n_estimators=300,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            random_state=42,
            n_jobs=-1
        ),
        'xgb_tuned': xgb.XGBClassifier(
            n_estimators=300,
            max_depth=8,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric='mlogloss'
        ),
        'lgb_tuned': lgb.LGBMClassifier(
            n_estimators=300,
            max_depth=8,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbose=-1
        ),
        'catboost_tuned': CatBoostClassifier(
            iterations=300,
            depth=8,
            learning_rate=0.05,
            random_state=42,
            verbose=False
        ),
        'mlp': MLPClassifier(
            hidden_layer_sizes=(200, 100, 50),
            activation='relu',
            solver='adam',
            alpha=0.001,
            learning_rate='adaptive',
            max_iter=500,
            random_state=42
        )
    }
    return models

def create_stacking_ensemble(base_models, X, y):
    """创建Stacking集成模型"""
    # 第一层基模型
    level1_models = [(name, model) for name, model in base_models.items() if name != 'mlp']
    
    # 第二层元模型
    meta_model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        eval_metric='mlogloss'
    )
    
    # 创建Stacking分类器
    stacking_clf = StackingClassifier(
        estimators=level1_models,
        final_estimator=meta_model,
        cv=5,
        stack_method='predict_proba',
        n_jobs=-1
    )
    
    return stacking_clf

def main():
    """主函数"""
    print("=== 高级特征工程 - 环状RNA组织表达预测 ===")
    print("目标：提升Macro-F1分数（当前基线：0.36485）")
    
    # 加载数据
    print("\n1. 加载数据...")
    train_df = pd.read_csv('train.csv')
    test_df = pd.read_csv('test.csv')
    
    print(f"训练集形状: {train_df.shape}")
    print(f"测试集形状: {test_df.shape}")
    
    # 特征工程
    print("\n2. 执行高级特征工程...")
    feature_engineer = AdvancedFeatureEngineer()
    X_train, y_train = feature_engineer.fit_transform(train_df)
    X_test = feature_engineer.transform(test_df)
    
    print(f"特征工程后训练集形状: {X_train.shape}")
    print(f"特征工程后测试集形状: {X_test.shape}")
    
    # 特征选择
    print("\n3. 执行特征选择...")
    selected_features = feature_engineer.select_features(X_train, y_train, method='all', k=150)
    print(f"选择的特征数量: {len(selected_features)}")
    
    X_train_selected = X_train[selected_features]
    X_test_selected = X_test[selected_features]
    
    # 创建模型
    print("\n4. 创建高级模型...")
    models = create_advanced_models()
    
    # 模型评估
    print("\n5. 模型交叉验证评估...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = {}
    
    for name, model in models.items():
        print(f"\n评估 {name}...")
        scores = cross_val_score(model, X_train_selected, y_train, 
                                cv=cv, scoring='f1_macro', n_jobs=-1)
        results[name] = {
            'mean': scores.mean(),
            'std': scores.std(),
            'scores': scores
        }
        print(f"{name} - Macro-F1: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")
    
    # 创建Stacking集成
    print("\n6. 创建Stacking集成模型...")
    stacking_model = create_stacking_ensemble(models, X_train_selected, y_train)
    stacking_scores = cross_val_score(stacking_model, X_train_selected, y_train,
                                    cv=cv, scoring='f1_macro', n_jobs=-1)
    results['stacking'] = {
        'mean': stacking_scores.mean(),
        'std': stacking_scores.std(),
        'scores': stacking_scores
    }
    print(f"Stacking - Macro-F1: {stacking_scores.mean():.4f} (+/- {stacking_scores.std() * 2:.4f})")
    
    # 选择最佳模型
    best_model_name = max(results.keys(), key=lambda x: results[x]['mean'])
    print(f"\n最佳模型: {best_model_name} (Macro-F1: {results[best_model_name]['mean']:.4f})")
    
    # 训练最佳模型并预测
    print("\n7. 训练最佳模型并生成预测...")
    if best_model_name == 'stacking':
        best_model = stacking_model
    else:
        best_model = models[best_model_name]
    
    best_model.fit(X_train_selected, y_train)
    predictions = best_model.predict(X_test_selected)
    
    # 转换预测结果回原始标签
    predictions_original = feature_engineer.target_encoder.inverse_transform(predictions)
    
    # 生成提交文件
    submission = pd.DataFrame({
        'ID': test_df['ID'],
        'Tissue': predictions_original
    })
    
    submission.to_csv('submit_advanced.csv', index=False)
    print("\n预测结果已保存到 submit_advanced.csv")
    
    # 输出结果总结
    print("\n=== 结果总结 ===")
    print(f"基线分数: 0.36485")
    print(f"最佳模型: {best_model_name}")
    print(f"交叉验证Macro-F1: {results[best_model_name]['mean']:.4f}")
    print(f"预期提升: {results[best_model_name]['mean'] - 0.36485:.4f}")
    
    return results, best_model, feature_engineer

if __name__ == "__main__":
    results, best_model, feature_engineer = main()