# new.py
"""
High-score pipeline (unsupervised density & shape features, no leakage)
- 强特征：ALR/组成熵/长度-能量/miRNA 衍生 + 组内残差 + 组内鲁棒缩放 + 组内分位(ECDF)
- 组内 KMeans 聚类标签：k=2 与 k=4（NaN 安全）
- 新增无监督特征（按 Circtype）：
    * 中心距离（IQR 归一化 L2）
    * kNN 平均距离密度（k=5）
    * IsolationForest 异常分数
- 可选安全 OOF stacking（默认关闭）
- LGBM 主干 + SelectKBest；折内调类权重 & 阈值；多种子 bagging
"""

import argparse
from typing import List, Tuple, Dict, Optional
import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin, clone as sk_clone
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.ensemble import IsolationForest

from lightgbm import LGBMClassifier

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# ---------- utils ----------
def weighted_argmax(prob: np.ndarray, weights: np.ndarray) -> np.ndarray:
    return np.argmax(prob * weights[None, :], axis=1)

def apply_thresholds(prob: np.ndarray, thr: np.ndarray) -> np.ndarray:
    meets = (prob >= thr[None, :])
    return np.where(meets.any(axis=1), np.argmax(prob * meets, axis=1), np.argmax(prob, axis=1))

def tune_class_weights(prob: np.ndarray, y_true: np.ndarray, grid: List[float]) -> np.ndarray:
    n_classes = prob.shape[1]; best_f1 = -1.0; w = np.ones(n_classes)
    for _ in range(3):  # 多跑一轮更稳
        for c in range(n_classes):
            best_local = w[c]
            for val in grid:
                w_try = w.copy(); w_try[c] = val
                pred = weighted_argmax(prob, w_try)
                f1 = f1_score(y_true, pred, average="macro")
                if f1 > best_f1:
                    best_f1, best_local = f1, val
            w[c] = best_local
    return w

def tune_thresholds(prob: np.ndarray, y_true: np.ndarray, grid: List[float]) -> np.ndarray:
    n_classes = prob.shape[1]; best_f1 = -1.0
    thr = np.full(n_classes, 1.0/n_classes)
    for _ in range(3):
        for c in range(n_classes):
            best_local = thr[c]
            for val in grid:
                thr_try = thr.copy(); thr_try[c] = val
                pred = apply_thresholds(prob, thr_try)
                f1 = f1_score(y_true, pred, average="macro")
                if f1 > best_f1:
                    best_f1, best_local = f1, val
            thr[c] = best_local
    return thr

# ---------- transformers ----------
class FeatureBuilder(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None): return self
    def transform(self, X):
        X = pd.DataFrame(X).copy()
        out = pd.DataFrame(index=X.index)
        length = X['length'].astype(float); count = X['miRNA Binding count'].astype(float)
        energy = X['Average free energy'].astype(float); gc = X['GC_content'].astype(float)
        ar, tr, gr, cr = (X['A_ratio'].astype(float), X['T_ratio'].astype(float),
                          X['G_ratio'].astype(float), X['C_ratio'].astype(float))

        out['length_log'] = np.log1p(length)
        out['GC_content'] = gc
        out['AT_ratio'] = ar+tr; out['GC_ratio'] = gr+cr
        out['GC_skew'] = ((gr-cr)/(gr+cr+1e-6)).clip(-1,1)
        out['AT_skew'] = ((ar-tr)/(ar+tr+1e-6)).clip(-1,1)

        comp = np.stack([ar,tr,gr,cr],1)+1e-8; comp /= comp.sum(1,keepdims=True)
        out['entropy'] = -np.sum(comp*np.log(comp),1)

        out['miRNA_log'] = np.log1p(count)
        mirna_per_len = count/(length+1)
        out['mirna_per_len'] = np.clip(mirna_per_len, 0, np.quantile(mirna_per_len,0.99))

        epl = energy/(length+1); q1,q99=np.quantile(epl,[0.01,0.99])
        out['energy'] = energy; out['energy_per_len'] = np.clip(epl,q1,q99)
        out['stability'] = energy*gc; out['binding_eff'] = count/(np.abs(energy)+1)

        out['has_N_num'] = X['has_N'].astype(int)

        out['Strand'] = X['Strand'].astype(str); out['Circtype'] = X['Circtype'].astype(str)
        out['Strand_Circ'] = out['Strand']+'|'+out['Circtype']

        out['alr_A'] = np.log((ar+1e-8)/(gr+1e-8))
        out['alr_T'] = np.log((tr+1e-8)/(gr+1e-8))
        out['alr_C'] = np.log((cr+1e-8)/(gr+1e-8))

        # discretized bins (字符串，后续 OHE)
        out['len_bin'] = pd.qcut(out['length_log'], q=10, duplicates='drop').astype(str)
        out['gc_bin']  = pd.qcut(out['GC_content'], q=10, duplicates='drop').astype(str)

        return out

class FrequencyEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, cols=('Strand','Circtype','Strand_Circ'), alpha=1.0):
        self.cols, self.alpha = cols, alpha
    def fit(self,X,y=None):
        self.maps={}; self.N=len(X)
        for c in self.cols:
            vc=X[c].astype(str).value_counts(); V=len(vc)
            self.maps[c]=((vc+self.alpha)/(self.N+self.alpha*max(V,1))).to_dict()
        return self
    def transform(self,X):
        X=X.copy()
        for c in self.cols:
            m=self.maps.get(c,{}); d=self.alpha/(self.N+self.alpha*max(len(m),1))
            X[f'freq_{c}']=X[c].astype(str).map(m).fillna(d)
        return X

class GroupResidualizer(BaseEstimator, TransformerMixin):
    def __init__(self, target_cols=('miRNA_log','energy','GC_content'), degree=2):
        self.target_cols, self.degree = target_cols, degree
    def fit(self,X,y=None):
        self.models={}; self.global_={}
        x=X['length_log'].astype(float)
        for t in self.target_cols:
            self.models[t]={}
            try: self.global_[t]=np.polyfit(x.values,X[t].astype(float).values,self.degree)
            except: self.global_[t]=np.zeros(self.degree+1)
            for grp,idx in X.groupby('Circtype').groups.items():
                xi=x.loc[idx].values; yi=X.loc[idx,t].astype(float).values
                if len(xi)>=self.degree+1 and np.var(xi)>1e-8:
                    coef=np.polyfit(xi,yi,self.degree)
                else: coef=self.global_[t]
                self.models[t][grp]=coef
        return self
    def transform(self,X):
        X=X.copy(); x=X['length_log'].astype(float).values; g=X['Circtype'].astype(str)
        for t in self.target_cols:
            coefs=g.map(self.models[t]).apply(lambda v: v if v is not None else self.global_[t]).values
            yhat=np.zeros(len(X))
            for k in range(self.degree+1): yhat+=[c[k] for c in coefs]*(x**(self.degree-k))
            X[f'resid_{t}_by_circ']=X[t].astype(float).values-yhat
        return X

class GroupRobustScaler(BaseEstimator, TransformerMixin):
    """(x - median_g) / (IQR_g + eps) 按 Circtype 组内缩放；稳定实现"""
    def fit(self,X,y=None):
        self.meds={}; self.iqrs={}; self.global_med={}; self.global_iqr={}
        g = X.groupby('Circtype')
        num_cols = [c for c in X.columns if np.issubdtype(X[c].dtype, np.number)]
        for col in num_cols:
            med_g = g[col].median()
            q1_g = g[col].quantile(0.25)
            q3_g = g[col].quantile(0.75)
            iqr_g = (q3_g - q1_g).replace(0, 1e-6)
            self.meds[col] = med_g.to_dict()
            self.iqrs[col] = iqr_g.to_dict()
            self.global_med[col] = X[col].median()
            q1, q3 = X[col].quantile([0.25, 0.75])
            self.global_iqr[col] = max(q3 - q1, 1e-6)
        return self
    def transform(self,X):
        X=X.copy(); g=X['Circtype'].astype(str)
        for col in self.meds:
            med = g.map(self.meds[col]).fillna(self.global_med[col]).values
            iqr = g.map(self.iqrs[col]).fillna(self.global_iqr[col]).values
            X[col]=(X[col].astype(float).values-med)/(iqr+1e-6)
        return X

class GroupKMeans(BaseEstimator, TransformerMixin):
    def __init__(self, n_clusters=3, suffix=""):
        self.n_clusters=n_clusters; self.suffix=suffix
    def fit(self,X,y=None):
        self.models={}; self.meds={}; self.feats=[c for c in X.columns if np.issubdtype(X[c].dtype,np.number)]
        for grp,idx in X.groupby('Circtype').groups.items():
            Xi=X.loc[idx,self.feats].copy()
            med=Xi.median(numeric_only=True)
            # 确保median不包含NaN
            med = med.fillna(0)
            Xi=Xi.fillna(med)
            # 再次检查并填充任何剩余的NaN
            Xi = Xi.fillna(0)
            self.meds[grp]=med
            if len(Xi)>=self.n_clusters and not Xi.isna().any().any():
                try:
                    km=KMeans(n_clusters=self.n_clusters,random_state=RANDOM_STATE,n_init=10)
                    km.fit(Xi.values); self.models[grp]=km
                except:
                    self.models[grp]=None
            else: self.models[grp]=None
        return self
    def transform(self,X):
        X=X.copy(); g=X['Circtype'].astype(str); labels=[]
        for i in range(len(X)):
            grp=g.iloc[i]; row=X.iloc[i:i+1]
            feats = row[self.feats].copy()
            # 更安全的NaN处理
            if grp in self.meds:
                feats = feats.fillna(self.meds[grp])
            else:
                feats = feats.fillna(feats.median(numeric_only=True))
            # 再次检查并填充任何剩余的NaN
            feats = feats.fillna(0)
            if grp in self.models and self.models[grp] is not None:
                try:
                    lbl=int(self.models[grp].predict(feats[self.feats].values)[0]); labels.append(f"{grp}_c{lbl}")
                except:
                    labels.append(f"{grp}_c0")  # 如果预测失败，使用默认标签
            else: labels.append(f"{grp}_c0")
        X[f'circ_cluster{self.suffix}']=labels; return X

class GroupPercentiles(BaseEstimator, TransformerMixin):
    """Per-Circtype ECDF percentiles for selected numeric columns (leak-safe)."""
    def __init__(self, cols=('length_log','GC_content','miRNA_log','energy','mirna_per_len','energy_per_len')):
        self.cols=cols
    def fit(self,X,y=None):
        self.sorted_vals={}
        g = X.groupby('Circtype')
        for grp, idx in g.groups.items():
            sub = X.loc[idx, self.cols].copy()
            self.sorted_vals[grp] = {c: np.sort(sub[c].astype(float).values) for c in self.cols}
        self.global_sorted = {c: np.sort(X[c].astype(float).values) for c in self.cols}
        return self
    @staticmethod
    def _percentile(arr_sorted, v):
        if arr_sorted.size==0: return 0.5
        pos = np.searchsorted(arr_sorted, v, side='right')
        return pos / arr_sorted.size
    def transform(self,X):
        X=X.copy(); g=X['Circtype'].astype(str)
        for c in self.cols:
            vals = []
            for i in range(len(X)):
                grp = g.iloc[i]; v = float(X.iloc[i][c])
                sv = self.sorted_vals.get(grp, self.global_sorted)[c]
                vals.append(self._percentile(sv, v))
            X[f'{c}_pct'] = np.array(vals, dtype=float)
        return X

class GroupCentroidDensity(BaseEstimator, TransformerMixin):
    """组内中心距离 + kNN密度 + IsolationForest 异常度（全都 target-free）"""
    def __init__(self, k=5, iforest_estimators=200, random_state=RANDOM_STATE):
        self.k=k; self.iforest_estimators=iforest_estimators; self.random_state=random_state
    def fit(self,X,y=None):
        X = pd.DataFrame(X).copy()
        num = [c for c in X.columns if np.issubdtype(X[c].dtype, np.number)]
        self.num_ = num
        self.grp_stats_ = {}
        self.nn_ = {}
        self.iforest_ = {}
        g = X.groupby('Circtype')
        for grp, idx in g.groups.items():
            Xi = X.loc[idx, num].astype(float)
            med = Xi.median()
            # 确保median不包含NaN
            med = med.fillna(0)
            q1 = Xi.quantile(0.25); q3 = Xi.quantile(0.75); iqr = (q3 - q1).replace(0, 1e-6)
            # 确保iqr不包含NaN
            iqr = iqr.fillna(1e-6)
            self.grp_stats_[grp] = (med, iqr)
            # 填充NaN值
            Xi_clean = Xi.fillna(med)
            # 再次检查并填充任何剩余的NaN
            Xi_clean = Xi_clean.fillna(0)
            # NearestNeighbors
            if not Xi_clean.isna().any().any():
                try:
                    nn = NearestNeighbors(n_neighbors=min(self.k, len(Xi_clean)), algorithm='auto')
                    nn.fit(Xi_clean.values)
                    self.nn_[grp] = (nn, Xi_clean.values)
                    # IsolationForest
                    iforest = IsolationForest(n_estimators=self.iforest_estimators, random_state=self.random_state, contamination='auto')
                    iforest.fit(Xi_clean.values)
                    self.iforest_[grp] = iforest
                except:
                    self.nn_[grp] = None
                    self.iforest_[grp] = None
            else:
                self.nn_[grp] = None
                self.iforest_[grp] = None
        # global fallback
        medg = X[num].median(); q1g = X[num].quantile(0.25); q3g = X[num].quantile(0.75)
        # 确保全局统计量不包含NaN
        medg = medg.fillna(0)
        iqr_g = (q3g - q1g).replace(0, 1e-6).fillna(1e-6)
        self.global_stats_ = (medg, iqr_g)
        # 全局数据清理
        X_global_clean = X[num].fillna(medg).fillna(0)
        if not X_global_clean.isna().any().any():
            try:
                nn = NearestNeighbors(n_neighbors=min(self.k, len(X_global_clean)), algorithm='auto')
                nn.fit(X_global_clean.values)
                self.global_nn_ = (nn, X_global_clean.values)
                self.global_iforest_ = IsolationForest(n_estimators=self.iforest_estimators, random_state=self.random_state, contamination='auto')
                self.global_iforest_.fit(X_global_clean.values)
            except:
                self.global_nn_ = None
                self.global_iforest_ = None
        else:
            self.global_nn_ = None
            self.global_iforest_ = None
        return self
    def transform(self,X):
        X = pd.DataFrame(X).copy()
        g = X['Circtype'].astype(str)
        dcen = np.zeros(len(X)); dknn = np.zeros(len(X)); ifs = np.zeros(len(X))
        for i in range(len(X)):
            grp = g.iloc[i]
            row = X.iloc[i][self.num_].astype(float)
            if grp in self.grp_stats_ and self.nn_[grp] is not None and self.iforest_[grp] is not None:
                try:
                    med, iqr = self.grp_stats_[grp]
                    row_f = row.fillna(med).fillna(0)
                    z = (row_f - med) / (iqr + 1e-6)
                    dcen[i] = float(np.sqrt(np.square(z).sum()))
                    nn, mat = self.nn_[grp]
                    dist, _ = nn.kneighbors([row_f.values], n_neighbors=min(self.k, len(mat)))
                    dknn[i] = float(dist.mean())
                    ifs[i] = -float(self.iforest_[grp].score_samples([row_f.values])[0])  # 越大越"异常"
                except:
                    dcen[i] = 0; dknn[i] = 0; ifs[i] = 0
            elif self.global_nn_ is not None and self.global_iforest_ is not None:
                try:
                    med, iqr = self.global_stats_
                    row_f = row.fillna(med).fillna(0)
                    z = (row_f - med) / (iqr + 1e-6)
                    dcen[i] = float(np.sqrt(np.square(z).sum()))
                    nn, mat = self.global_nn_
                    dist, _ = nn.kneighbors([row_f.values], n_neighbors=min(self.k, len(mat)))
                    dknn[i] = float(dist.mean())
                    ifs[i] = -float(self.global_iforest_.score_samples([row_f.values])[0])
                except:
                    dcen[i] = 0; dknn[i] = 0; ifs[i] = 0
            else:
                dcen[i] = 0; dknn[i] = 0; ifs[i] = 0
        X['grp_d_center'] = dcen
        X['grp_d_knn'] = dknn
        X['grp_iforest'] = ifs
        return X

class DynamicCT(BaseEstimator, TransformerMixin):
    def __init__(self):
        try: self.ohe=OneHotEncoder(handle_unknown="ignore",sparse_output=False)
        except: self.ohe=OneHotEncoder(handle_unknown="ignore",sparse=False)
    def fit(self,X,y=None):
        X=pd.DataFrame(X).copy()
        self.cat=[c for c in X.columns if X[c].dtype==object]
        self.num=[c for c in X.columns if c not in self.cat and np.issubdtype(X[c].dtype,np.number)]
        self.ct=ColumnTransformer([("num",SimpleImputer(strategy="median"),self.num),
                                   ("cat",Pipeline([("imp",SimpleImputer(strategy="most_frequent")),("oh",self.ohe)]),self.cat)],
                                  remainder="drop")
        self.ct.fit(X,y); return self
    def transform(self,X): return self.ct.transform(pd.DataFrame(X).copy())

class StackingProba(BaseEstimator, TransformerMixin):
    """安全 OOF stacking（严格 index 对齐；缺失用 full-model 概率补齐）"""
    def __init__(self, n_classes: int = None, n_splits: int = 4, seed: int = RANDOM_STATE):
        self.n_classes=n_classes; self.n_splits=n_splits; self.seed=seed
        self.model_params=dict(n_estimators=250, num_leaves=31, learning_rate=0.07,
                               subsample=0.9, colsample_bytree=0.9,
                               objective='multiclass', class_weight='balanced',
                               random_state=seed, n_jobs=-1, verbosity=-1)
    def fit(self,X,y=None):
        X=pd.DataFrame(X).copy()
        if y is None: raise ValueError("StackingProba requires y during fit")
        if self.n_classes is None: self.n_classes=int(len(np.unique(y)))
        oof=np.zeros((len(X), self.n_classes), dtype=float)
        skf=StratifiedKFold(self.n_splits, shuffle=True, random_state=self.seed)
        feat_cols=[c for c in X.columns if np.issubdtype(X[c].dtype,np.number)]
        for tr,va in skf.split(X,y):
            mdl=LGBMClassifier(**self.model_params)
            mdl.fit(X.loc[tr,feat_cols], y[tr])
            oof[va]=mdl.predict_proba(X.loc[va,feat_cols])
        self.feat_cols_=feat_cols
        self.full_model_=LGBMClassifier(**self.model_params)
        self.full_model_.fit(X[self.feat_cols_], y)
        self.oof_cols_=[f"stack_p{i}" for i in range(self.n_classes)]
        self.oof_train_=pd.DataFrame(oof, columns=self.oof_cols_, index=X.index)
        return self
    def transform(self,X):
        X=pd.DataFrame(X).copy()
        idx = X.index
        add = pd.DataFrame(index=idx, columns=self.oof_cols_, dtype=float)
        inter = idx.intersection(self.oof_train_.index)
        if len(inter)>0:
            add.loc[inter] = self.oof_train_.loc[inter].values
        missing = add.index[add.isna().any(axis=1)]
        if len(missing)>0:
            proba = self.full_model_.predict_proba(X.loc[missing, self.feat_cols_])
            add.loc[missing] = proba
        return pd.concat([X, add], axis=1)

# ---------- model & pipeline ----------
def make_lgbm(seed=RANDOM_STATE):
    return LGBMClassifier(
        n_estimators=900, num_leaves=63, learning_rate=0.045,
        subsample=0.9, colsample_bytree=0.9, max_depth=-1,
        min_child_samples=25,
        reg_alpha=0.5, reg_lambda=1.0,
        class_weight='balanced', objective='multiclass',
        random_state=seed, n_jobs=-1, verbosity=-1
    )

def build_pipeline(k_features=75, seed=RANDOM_STATE, use_stack=False):
    steps=[("fb",FeatureBuilder()),
           ("freq",FrequencyEncoder()),
           ("resid",GroupResidualizer()),
           ("grp",GroupRobustScaler()),
           ("pct",GroupPercentiles()),
           ("gkm2",GroupKMeans(n_clusters=2, suffix="_2")),
           ("gkm4",GroupKMeans(n_clusters=4, suffix="_4")),
           ("dens",GroupCentroidDensity())]
    if use_stack:
        steps.append(("stack", StackingProba(n_splits=4, seed=seed)))
    steps += [("ct",DynamicCT()),
              ("sel",SelectKBest(mutual_info_classif,k=k_features)),
              ("clf",make_lgbm(seed))]
    return Pipeline(steps)

# ---------- CV ----------
def cv_eval(model,X,y,folds,do_w=True,do_t=True):
    skf=StratifiedKFold(folds,shuffle=True,random_state=RANDOM_STATE)
    ws,ths,scores=[],[],[]
    for tr,va in skf.split(X,y):
        mdl=sk_clone(model); mdl.fit(X.iloc[tr],y[tr]); prob=mdl.predict_proba(X.iloc[va])
        w=np.ones(prob.shape[1]); thr=np.full(prob.shape[1],1/prob.shape[1])
        if do_w: w=tune_class_weights(prob,y[va],np.linspace(0.8,1.2,9))
        if do_t: thr=tune_thresholds(prob,y[va],np.linspace(0.05,0.95,19))
        pred=apply_thresholds(prob*w[None,:],thr) if do_t else weighted_argmax(prob,w)
        scores.append(f1_score(y[va],pred,average="macro")); ws.append(w); ths.append(thr)
    return float(np.mean(scores)),np.mean(ws,0),np.mean(ths,0)

# ---------- run ----------
def run(train_path,test_path,out,k_features,folds,bag_seeds,threshold_mode,save_proba=None,use_stack=False):
    train=pd.read_csv(train_path); test=pd.read_csv(test_path)
    y_raw=train['Tissue'].astype(str).values; classes=np.unique(y_raw)
    c2i={c:i for i,c in enumerate(classes)}; i2c={i:c for c,i in c2i.items()}
    y=np.array([c2i[c] for c in y_raw])
    X = train.drop(['ID','Tissue'], axis=1)
    Xtest = test.drop(['ID'], axis=1)

    do_w=threshold_mode in ("weights","both"); do_t=threshold_mode in ("thresholds","both")
    probs=[]; last_thr=None
    for seed in bag_seeds:
        mdl=build_pipeline(k_features,seed,use_stack=use_stack)
        f1,w,thr=cv_eval(mdl,X,y,folds,do_w,do_t)
        print(f"[CV seed {seed}] Macro-F1={f1:.4f}")
        mdl.fit(X,y); p=mdl.predict_proba(Xtest)*w[None,:]
        probs.append(p); last_thr=thr
    pfin=np.mean(probs,0)
    if save_proba:
        proba_df=pd.DataFrame(pfin,columns=[f"prob_{i}" for i in range(pfin.shape[1])])
        proba_df.insert(0,"ID",test["ID"].values)
        proba_df.to_csv(save_proba,index=False)
        print("Saved probabilities to", save_proba)
    pred=apply_thresholds(pfin,last_thr) if do_t else np.argmax(pfin,1)
    sub=pd.DataFrame({"ID":test["ID"],"Tissue":[i2c[i] for i in pred]}); sub.to_csv(out,index=False)
    print("Saved submission to:", out)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--train",type=str,default="train.csv")
    ap.add_argument("--test",type=str,default="test.csv")
    ap.add_argument("--out",type=str,default="submit.csv")
    ap.add_argument("--k",type=int,default=75)
    ap.add_argument("--folds",type=int,default=5)
    ap.add_argument("--bag_seeds",type=str,default="42")
    ap.add_argument("--threshold_mode",type=str,default="both",choices=["none","weights","thresholds","both"])
    ap.add_argument("--save_proba",type=str,default=None)
    ap.add_argument("--use_stack", action="store_true", help="Enable OOF stacking features")
    a=ap.parse_args()
    run(a.train,a.test,a.out,a.k,a.folds,[int(x) for x in a.bag_seeds.split(",") if x.strip()],
        a.threshold_mode,a.save_proba,a.use_stack)

if __name__=="__main__":
    main()
