#!/usr/bin/env python3

# ===== Imports =====
import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    confusion_matrix, roc_auc_score, roc_curve, precision_recall_fscore_support
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import Pipeline



def ensure_dir(p: Path):
 
    p.mkdir(parents=True, exist_ok=True)

def savefig(fig, path: Path):
 
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)

def confusion_dict(cm):
  
    tn, fp, fn, tp = cm.ravel()
    return {'TN': int(tn), 'FP': int(fp), 'FN': int(fn), 'TP': int(tp)}

def metrics_from_cm(cm):
  
    tn, fp, fn, tp = cm.ravel()
    acc = (tp + tn) / (tp + tn + fp + fn)
    prec = tp / (tp + fp) if (tp + fp) else np.nan
    rec = tp / (tp + fn) if (tp + fn) else np.nan
    spec = tn / (tn + fp) if (tn + fp) else np.nan
    f1 = (2*tp) / (2*tp + fp + fn) if (2*tp + fp + fn) else np.nan
    return {'accuracy': acc, 'precision': prec, 'recall': rec, 'specificity': spec, 'f1': f1}

def plot_cm(cm, title='Confusion Matrix'):
   
    from sklearn.metrics import ConfusionMatrixDisplay
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    fig, ax = plt.subplots(figsize=(5,4))
    disp.plot(ax=ax, colorbar=False)
    ax.set_title(title)
    return fig

def plot_roc(fpr, tpr, label):

    fig, ax = plt.subplots(figsize=(6,5))
    ax.plot(fpr, tpr, label=label)
    ax.plot([0,1],[0,1],'--')  
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve')
    ax.legend()
    return fig

def plot_multi_roc(roc_list):
 
    fig, ax = plt.subplots(figsize=(6,5))
    for (fpr, tpr, lbl) in roc_list:
        ax.plot(fpr, tpr, label=lbl)
    ax.plot([0,1],[0,1],'--')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curves (All Models)')
    ax.legend()
    return fig


def main(args):
    data_path = Path(args.data)
    out_dir = Path(args.outputs)
    ensure_dir(out_dir)

    df = pd.read_csv(data_path)
    df.columns = [c.strip() for c in df.columns]  
    target = 'Diabetes_binary' 

    eda = {}
    eda['shape'] = {'rows': int(df.shape[0]), 'cols': int(df.shape[1])}
    eda['class_balance'] = df[target].value_counts().sort_index().to_dict()
    eda['missing'] = df.isna().sum().to_dict()
    pd.Series(eda['class_balance']).to_csv(out_dir/'class_balance.csv', header=['count'])
    pd.Series(eda['missing']).to_csv(out_dir/'missing_counts.csv', header=['missing'])

    if 'GenHlth' in df.columns:
        rates = df.groupby('GenHlth')[target].mean().sort_index()
        fig, ax = plt.subplots(figsize=(7,4))
        rates.plot(kind='bar', ax=ax)
        ax.set_ylabel('Share with pre/diabetes')
        ax.set_xlabel('General Health (1=Excellent … 5=Poor)')
        ax.set_title('Diabetes Prevalence by Self-Reported General Health')
        savefig(fig, out_dir/'viz_prevalence_by_genhlth.png')

    X = df.drop(columns=[target])  
    y = df[target]                 
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )


    models = {}

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    lr = LogisticRegression(max_iter=1000, class_weight='balanced')
    lr.fit(X_train_s, y_train)
    models['LogisticRegression'] = ('lr', lr, X_test_s)

    gb = GradientBoostingClassifier(random_state=42, n_estimators=100, max_depth=3)
    gb.fit(X_train, y_train)
    models['GradientBoosting'] = ('gb', gb, X_test)


    dt = DecisionTreeClassifier(max_depth=6, random_state=42, class_weight='balanced')
    dt.fit(X_train, y_train)
    models['DecisionTree'] = ('dt', dt, X_test)

  
    results = []   
    roc_list = []  

    for name, (_, mdl, X_eval) in models.items():
        if hasattr(mdl, 'predict_proba'):
            y_proba = mdl.predict_proba(X_eval)[:, 1]
        else:
            from sklearn.preprocessing import MinMaxScaler
            y_proba = mdl.decision_function(X_eval).reshape(-1,1)
            y_proba = MinMaxScaler().fit_transform(y_proba).ravel()

        y_pred = (y_proba >= 0.5).astype(int)

        cm = confusion_matrix(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba)
        met = metrics_from_cm(cm)
        met.update({'model': name, 'roc_auc': auc})
        results.append(met)

        fig_cm = plot_cm(cm, title=f'Confusion Matrix - {name} (thr=0.50)')
        savefig(fig_cm, out_dir/f'cm_{name}.png')

        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_list.append((fpr, tpr, f'{name} (AUC={auc:.3f})'))

    pd.DataFrame(results).to_csv(out_dir/'model_results.csv', index=False)

    fig_roc = plot_multi_roc(roc_list)
    savefig(fig_roc, out_dir/'roc_all_models.png')


    coef_df = pd.DataFrame({'feature': X.columns, 'coef': models['LogisticRegression'][1].coef_[0]})
    coef_df['abs_coef'] = coef_df['coef'].abs()
    coef_df.sort_values('abs_coef', ascending=False).head(15).to_csv(out_dir/'lr_top_coefficients.csv', index=False)

    importances_gb = pd.Series(models['GradientBoosting'][1].feature_importances_, index=X.columns).sort_values(ascending=False)
    importances_dt = pd.Series(models['DecisionTree'][1].feature_importances_, index=X.columns).sort_values(ascending=False)
    importances_gb.head(15).to_csv(out_dir/'gb_top_importances.csv', header=['importance'])
    importances_dt.head(15).to_csv(out_dir/'dt_top_importances.csv', header=['importance'])


    pipe = Pipeline([
        ('select', SelectKBest(score_func=f_classif, k=10)),
        ('clf', LogisticRegression(max_iter=500, class_weight='balanced'))
    ])
    pipe.fit(X_train, y_train)
    y_proba_k10 = pipe.predict_proba(X_test)[:, 1]
    y_pred_k10 = (y_proba_k10 >= 0.5).astype(int)
    cm_k10 = confusion_matrix(y_test, y_pred_k10)
    auc_k10 = roc_auc_score(y_test, y_proba_k10)

    pd.DataFrame([{
        'k': 10, 'roc_auc': auc_k10, **metrics_from_cm(cm_k10), **confusion_dict(cm_k10)
    }]).to_csv(out_dir/'reduced_features_k10.csv', index=False)


    thr = 0.35  
    y_proba_lr = models['LogisticRegression'][1].predict_proba(models['LogisticRegression'][2])[:,1]
    y_pred_lr_t = (y_proba_lr >= thr).astype(int)
    cm_thr = confusion_matrix(y_test, y_pred_lr_t)
    pd.DataFrame([{
        'threshold': thr, **metrics_from_cm(cm_thr), **confusion_dict(cm_thr)
    }]).to_csv(out_dir/'lr_threshold_tuning.csv', index=False)

"adding experiments to check performance"