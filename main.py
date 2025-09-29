#!/usr/bin/env python3

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
