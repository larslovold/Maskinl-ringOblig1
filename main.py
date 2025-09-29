#!/usr/bin/env python3

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime

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


"test"