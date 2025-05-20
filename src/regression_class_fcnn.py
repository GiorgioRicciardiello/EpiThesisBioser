#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Deep Learning Pipeline for OSA Component Regression and Classification

This script:
1. Loads PCA‐reduced sleep study data.
2. Applies log1p/rank transforms to respiratory targets.
3. Uses Optuna to tune DeepRegressor hyperparameters (layer sizes, dropout, LR, etc.).
4. Performs stratified K-fold CV, collecting train/val true & predicted values per fold.
5. Plots True vs Predicted scatter by OSA category for each fold.
6. Trains a DeepClassifier on stacked component predictions to predict OSA severity.
7. Logs training curves to TensorBoard and saves metrics & plots.

Requirements:
    torch, optuna, scikit-learn, seaborn, matplotlib, tensorboard

Usage:
    python deep_nn_pipeline.py
"""
import os
import json
import pathlib
from typing import Dict, List, Any, Optional

import optuna
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from optuna.samplers import TPESampler
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    mean_squared_error, r2_score,
    mean_absolute_error, median_absolute_error,
    classification_report, accuracy_score
)
from sklearn.preprocessing import StandardScaler
import matplotlib.lines as mlines
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


#%% ---------- Configuration ----------
from config.config import config, sections

DATA_PATH: pathlib.Path = pathlib.Path(config.get('data')['pp_data']['pca_reduced_transf'])
LOG_DIR: pathlib.Path = config.get("results")["dir"].joinpath("neural_net", 'deep_fcnn')
LOG_DIR.mkdir(parents=True, exist_ok=True)

N_SPLITS: int = 5
N_TRIALS: int =  100 # 30
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Transform recommendations
transform_recs = {
    'ahi': 'log1p',
    "resp-oa-total": "log1p",
    "resp-ca-total": "log1p",
    "resp-ma-total": "rank",
    "resp-hi_hypopneas_only-total": "log1p",
}

HUE_ORDER = ['Normal', 'Mild', 'Moderate', 'Severe']


#%% ---------- Utility Functions ----------
# def apply_transform(y: pd.Series, transform: str) -> np.ndarray:
#     if transform == 'log1p':
#         return np.log1p(y.values)
#     elif transform == 'rank':
#         return y.rank(pct=True).values
#     else:
#         return y.values


def evaluate_regression(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'r2': r2_score(y_true, y_pred),
        'mae': mean_absolute_error(y_true, y_pred),
        'medae': median_absolute_error(y_true, y_pred)
    }

def evaluate_classifier(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray = None) -> Dict[str, float]:
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='binary', zero_division=0),
        'recall': recall_score(y_true, y_pred, average='binary', zero_division=0),
        'f1': f1_score(y_true, y_pred, average='binary', zero_division=0),
    }
    if y_proba is not None:
        try:
            metrics['roc_auc'] = roc_auc_score(y_true, y_proba)
        except ValueError:
            metrics['roc_auc'] = float('nan')  # handle edge cases
    return metrics

def plot_summary_stats(df_summary: pd.DataFrame, figsize=(14, 10)) -> None:
    """
    Plot summary statistics (mean ± std) for each metric across targets.

    Creates a 2x2 grid of subplots for metrics:
        - RMSE
        - R²
        - MAE
        - MedAE

    Within each subplot, bars represent each target's mean, with error bars for std.
    Colors are consistent per target, and a single global legend is displayed.

    Args:
        df_summary (pd.DataFrame): Summary DataFrame with columns:
            ['target', 'rmse_mean', 'rmse_std', 'r2_mean', 'r2_std',
             'mae_mean', 'mae_std', 'medae_mean', 'medae_std']
    """
    metrics = ['rmse', 'r2', 'mae', 'medae']
    titles = ['RMSE', 'R²', 'MAE', 'MedAE']

    targets = df_summary['target'].tolist()
    n_targets = len(targets)
    x = np.arange(n_targets)

    # Choose a color map
    cmap = plt.get_cmap('Set3')
    colors = {t: cmap(i+3 % 10) for i, t in enumerate(targets)}

    fig, axes = plt.subplots(2, 2, figsize=figsize, constrained_layout=False)
    axes = axes.flatten()

    for ax, metric, title in zip(axes, metrics, titles):
        means = df_summary[f'{metric}_mean']
        stds = df_summary[f'{metric}_std']
        ax.bar(
            x, means,
            yerr=stds,
            color=[colors[t] for t in targets],
            capsize=5,
            alpha=0.8
        )
        ax.set_xticks(x)
        ax.set_xticklabels(targets, rotation=45, ha='right')
        ax.set_title(title)
        ax.set_xlabel('')
        # Remove x-axis tick labels
        ax.tick_params(axis='x', labelbottom=False)
        ax.grid(axis='y', alpha=0.5)
        ax.set_ylabel(title)

    # Global legend at bottom center with multiple columns
    handles = [plt.Line2D([0], [0], color=colors[t], lw=4) for t in targets]
    fig.legend(
        handles, targets,
        title='Target',
        loc='lower center',
        bbox_to_anchor=(0.5, -0.001),  # Adjusted to fit below the plot
        ncol=min(n_targets, 3),  # Limit columns for readability
        # nrow=2,
        bbox_transform=fig.transFigure
    )

    # Adjust layout to prevent cropping
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15 + 0.05 * (n_targets // 5))  # Dynamic bottom margin based on number of targets

    plt.savefig(LOG_DIR / "fold_metrics_cv.png", dpi=300)
    plt.show()


#%% ---------- Model Definitions ----------
class DeepRegressor(nn.Module):
    """Deep feedforward regressor with BatchNorm & Dropout."""
    def __init__(self, input_dim: int, hidden_dims: List[int], dropout_rate: float) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        dims = [input_dim] + hidden_dims
        for i in range(len(hidden_dims)):
            layers += [
                nn.Linear(dims[i], dims[i+1]),
                nn.BatchNorm1d(dims[i+1]),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout_rate)
            ]
        layers.append(nn.Linear(dims[-1], 1))
        self.model = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)


class DeepClassifier(nn.Module):
    """Deep feedforward classifier with BatchNorm & Dropout."""
    def __init__(self, input_dim: int, hidden_dims: List[int], dropout_rate: float, num_classes: int) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        dims = [input_dim] + hidden_dims
        for i in range(len(hidden_dims)):
            layers += [
                nn.Linear(dims[i], dims[i+1]),
                nn.BatchNorm1d(dims[i+1]),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout_rate)
            ]
        layers.append(nn.Linear(dims[-1], num_classes))
        self.model = nn.Sequential(*layers)
        # print("DeepClassifier architecture:\n", self.model)

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)

# another optional regression for tabular data
class TabularRegressor(nn.Module):
    """
    Tabular regressor tailored for mixed continuous + ordinal inputs.

    - Continuous inputs (e.g. PCA components) are batch-normalized.
    - Ordinal inputs (0–3 questionnaire items) each get their own small Embedding.
    - Embeddings + normalized continuous features are concatenated and run through an MLP.
    Usage:

    # suppose your DataFrame split gave:
    #   X_cont: numpy array of shape (n_samples, n_pca_components)
    #   X_ord : numpy array of shape (n_samples, n_questionnaire_items) with values 0–3

    model = TabularRegressor(
        num_numeric = X_cont.shape[1],
        ordinal_cardinalities = [4]*X_ord.shape[1],   # 4 levels each
        embed_dims = [2]*X_ord.shape[1],              # e.g. embed 0–3 into 2-D
        hidden_dims = [64, 32],                       # your choice
        dropout = 0.3
    ).to(DEVICE)

    # In your training loop:
    x_num_t = torch.tensor(X_cont_train, dtype=torch.float32, device=DEVICE)
    x_ord_t = torch.tensor(X_ord_train, dtype=torch.long,    device=DEVICE)
    y_t     = torch.tensor(y_train,      dtype=torch.float32,device=DEVICE).view(-1,1)

    preds = model(x_num_t, x_ord_t)
    loss  = criterion(preds, y_t)

    Args:
        num_numeric (int): # of continuous features (PCA dims).
        ordinal_cardinalities (List[int]): number of categories for each ordinal feature (here all = 4).
        embed_dims (List[int]): embedding dimension for each ordinal feature.
        hidden_dims (List[int]): sizes of hidden layers in the post-concat MLP.
        dropout (float): dropout probability between MLP layers.
    """
    def __init__(
        self,
        num_numeric: int,
        ordinal_cardinalities: List[int],
        embed_dims: List[int],
        hidden_dims: List[int],
        dropout: float
    ) -> None:
        super().__init__()
        assert len(ordinal_cardinalities) == len(embed_dims), "Must supply one embed_dim per ordinal feature"

        # Embedding layers for each ordinal column
        self.embeddings = nn.ModuleList([
            nn.Embedding(card, dim)
            for card, dim in zip(ordinal_cardinalities, embed_dims)
        ])

        # BatchNorm for numeric inputs
        self.bn_numeric = nn.BatchNorm1d(num_numeric)

        # Build post-concat MLP
        total_embed_dim = sum(embed_dims)
        mlp_input_dim = num_numeric + total_embed_dim
        layers: List[nn.Module] = []
        dims = [mlp_input_dim] + hidden_dims
        for i in range(len(hidden_dims)):
            layers += [
                nn.Linear(dims[i], dims[i+1]),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout)
            ]
        layers.append(nn.Linear(dims[-1], 1))  # final regression output
        self.mlp = nn.Sequential(*layers)

    def forward(self, x_numeric: Tensor, x_ordinal: Tensor) -> Tensor:
        """
        Args:
            x_numeric: Tensor of shape (batch_size, num_numeric)
            x_ordinal: LongTensor of shape (batch_size, num_ordinal), values in [0, card-1]
        Returns:
            Tensor of shape (batch_size, 1)
        """
        # 1) Normalize continuous
        x_num = self.bn_numeric(x_numeric)

        # 2) Embed ordinals
        emb_list = []
        for i, emb in enumerate(self.embeddings):
            emb_list.append(emb(x_ordinal[:, i]))
        x_emb = torch.cat(emb_list, dim=1)

        # 3) Concat and run MLP
        x = torch.cat([x_num, x_emb], dim=1)
        return self.mlp(x)

# ---------- Training Helpers ----------
def train_regression(
    model: nn.Module,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    X_train: Tensor,
    y_train: Tensor,
    X_val: Tensor,
    y_val: Tensor,
    epochs: int,
    writer: SummaryWriter,
    fold: int
) -> nn.Module:
    """Train regression model with early stopping on val loss."""
    best_loss = float('inf')
    best_state: Dict[str, Any] = model.state_dict()
    for epoch in range(1, epochs+1):
        model.train()
        optimizer.zero_grad()
        loss = criterion(model(X_train), y_train)
        loss.backward()
        optimizer.step()
        writer.add_scalar(f"reg/train_loss_fold{fold}", loss.item(), epoch)

        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(X_val), y_val)
            writer.add_scalar(f"reg/val_loss_fold{fold}", val_loss.item(), epoch)
            if val_loss.item() < best_loss:
                best_loss = val_loss.item()
                best_state = model.state_dict()
    model.load_state_dict(best_state)
    return model


def train_classifier(
    model: nn.Module,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    X_train: Tensor,
    y_train: Tensor,
    X_val: Tensor,
    y_val: Tensor,
    epochs: int,
    writer: SummaryWriter,
    fold: int
) -> nn.Module:
    """Train classifier model with early stopping on val accuracy."""
    best_acc = 0.0
    best_state: Dict[str, Any] = model.state_dict()
    for epoch in range(1, epochs+1):
        model.train()
        optimizer.zero_grad()
        logits = model(X_train)
        loss = criterion(logits, y_train)
        loss.backward()
        optimizer.step()
        writer.add_scalar(f"clf/train_loss_fold{fold}", loss.item(), epoch)

        model.eval()
        with torch.no_grad():
            preds = model(X_val).argmax(dim=1)
            acc = (preds == y_val).float().mean().item()
            writer.add_scalar(f"clf/val_acc_fold{fold}", acc, epoch)
            if acc > best_acc:
                best_acc = acc
                best_state = model.state_dict()
    model.load_state_dict(best_state)
    return model


# ---------- Optuna Objective ----------
def objective_regression(trial,
                         X: np.ndarray,
                         y: np.ndarray,
                         strata: np.ndarray) -> float:
    if isinstance(y, pd.Series):
        y = y.to_numpy()

    layers = trial.suggest_int("n_layers", 2, 5)
    hidden_dims = [trial.suggest_int(f"dim_{i}", 32, 256) for i in range(layers)]
    dropout = trial.suggest_float("dropout", 0.1, 0.5)
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    wd = trial.suggest_float("wd", 1e-6, 1e-3, log=True)
    epochs = trial.suggest_int("epochs", 50, 200)

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
    r2_scores: List[float] = []
    for tr, val in skf.split(X, strata):
        model = DeepRegressor(X.shape[1], hidden_dims, dropout).to(DEVICE)
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
        crit = nn.MSELoss()

        X_tr = torch.tensor(X[tr], dtype=torch.float32, device=DEVICE)
        y_tr = torch.tensor(y[tr], dtype=torch.float32, device=DEVICE).view(-1,1)
        X_va = torch.tensor(X[val], dtype=torch.float32, device=DEVICE)
        y_va = torch.tensor(y[val], dtype=torch.float32, device=DEVICE).view(-1,1)

        # TRAIN on X_tr, y_tr
        for _ in range(epochs):
            model.train()
            optimizer.zero_grad()
            preds_tr = model(X_tr)               # <-- use X_tr
            loss = crit(preds_tr, y_tr)          # <-- compute loss
            loss.backward()                      # <-- backprop
            optimizer.step()                     # <-- update

        # EVALUATE on X_va
        model.eval()
        with torch.no_grad():
            preds = model(X_va).cpu().numpy().flatten()
        r2_scores.append(r2_score(y[val], preds))

        # --- GPU cleanup after fold ---
        del model, optimizer, crit
        del X_tr, y_tr, X_va, y_va
        torch.cuda.empty_cache()

    return float(np.mean(r2_scores))


def objective_classifier(
    trial: optuna.trial.Trial,
    X: np.ndarray,
    y: np.ndarray,
    num_classes: int,
    strata: np.ndarray
) -> float:
    """
    Optuna objective for tuning the final classifier. Uses stratified K-fold CV
    on the provided strata labels to ensure each fold preserves the OSA severity distribution.

    Args:
        trial: Optuna trial.
        X: Feature matrix (n_samples, n_components).
        y: True class labels (n_samples,).
        num_classes: Number of distinct classes.
        strata: Stratification labels (e.g. osa_four) of shape (n_samples,).

    Returns:
        Mean validation accuracy across folds.
    """
    # Hyperparameter suggestions
    n_layers    = trial.suggest_int("n_layers", 1, 3)
    hidden_dims = [trial.suggest_int(f"dim_{i}", 32, 128) for i in range(n_layers)]
    dropout     = trial.suggest_float("dropout", 0.1, 0.5)
    lr          = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    wd          = trial.suggest_float("wd", 1e-6, 1e-3, log=True)
    epochs      = trial.suggest_int("epochs", 50, 200)

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
    accuracies: List[float] = []

    for tr_idx, val_idx in skf.split(X, strata):
        # Build a new classifier for this fold
        model = DeepClassifier(
            input_dim=X.shape[1],
            hidden_dims=hidden_dims,
            dropout_rate=dropout,
            num_classes=num_classes
        ).to(DEVICE)

        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
        criterion = nn.CrossEntropyLoss()

        X_tr = torch.tensor(X[tr_idx], dtype=torch.float32, device=DEVICE)
        y_tr = torch.tensor(y[tr_idx], dtype=torch.long,   device=DEVICE)
        X_va = torch.tensor(X[val_idx], dtype=torch.float32, device=DEVICE)
        y_va = torch.tensor(y[val_idx], dtype=torch.long,   device=DEVICE)

        # Training loop
        for _ in range(epochs):
            model.train()
            optimizer.zero_grad()
            loss = criterion(model(X_tr), y_tr)
            loss.backward()
            optimizer.step()

        # Validation
        model.eval()
        with torch.no_grad():
            preds = model(X_va).argmax(dim=1).cpu().numpy()
        accuracies.append(accuracy_score(y[val_idx], preds))

        # --- GPU cleanup after fold ---
        del model, optimizer, criterion
        del X_tr, y_tr, X_va, y_va
        torch.cuda.empty_cache()


    # Return average accuracy
    return float(np.mean(accuracies))

# ---------- CV Collection & Plotting ----------
def run_cv_collect(
    X: np.ndarray,
    y: np.ndarray,
    strata: np.ndarray,
    input_dim: int,
    hidden_dims: List[int],
    dropout: float,
    lr: float,
    wd: float,
    epochs: int,
    writer: SummaryWriter
) -> Dict[int, Dict[str, Dict[str, np.ndarray]]]:
    """
    Performs stratified K-fold cross-validation, trains a DeepRegressor on each fold,
    and collects train/val true & predicted values plus OSA labels.

    Args:
        X (np.ndarray): Feature matrix of shape (n_samples, n_features).
        y (np.ndarray): Target array of shape (n_samples,).
        strata (np.ndarray): Stratification labels (e.g. osa_four) of shape (n_samples,).
        input_dim (int): Number of input features for the network.
        hidden_dims (List[int]): Sizes of each hidden layer.
        dropout (float): Dropout rate for the network.
        lr (float): Learning rate for Adam optimizer.
        wd (float): Weight decay (L2) for Adam optimizer.
        epochs (int): Number of training epochs per fold.
        writer (SummaryWriter): TensorBoard writer for logging.

    Returns:
        Dict[int, Dict[str, Dict[str, np.ndarray]]]:
            A dict keyed by fold index, each containing:
            {
              'train': {'true': np.ndarray, 'pred': np.ndarray, 'osa': np.ndarray},
              'val':   {'true': np.ndarray, 'pred': np.ndarray, 'osa': np.ndarray}
            }
    """
    if isinstance(y, pd.Series):
        y = y.to_numpy()

    cv_results: Dict[int, Dict[str, Dict[str, np.ndarray]]] = {}
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

    for fold, (tr_idx, val_idx) in enumerate(skf.split(X, strata), start=1):
        # Initialize model, optimizer, loss
        model = DeepRegressor(input_dim, hidden_dims, dropout).to(DEVICE)
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
        criterion = nn.MSELoss()

        # Prepare tensors
        X_tr = torch.tensor(X[tr_idx], dtype=torch.float32, device=DEVICE)
        y_tr = torch.tensor(y[tr_idx], dtype=torch.float32, device=DEVICE).view(-1, 1)
        X_va = torch.tensor(X[val_idx], dtype=torch.float32, device=DEVICE)
        y_va = torch.tensor(y[val_idx], dtype=torch.float32, device=DEVICE).view(-1, 1)

        # Train with early stopping
        model = train_regression(
            model, optimizer, criterion,
            X_tr, y_tr, X_va, y_va,
            epochs, writer, fold
        )

        # Evaluate on train & val
        model.eval()
        with torch.no_grad():
            pred_tr = model(X_tr).cpu().numpy().flatten()
            pred_va = model(X_va).cpu().numpy().flatten()

        # Store results
        cv_results[fold] = {
            'train': {
                'idx': tr_idx,
                'true': y[tr_idx],
                'pred': pred_tr,
                'osa': strata[tr_idx]
            },
            'val': {
                'idx': val_idx,
                'true': y[val_idx],
                'pred': pred_va,
                'osa': strata[val_idx]
            }
        }
        # --- GPU cleanup ---
        del model, optimizer, criterion
        del X_tr, y_tr, X_va, y_va
        torch.cuda.empty_cache()
    return cv_results

def plot_cv_splits(
    cv_results: Dict[int, Any],
    target: str,
    hue_order: List[str] = HUE_ORDER
) -> None:
    """
    Plot each fold's Train (left column) and Val (right column) scatter,
    with equal axes limits, consistent marker size, and per-plot RMSE & MSE in title.
    """
    n_folds = len(cv_results)
    # Prepare global limits
    # Gather train/val values separately
    all_train_true, all_train_pred = [], []
    all_val_true,   all_val_pred   = [], []
    for data in cv_results.values():
        all_train_true.extend(data['train']['true'])
        all_train_pred.extend(data['train']['pred'])
        all_val_true.extend(  data['val']['true'])
        all_val_pred.extend(  data['val']['pred'])


    # mn_train = min(min(all_train_true), min(all_train_pred))
    # mx_train = max(max(all_train_true), max(all_train_pred))
    # mn_val = min(min(all_val_true), min(all_val_pred))
    # mx_val = max(max(all_val_true), max(all_val_pred))

    # ensure axes only cover non-negative values
    mn_train = 0.0
    mx_train = max(max(all_train_true), max(all_train_pred), 0.0)

    mn_val = 0.0
    mx_val = max(max(all_val_true), max(all_val_pred), 0.0)

    fig, axes = plt.subplots(n_folds, 2, figsize=(16, 4 * n_folds), sharex=False, sharey=False)

    palette = sns.color_palette('tab10', n_colors=len(hue_order))
    color_map = dict(zip(hue_order, palette))
    for i, (fold, data) in enumerate(cv_results.items(), start=0):
        for j, split in enumerate(['train', 'val']):
            ax = axes[i, j] if n_folds > 1 else axes[j]
            arr_t = data[split]['true']
            arr_p = data[split]['pred']
            df_sc = pd.DataFrame({
                'True': arr_t,
                'Predicted': arr_p,
                'OSA': pd.Categorical(data[split]['osa'],
                                      categories=hue_order,
                                      ordered=True)
            })

            # scatter
            sns.scatterplot(
                data=df_sc, x='True', y='Predicted',
                hue='OSA', hue_order=hue_order,
                palette='tab10', s=10, alpha=0.6,
                edgecolor='none', legend=False, ax=ax
            )
            # identity line
            ax.plot(
                [mn_train if split == 'train' else mn_val,
                 mx_train if split == 'train' else mx_val],
                [mn_train if split == 'train' else mn_val,
                 mx_train if split == 'train' else mx_val],
                ls='--', c='gray'
            )

            # axis limits
            if split == 'train':
                ax.set_xlim(mn_train, mx_train)
                ax.set_ylim(mn_train, mx_train)
            else:
                ax.set_xlim(mn_val, mx_val)
                ax.set_ylim(mn_val, mx_val)

            # title with RMSE/MSE
            rmse = np.sqrt(mean_squared_error(arr_t, arr_p))
            mse = np.mean((arr_p - arr_t) ** 2)
            ax.set_title(f"{split.capitalize()} Fold {fold} (n={len(arr_t)})\n"
                         f"RMSE={rmse:.2f}, MSE={mse:.2f}")
            ax.set_xlabel("True")
            ax.set_ylabel("Predicted")
            ax.grid(alpha=0.3)

    # Global legend
    handles = [mlines.Line2D([], [], color=color_map[c], marker='o', linestyle='', markersize=6)
               for c in hue_order]
    fig.legend(
        handles, hue_order,
        title="OSA Category",
        loc='lower center',
        bbox_to_anchor=(0.5, 0),
        ncol=len(hue_order)
    )
    fig.suptitle(f"{target.replace('_', ' ').title()}: True vs. Predicted by Fold & Split")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(LOG_DIR.joinpath(f"{target}_split_cv.png"),
                bbox_inches='tight',
                pad_inches=0.1)
    plt.show()




# ---------- Main Pipeline ----------
# def run_full_pipeline(
# pre_tuned_params: Optional[Dict[str, Dict[str, Any]]] = None
# ) -> None:
#     writer = SummaryWriter(str(LOG_DIR))
#     df = pd.read_csv(DATA_PATH, low_memory=False)
#
#     # Feature selection
#     section_aliases = [s for s in sections if s != 'resp']
#     features = [c for c in df.columns if any(c.startswith(alias) for alias in section_aliases)]
#     features.append('sleep_hours')
#
#     # Prepare full modeling frame
#     df_model = df.copy()
#     df_model[features] = df_model[features].fillna(0)
#     target_cols = [f"{t}_{transform_recs[t]}" for t in transform_recs]
#     df_model.dropna(subset=target_cols, inplace=True)
#     df_model.reset_index(drop=True, inplace=True)
#
#     # 1) Hold out test set (20%)
#     train_dev, test_df = train_test_split(
#         df_model,
#         test_size=0.2,
#         stratify=df_model['osa_four'],
#         random_state=42
#     )
#     print(f"Train/Dev size: {len(train_dev)},  Test size: {len(test_df)}")
#
#     scaler = StandardScaler().fit(train_dev[features])
#     all_preds: List[np.ndarray] = []
#     results: Dict[str, Any] = {}
#
#     # Prepare container for OOFs
#     oof_df = pd.DataFrame(index=df_model.index)  # Out‐of‐fold (OOF) predictions
#     df_all_fold_metrics = pd.DataFrame()
#     for target, transform in transform_recs.items():
#         if target == 'ahi':
#             continue
#         col = f"{target}_{transform}"
#         print(f"\n=== Processing {col} ===")
#         df_t = df_model[features + [col, 'osa_four']].dropna()
#         y = apply_transform(df_t[col], transform)
#         X = scaler.fit_transform(df_t[features])
#         strata = df_t['osa_four'].values
#
#         if pre_tuned_params and col in pre_tuned_params:
#             params = pre_tuned_params[col]
#             print(f"Using pre-tuned params for {col}: {params}")
#         else:
#             # Optuna tuning
#             study = optuna.create_study(direction="maximize", sampler=TPESampler(seed=42))
#             study.optimize(lambda tr: objective_regression(tr, X, y, strata), n_trials=N_TRIALS)
#             params = study.best_params
#             print("Best params:", params)
#
#         # CV collection
#         cv_results = run_cv_collect(
#             X=X, y=y, strata=strata,
#             input_dim=X.shape[1],
#             hidden_dims=[params[f"dim_{i}"] for i in range(params["n_layers"])],
#             dropout=params["dropout"],
#             lr=params["lr"],
#             wd=params["wd"],
#             epochs=params["epochs"],
#             writer=writer
#         )
#
#         # Plot CV results
#         plot_cv_splits(cv_results, col)
#
#         # Gather out-of-fold preds for classifier
#         # --- after you get cv_results for this target ---
#         oof_preds = np.zeros_like(y)
#
#         for fold_data in cv_results.values():
#             val_idx = fold_data['val']['idx']  # the actual validation indices
#             oof_preds[val_idx] = fold_data['val']['pred']
#
#         oof_df[col] = oof_preds
#         all_preds.append(oof_preds)
#
#         # Store regression metrics
#         fold_metrics = [evaluate_regression(d['val']['true'], d['val']['pred']) for d in cv_results.values()]
#         results[col] = fold_metrics
#         df_fold_metrics = pd.DataFrame(fold_metrics)
#         df_fold_metrics['target'] = target
#         df_all_fold_metrics = pd.concat([df_all_fold_metrics, df_fold_metrics], axis=0)
#
#     # compute symmary metrics within each fold to extract summary stats
#     df_all_folds_metrics_summary = df_all_fold_metrics.groupby('target').agg(
#         rmse_mean=('rmse', 'mean'),
#         rmse_std=('rmse', 'std'),
#         r2_mean=('r2', 'mean'),
#         r2_std=('r2', 'std'),
#         mae_mean=('mae', 'mean'),
#         mae_std=('mae', 'std'),
#         medae_mean=('medae', 'mean'),
#         medae_std=('medae', 'std')
#     ).reset_index()
#
#     plot_summary_stats(df_all_folds_metrics_summary)
#
#     # Save to CSV
#
#     # Save OOF predictions
#     oof_df['osa_four'] = df_model['osa_four'].values
#     oof_df.to_csv(LOG_DIR.joinpath("oof_predictions.csv"), index=True)
#     df_all_fold_metrics.to_csv(LOG_DIR.joinpath("fold_metrics.csv"), index=False)
#     df_all_fold_metrics.to_csv(LOG_DIR.joinpath("fold_summary_all_targets.csv"), index=False)
#
#     # Final classification
#     print("\n=== Training Final Classifier ===")
#     X_class = np.stack(all_preds, axis=1)
#     y_class = df_model['osa_four_numeric'].values
#
#     skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
#     clf_preds = np.zeros(len(y_class), dtype=int)
#     for fold, (tr, val) in enumerate(skf.split(X_class, y_class), start=1):
#         model = DeepClassifier(X_class.shape[1], [128,64,32], 0.3, num_classes=4).to(DEVICE)
#         optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
#         crit = nn.CrossEntropyLoss()
#         X_tr = torch.tensor(X_class[tr], dtype=torch.float32, device=DEVICE)
#         y_tr = torch.tensor(y_class[tr], dtype=torch.long, device=DEVICE)
#         X_va = torch.tensor(X_class[val], dtype=torch.float32, device=DEVICE)
#         y_va = torch.tensor(y_class[val], dtype=torch.long, device=DEVICE)
#
#         model = train_classifier(model, optimizer, crit, X_tr, y_tr, X_va, y_va, epochs=256, writer=writer, fold=fold)
#         preds = model(X_va).argmax(dim=1).cpu().numpy()
#         clf_preds[val] = preds
#         print(f"Fold {fold} report:\n", classification_report(y_va.cpu().numpy(), preds))
#
#     print("Overall Classifier Report:\n", classification_report(y_class, clf_preds))
#
#     # Save metrics
#     with open("deep_nn_regression_metrics.json", "w") as f:
#         json.dump(results, f, indent=2)
#
#     writer.close()


def run_full_pipeline(
    pre_tuned_params: Optional[Dict[str, Dict[str, Any]]] = None
) -> None:
    """
    Full training pipeline for OSA component regression and classification with a held-out test set.

    Steps:
    1. Load PCA‐reduced and questionnaire data.
    2. Hold out 20% of data as test set (stratified by osa_four).
    3. Scale numeric features on train_dev only.
    4. Loop through each respiratory target:
       a. Optionally use pre-tuned params or run Optuna on train_dev via CV.
       b. Run stratified K‐fold CV on train_dev to collect OOF predictions and fold metrics.
       c. Retrain regressor on full train_dev and predict on test set.
       d. Save fold metrics and OOF predictions.
       e. Save test‐set regressor predictions.
    5. Summarize fold metrics (mean ± std) and plot.
    6. Stack OOF predictions to train final classifier; stack test preds for final evaluation.
    7. Optuna‐tune classifier on train_dev, train on full train_dev, evaluate on test set.
    8. Save CSVs: OOF preds, fold metrics, summary metrics, test reg preds, test class report.

    Args:
        pre_tuned_params: Optional mapping of target -> hyperparameter dict to skip tuning.
    """
    writer = SummaryWriter(str(LOG_DIR))
    df = pd.read_csv(DATA_PATH, low_memory=False)

    # Feature selection
    section_aliases = [s for s in sections if s != 'resp']
    features = [c for c in df.columns if any(c.startswith(alias) for alias in section_aliases)]
    features.append('sleep_hours')

    # Prepare modeling frame
    df_model = df.copy()
    df_model[features] = df_model[features].fillna(0)
    target_cols = [f"{t}_{transform_recs[t]}" for t in transform_recs]
    df_model.dropna(subset=target_cols, inplace=True)
    df_model.reset_index(drop=True, inplace=True)

    # Hold out test set
    train_dev, test_df = train_test_split(
        df_model, test_size=0.2,
        stratify=df_model['osa_four'],
        random_state=42
    )
    print(f"Train/Dev size: {len(train_dev)}, Test size: {len(test_df)}")

    # Scale features
    scaler = StandardScaler().fit(train_dev[features])

    # Containers
    all_preds_train: List[np.ndarray] = []
    all_preds_test: List[np.ndarray] = []
    df_all_fold_metrics = pd.DataFrame()
    oof_df = pd.DataFrame(index=train_dev.index)
    best_params: Dict[str, Any] = {}

    # Loop through regression targets
    for target, transform in transform_recs.items():
        if target == 'ahi': continue
        col = f"{target}_{transform}"
        print(f"\n=== Processing {col} ===")

        # Prepare train_dev split
        df_td = train_dev[features + [col, 'osa_four', 'osa_four_numeric']].dropna()
        y_td = np.array(df_td[col])
        X_td = scaler.transform(df_td[features])
        strata_td = df_td['osa_four'].values

        # Hyperparameter tuning or pre-tuned
        if pre_tuned_params and col in pre_tuned_params:
            params = pre_tuned_params[col]
        else:
            study = optuna.create_study(direction="maximize", sampler=TPESampler(seed=42))
            study.optimize(lambda tr: objective_regression(trial=tr,
                                                           X=X_td,
                                                           y=y_td,
                                                           strata=strata_td), n_trials=N_TRIALS)
            params = study.best_params
        best_params[col] = params

        # CV collection for OOF preds
        cv_results = run_cv_collect(
            X_td,
            y_td,
            strata_td,
            input_dim=X_td.shape[1],
            hidden_dims=[params[f"dim_{i}"] for i in range(params["n_layers"])],
            dropout=params["dropout"],
            lr=params["lr"], wd=params["wd"],
            epochs=params["epochs"],
            writer=writer
        )
        plot_cv_splits(cv_results, col)

        # Assemble OOF preds
        oof_preds = np.zeros(len(df_td), dtype=float)
        for fold_data in cv_results.values():
            oof_preds[fold_data['val']['idx']] = fold_data['val']['pred']
        oof_df[col] = pd.Series(oof_preds, index=df_td.index)
        all_preds_train.append(oof_preds)

        # Collect fold metrics
        fold_metrics = [evaluate_regression(d['val']['true'], d['val']['pred']) for d in cv_results.values()]
        df_fold = pd.DataFrame(fold_metrics)
        df_fold['target'] = col
        df_all_fold_metrics = pd.concat([df_all_fold_metrics, df_fold], ignore_index=True)

        # Retrain on full train_dev and predict on test set
        X_full = torch.tensor(X_td, dtype=torch.float32, device=DEVICE)
        y_full = torch.tensor(y_td, dtype=torch.float32, device=DEVICE).view(-1, 1)
        reg_full = DeepRegressor(
            X_td.shape[1],
            [params[f"dim_{i}"] for i in range(params["n_layers"])],
            params["dropout"]
        ).to(DEVICE)
        opt = optim.Adam(reg_full.parameters(), lr=params["lr"], weight_decay=params["wd"])
        crit = nn.MSELoss()
        for _ in range(params["epochs"]):
            reg_full.train()
            opt.zero_grad()
            loss = crit(reg_full(X_full), y_full)
            loss.backward()
            opt.step()

        # Predict on test set
        X_test = scaler.transform(test_df[features])
        with torch.no_grad():
            pred_test = reg_full(torch.tensor(X_test, dtype=torch.float32, device=DEVICE)) \
                .cpu().numpy().flatten()
        all_preds_test.append(pred_test)
        torch.cuda.empty_cache()

    # Summarize fold metrics & plot
    summary = df_all_fold_metrics.groupby('target').agg(
        rmse_mean=('rmse', 'mean'), rmse_std=('rmse', 'std'),
        r2_mean=('r2', 'mean'), r2_std=('r2', 'std'),
        mae_mean=('mae', 'mean'), mae_std=('mae', 'std'),
        medae_mean=('medae', 'mean'), medae_std=('medae', 'std')
    ).reset_index()
    plot_summary_stats(summary)

    # Save CSVs
    oof_df['osa_four'] = train_dev.loc[oof_df.index, 'osa_four']
    oof_df.to_csv(LOG_DIR / "oof_predictions.csv", index=True)
    df_all_fold_metrics.to_csv(LOG_DIR / "fold_metrics.csv", index=False)
    summary.to_csv(LOG_DIR / "fold_metrics_summary.csv", index=False)

    # Save test-set regression predictions
    test_reg_df = pd.DataFrame(
        np.stack(all_preds_test, axis=1),
        index=test_df.index,
        columns=[f"{t}_{transform_recs[t]}" for t in transform_recs if t != 'ahi']
    )
    test_reg_df['osa_four'] = test_df['osa_four'].values
    test_reg_df.to_csv(LOG_DIR / "test_regression_predictions.csv", index=True)

    # Prepare classifier data with correct alignment
    features_classifier = [ f"{target}_{transform}" for target, transform in transform_recs.items() if target != 'ahi']
    X_cl_train = oof_df[features_classifier].values
    y_cl_train = train_dev.loc[oof_df.index, 'osa_four_numeric'].values
    strata_cl = train_dev.loc[oof_df.index, 'osa_four'].values
    X_cl_test = np.stack(all_preds_test, axis=1)
    y_cl_test = test_df['osa_four_numeric'].values

    # Tune classifier
    clf_study = optuna.create_study(direction="maximize", sampler=TPESampler(seed=42))
    clf_study.optimize(
        lambda tr: objective_classifier(trial=tr,
                                        X=X_cl_train,
                                        y=y_cl_train,
                                        num_classes=4,
                                        strata=strata_cl),
        n_trials=N_TRIALS
    )
    best_clf = clf_study.best_params

    best_params['classifier'] = best_clf

    # Train final classifier
    clf = DeepClassifier(
        X_cl_train.shape[1],
        [best_clf[f"dim_{i}"] for i in range(best_clf["n_layers"])],
        best_clf["dropout"],
        num_classes=4
    ).to(DEVICE)
    optc = optim.Adam(clf.parameters(), lr=best_clf["lr"], weight_decay=best_clf["wd"])
    critc = nn.CrossEntropyLoss()
    X_tr = torch.tensor(X_cl_train, dtype=torch.float32, device=DEVICE)
    y_tr = torch.tensor(y_cl_train, dtype=torch.long, device=DEVICE)
    for _ in range(best_clf["epochs"]):
        clf.train()
        optc.zero_grad()
        loss = critc(clf(X_tr), y_tr)
        loss.backward()
        optc.step()

    # Evaluate classifier on test set
    clf.eval()
    with torch.no_grad():
        preds_cl_test = clf(torch.tensor(X_cl_test, dtype=torch.float32, device=DEVICE)) \
            .argmax(dim=1).cpu().numpy()
    print("Final Test Classification Report:")
    report_dict = classification_report(y_cl_test, preds_cl_test, output_dict=True)
    print(report_dict)

    df_report = pd.DataFrame(report_dict).transpose()

    # Save to CSV
    df_report.to_csv(LOG_DIR / "test_classifier_report.csv", index=True)

    import json
    with open(LOG_DIR / "best_params.json", "w") as fp:
        json.dump(best_params, fp, indent=2)

    # Save classifier test predictions
    test_class_df = pd.DataFrame({
        'osa_four_true': y_cl_test,
        'osa_four_pred': preds_cl_test
    }, index=test_df.index)
    test_class_df.to_csv(LOG_DIR / "test_classifier_predictions.csv", index=True)

    writer.close()
    torch.cuda.empty_cache()

if __name__ == "__main__":

    if LOG_DIR / "best_params.json":
        in_Params = json.load(open(LOG_DIR / "best_params.json", 'rb'))
    else:
        in_Params = None

    run_full_pipeline(in_Params)
