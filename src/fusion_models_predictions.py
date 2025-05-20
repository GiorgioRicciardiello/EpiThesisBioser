import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from config.config import config, metrics_psg, encoding, sections
from library.ml_tabular_data.my_simple_xgb import (
    train_late_fusion,
    create_feature_constraints
)

import torch
from torch import nn
import pytorch_lightning as pl
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split


class MetaRegressor(pl.LightningModule):
    def __init__(self, input_dim, hidden_dims=(64, 32), lr=1e-3):
        super().__init__()
        # Build a simple feedforward network with BatchNorm and ReLU
        layers = []
        dims = [input_dim] + list(hidden_dims)
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.BatchNorm1d(dims[i + 1]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(dims[-1], 1))
        self.model = nn.Sequential(*layers)
        self.criterion = nn.MSELoss()
        self.lr = lr

    def forward(self, x):
        return self.model(x).squeeze(-1)

    def training_step(self, batch, batch_idx):
        x, y = batch
        pred = self(x)
        loss = self.criterion(pred, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        pred = self(x)
        val_loss = self.criterion(pred, y)
        self.log("val_loss", val_loss, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss"
            }
        }


class MetaDataModule(pl.LightningDataModule):
    def __init__(self, X_train, y_train, X_val, y_val, batch_size=32):
        super().__init__()
        # Convert numpy arrays to tensors
        self.train_ds = TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                                      torch.tensor(y_train, dtype=torch.float32))
        self.val_ds = TensorDataset(torch.tensor(X_val, dtype=torch.float32),
                                    torch.tensor(y_val, dtype=torch.float32))
        self.batch_size = batch_size

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size)



def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "mse": mean_squared_error(y_true, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
        "mae": mean_absolute_error(y_true, y_pred),
        "r2": r2_score(y_true, y_pred),
    }

def evaluate_ensemble(ensemble_df: pd.DataFrame) -> pd.DataFrame:
    results = {}
    y_true = ensemble_df["true"].values
    for col in ensemble_df.columns:
        if col == "true":
            continue
        results[col] = compute_metrics(y_true, ensemble_df[col].values)
    return pd.DataFrame(results).T



if __name__ == "__main__":
    # Load and prepare
    df = pd.read_csv(config.get("data")["pp_data"]["q_resp"], low_memory=False)
    out_dir = Path(config.get("results")["within_stats"]) / "fusion"
    out_dir.mkdir(parents=True, exist_ok=True)

    events = metrics_psg.get("resp_events")["raw_events"]
    targets = [f"resp-{ev}-total" for ev in events][:2]
    dem_cols = ["dem_age", "dem_bmi", "dem_gender", "dem_race"]
    stratify_col = "osa_four"

    for tgt in targets:
        print(f"\n--- Late fusion for {tgt} ---")
        df_m = df[df[tgt].notna()].copy()
        ensemble_df, blocks, meta_lin = train_late_fusion(
            df=df_m,
            target_col=tgt,
            sections=sections,
            dem_cols=dem_cols,
            fusion_strategies=["mean","mse_weighted","r2_weighted","stacking"],
            optimization=True,
            n_trials=30,
            cv_folds=5,
            test_size=0.2,
            random_state=42,
            use_gpu=True,
            stratify_col=stratify_col
        )
        # evaluate
        metrics = evaluate_ensemble(ensemble_df)
        print(metrics)

        # optional: nonlinear NN stacking
        # build OOF preds & split
        blocks_preds = np.column_stack([blocks[b]["preds"] for b in blocks])
        oof_preds = blocks_preds  # replace with real OOF in prod
        X_meta_tr, X_meta_val, y_meta_tr, y_meta_val = train_test_split(
            oof_preds, df_m[tgt].values, test_size=0.2, random_state=42
        )
        dm = MetaDataModule(X_meta_tr, y_meta_tr, X_meta_val, y_meta_val)
        nn = MetaRegressor(input_dim=oof_preds.shape[1])
        trainer = pl.Trainer(
            gpus=1 if torch.cuda.is_available() else 0,
            max_epochs=20,
            callbacks=[pl.callbacks.EarlyStopping("val_loss", patience=5)],
            logger=False
        )
        trainer.fit(nn, dm)
        # predict
        test_tensor = torch.tensor(
            np.column_stack([blocks[b]["preds"] for b in blocks]),
            dtype=torch.float32
        ).to(nn.device)
        ensemble_df["nn_stack"] = nn(test_tensor).cpu().numpy()
        print("--- with nn_stack ---")
        print(evaluate_ensemble(ensemble_df))
