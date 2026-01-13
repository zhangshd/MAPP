'''
Author: zhangshd
Date: 2024-08-17 19:01:41
LastEditors: zhangshd
LastEditTime: 2024-12-02 23:06:00
'''
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from argparse import ArgumentParser
from CGCNN_MT.datamodule import data_interface
from CGCNN_MT.datamodule.dataset import *
import yaml
import torch
import numpy as np
from pytorch_lightning import Trainer
from pathlib import Path
import pandas as pd
import matplotlib
from sklearn.metrics import r2_score, confusion_matrix, mean_absolute_error
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, matthews_corrcoef, roc_auc_score, roc_curve
from CGCNN_MT.module.module_utils import plot_roc_curve, plot_scatter, plot_confusion_matrix
import matplotlib.pyplot as plt
from CGCNN_MT.utils import load_model_from_dir, MODEL_NAME_TO_DATASET_CLS
from torch.utils.data import DataLoader
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
matplotlib.use('Agg')

def main(model_dir, split='test', result_dir=None):
    
    model_dir = Path(model_dir)
    model, trainer = load_model_from_dir(model_dir)
    model.eval()
    hparams = model.hparams
    print("Model hyperparameters:" + "///"*20)
    for k, v in hparams.items():
        if isinstance(v, (str, int, float, bool)):
            print(f"{k}: {v}")
    print("Model hyperparameters:" + "///"*20)
    model_name = "@".join(str(model_dir).split("/")[-2:]) ## model_name = "model_name@version"
    if result_dir is not None:
        result_dir = Path(result_dir)
        log_dir = result_dir / f"{model_name}"
    else:
        result_dir = log_dir = model_dir
    log_dir.mkdir(exist_ok=True, parents=True)
    
    all_outputs = {}
    dm = data_interface.DInterface(**hparams)
    dm.setup(stage='test')
    outputs = trainer.predict(model, dm.test_dataloader())
    for task_id, task in enumerate(hparams["tasks"]):
        print(f"Predicting {task}...")
        task_tp = hparams["task_types"][task_id]
        task_outputs = {}
        task_outputs[f"Predicted"] = torch.cat([d[f"{task}_pred"] for d in outputs], dim=0).cpu().numpy().squeeze()
        task_outputs[f'last_layer_fea'] = torch.cat([d[f'{task}_last_layer_fea'] for d in outputs], dim=0).cpu().numpy().squeeze()
        task_outputs[f"GroundTruth"] = torch.cat([d[f"{task}_labels"] for d in outputs], dim=0).cpu().numpy().squeeze()
        task_outputs[f"CifId"] = np.concatenate([d[f"{task}_cif_id"] for d in outputs], axis=0)
        
        # Handle pressure recovery based on config
        # For arcsinh pressure: P = sinh(arcsinh_P)
        # For log pressure: P = 10^(log_P) - eps
        extra_fea_list = [d[f"{task}_extra_fea"] for d in outputs]
        extra_fea_dim = extra_fea_list[0].shape[1]
        condi_cols = hparams.get("condi_cols", [])
        
        # Determine pressure column and recovery method
        if len(condi_cols) > 0 and "Arcsinh" in condi_cols[0]:
            # Arcsinh pressure: P = sinh(arcsinh_P)
            pressure_vals = torch.cat([torch.sinh(d[f"{task}_extra_fea"][:,0]) for d in outputs], dim=0).cpu().numpy().squeeze()
        else:
            # Log pressure: P = 10^(log_P) - eps (legacy format)
            pressure_vals = torch.cat([10**(d[f"{task}_extra_fea"][:,0]) - 1e-5 for d in outputs], dim=0).cpu().numpy().squeeze()
        task_outputs[f"Pressure[bar]"] = pressure_vals
        
        # Handle CO2Fraction - index depends on condi_cols
        # For arcsinh format: [ArcsinhPressure, SymlogPressure, CO2Fraction] -> index 2
        # For log format: [Pressure, CO2Fraction] -> index 1
        co2_fraction_idx = hparams.get("co2_fraction_idx", None)
        if co2_fraction_idx is None:
            # Auto-detect based on condi_cols
            co2_fraction_idx = 2 if (len(condi_cols) > 2 and "CO2Fraction" in condi_cols[2]) else 1
        
        if extra_fea_dim > co2_fraction_idx:
            task_outputs[f"CO2Fraction"] = torch.cat([d[f"{task}_extra_fea"][:,co2_fraction_idx] for d in outputs], dim=0).cpu().numpy().squeeze()
        elif "CO2" in task:
            task_outputs[f"CO2Fraction"] = 1
        elif "N2" in task:
            task_outputs[f"CO2Fraction"] = 0
        else:
            task_outputs[f"CO2Fraction"] = None
        if "classification" in task_tp:
            task_outputs[f"PredictedProb"] = torch.cat([d[f"{task}_logits"] for d in outputs], dim=0).cpu().numpy()
        # for k, v in task_outputs.items():
        #     print(f"{k}: {v.shape}")
        all_outputs[task] = task_outputs
        df_res = pd.DataFrame({k:v for k,v in task_outputs.items() if k not in ["last_layer_fea"]})
        out_cols = ["CifId", "Pressure[bar]", "CO2Fraction", "GroundTruth", "Predicted"]
        if "PredictedProb" in task_outputs:
            out_cols.append("PredictedProb")
        df_res = df_res.reindex(columns=out_cols)
        df_res.to_csv(result_dir/f"{split}_{task}_predictions.csv", index=False)
        np.savez(result_dir/f"{split}_{task}_last_layer_fea.npz", task_outputs["last_layer_fea"])
    return all_outputs

def process_reg_outputs(targets, preds, cifids, task, split, **kwargs):

    targets = np.concatenate(targets)
    preds = np.concatenate(preds)
    print(targets.shape, preds.shape)
    log_dir = kwargs.get("log_dir", None)
    r2 = r2_score(targets, preds)
    mae = mean_absolute_error(targets, preds)
    ret = {
            f"{task}/{split}_R2Score": r2,
            f"{task}/{split}_MeanAbsoluteError": mae,
                            }
    print (f"{task}/{split}_R2Score: {r2:.4f}, {task}/{split}_MeanAbsoluteError: {mae:.4f}")
    if log_dir is None:
        return ret
    
    csv_file = os.path.join(log_dir, f"{split}_results_{task}.csv")
    df_results = pd.DataFrame(
                {
                    "CifId": cifids,
                    "GroundTruth": targets,
                    "Predicted": preds,
                })
    df_results["Error"] = (df_results["GroundTruth"] - df_results["Predicted"]).abs()
    df_results.sort_values(by="Error", inplace=True, ascending=False)
    df_results.to_csv(csv_file, index=False)

    img_file = os.path.join(log_dir, f"{split}_scatter_{task}.png")
    ax = plot_scatter(
            targets,
            preds,
            title=f"{split}/scatter_{task}",
            metrics={"R2": r2, "MAE": mae},
            outfile
            =img_file,
        )
    
    return ret
    
def process_clf_outputs(targets, preds, logits, cifids, task, split, **kwargs):
    targets = np.concatenate(targets)
    preds = np.concatenate(preds)
    logits = np.stack(logits)
    log_dir = kwargs.get("log_dir", None)
    acc = accuracy_score(targets, preds)
    bacc = balanced_accuracy_score(targets, preds)
    f1 = f1_score(targets, preds, average='macro')
    mcc = matthews_corrcoef(targets, preds)
    
    if len(logits[0]) == 2:
        # print(self.hparams.tasks[task_id], labels[task_id])
        try:
            auc_score = roc_auc_score(targets, logits[:, 1])
        except Exception:   ## for binary classification, only one class is present in the dataset
            auc_score = 0.0
    else:
        try:
            auc_score = roc_auc_score(targets, logits, multi_class='ovo', average='macro')
        except Exception:   ## for multi-class classification, only one class is present in the dataset
            auc_score = 0.0

    if log_dir is None:
        return {
                f"{task}/{split}_Accuracy": acc, 
                f"{task}/{split}_BalancedAccuracy": bacc, 
                f"{task}/{split}_F1Score": f1, 
                f"{task}/{split}_MatthewsCorrCoef": mcc, 
                f"{task}/{split}_AUROC": auc_score
                }
    
    csv_file = os.path.join(log_dir, f"{split}_results_{task}.csv")
    df_results = pd.DataFrame(
                {
                    "CifId": cifids,
                    "GroundTruth": targets,
                    "Predicted": preds,
                    "Prob": logits[:, 1] if len(logits[0]) == 2 else logits.tolist(),
                })
    df_results.to_csv(csv_file, index=False)


    cm = confusion_matrix(targets, preds)
    img_file = os.path.join(log_dir, f"{split}_confusion_matrix_{task}.png")
    ax = plot_confusion_matrix(
            cm,
            title=f"{split}/confusion_matrix_{task}",
            outfile=img_file,
        )
    if len(logits[0]) == 2:
        fpr, tpr, thresholds = roc_curve(
            targets,
            logits[:, 1],
            drop_intermediate=False
        )
        img_file = os.path.join(log_dir, f"{split}_roc_curve_{task}.png")
        ax = plot_roc_curve(
            fpr,
            tpr,
            auc_score,
            title=f"{split}/roc_curve_{task}",
            outfile=img_file,
        )
    
    return {
        f"{task}/{split}_Accuracy": acc, 
        f"{task}/{split}_BalancedAccuracy": bacc, 
        f"{task}/{split}_F1Score": f1, 
        f"{task}/{split}_MatthewsCorrCoef": mcc, 
        f"{task}/{split}_AUROC": auc_score
        }
    
if __name__ == '__main__':
    
    # parser = ArgumentParser()
    # parser.add_argument('--model_dir', type=str)

    # args = parser.parse_args()
    # main(args.model_dir)
    # model_dir = Path(__file__).parent/"logs/SymlogAbsLoadingCO2_SymlogAbsLoadingN2_seed42_cgcnn/version_5"  ## GMOF
    # model_dir = Path(__file__).parent/"logs/SymlogAbsLoadingCO2_SymlogAbsLoadingN2_seed42_cgcnn/version_7"  ## GCluster
    # model_dir = Path(__file__).parent/"logs/SymlogAbsLoadingCO2_SymlogAbsLoadingN2_seed42_cgcnn/version_9"  ## GMOF, with softplus output, no selectivity loss
    model_dir = Path(__file__).parent/"logs/SymlogAbsLoadingCO2_SymlogAbsLoadingN2_seed42_cgcnn/version_10"  ## GMOF, with softplus output, with selectivity loss
    # model_dir = Path(__file__).parent/"logs/SymlogAbsLoadingCO2_SymlogAbsLoadingN2_seed42_cgcnn_langmuir/version_1"  ## GMOF with langmuir gate
    # model_dir = Path(__file__).parent/"logs/SymlogAbsLoadingCO2_SymlogAbsLoadingN2_seed42_cgcnn_langmuir/version_2"  ## GCluster with langmuir gate
    main(model_dir)