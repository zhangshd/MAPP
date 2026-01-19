# MOFTransformer version 2.1.0
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.functional import mean_absolute_error, r2_score, mean_absolute_percentage_error
import numpy as np


def init_weights(module):
    if isinstance(module, (nn.Linear, nn.Embedding)):
        module.weight.data.normal_(mean=0.0, std=0.02)
    elif isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)

    if isinstance(module, nn.Linear) and module.bias is not None:
        module.bias.data.zero_()


def symlog(x, threshold=1e-4):
    """
    Symmetric log transform: linear near 0, logarithmic away from 0.
    Formula: sign(x) * log10(1 + |x|/threshold)
    """
    return torch.sign(x) * torch.log10(1 + torch.abs(x) / threshold)


def symlog_inverse(y, threshold=1e-4):
    """
    Inverse of symlog transform.
    Formula: sign(y) * threshold * (10^|y| - 1)
    """
    return torch.sign(y) * threshold * (torch.pow(10, torch.abs(y)) - 1)


def compute_selectivity_loss(pl_module, batch, infer, co2_task, n2_task, 
                              co2_fraction_idx=2, min_loading=1e-6, eps=1e-8):
    """
    Compute log-selectivity auxiliary loss for CO2/N2 adsorption.
    
    Selectivity: S = (q_CO2 / y_CO2) / (q_N2 / y_N2)
    Log-selectivity: log(S) = log(q_CO2) - log(q_N2) + log(1-y_CO2) - log(y_CO2)
    
    This loss encourages the model to predict correct relative adsorption between 
    CO2 and N2, providing additional physical constraint beyond individual MSE losses.
    
    Args:
        pl_module: Lightning module with downstream_heads
        batch: Batch dictionary containing targets and extra_fea
        infer: Inference result containing cls_feats (avoid duplicate forward pass)
        co2_task: Name of CO2 loading task (e.g., 'ArcsinhAbsLoadingCO2')
        n2_task: Name of N2 loading task (e.g., 'ArcsinhAbsLoadingN2')
        co2_fraction_idx: Index of CO2 fraction in extra_fea
        min_loading: Minimum loading threshold for valid samples
        eps: Small constant for numerical stability
        
    Returns:
        Selectivity loss tensor (0 if not enough valid samples, for DDP compatibility)
    """
    # Return 0 loss instead of None for DDP compatibility
    device = batch["target"].device
    zero_loss = torch.tensor(0.0, device=device, requires_grad=True)
    
    # Get task indices
    task_list = list(pl_module.current_tasks.keys())
    if co2_task not in task_list or n2_task not in task_list:
        return zero_loss
    
    co2_task_id = task_list.index(co2_task)
    n2_task_id = task_list.index(n2_task)
    
    # Get masks for both tasks - samples where both CO2 and N2 are valid
    co2_mask = batch["target_mask"][:, co2_task_id]
    n2_mask = batch["target_mask"][:, n2_task_id]
    joint_mask = co2_mask & n2_mask  # Samples with both CO2 and N2 labels
    
    if joint_mask.sum() < 2:
        return zero_loss
    
    # Get ground truth loadings (transformed) for jointly valid samples
    transformed_q_co2_gt = batch["target"][joint_mask, co2_task_id]
    transformed_q_n2_gt = batch["target"][joint_mask, n2_task_id]
    
    # Get CO2 fraction
    extra_fea = batch["extra_fea"][joint_mask]
    y_co2 = extra_fea[:, co2_fraction_idx]
    
    # Get predictions from heads using pre-computed cls_feats
    cls_feats = infer["cls_feats"][joint_mask]
    extra_fea_device = batch["extra_fea"][joint_mask].to(cls_feats.device)
    
    # Get CO2 head prediction
    co2_head = pl_module.downstream_heads[co2_task_id]
    if hasattr(co2_head, 'langmuir_gate'):
        transformed_q_co2_pred = co2_head(cls_feats, extra_fea_device)
    else:
        transformed_q_co2_pred = co2_head(cls_feats).squeeze(-1)
    
    # Get N2 head prediction  
    n2_head = pl_module.downstream_heads[n2_task_id]
    if hasattr(n2_head, 'langmuir_gate'):
        transformed_q_n2_pred = n2_head(cls_feats, extra_fea_device)
    else:
        transformed_q_n2_pred = n2_head(cls_feats).squeeze(-1)
    
    # Recover original loading based on transformation type
    # Detect from task name: "Symlog" or "Arcsinh"
    # Add clamp for numerical stability
    max_exp = 6  # Clamp to prevent overflow (10^6 = 1e6)
    
    if 'SYMLOG' in co2_task.upper():
        # Symlog transform: use symlog_inverse with clamp
        symlog_threshold = pl_module.hparams["config"].get("symlog_threshold", 0.01)
        
        # Clamp to prevent overflow
        q_co2_pred_clamped = torch.clamp(transformed_q_co2_pred, min=-max_exp, max=max_exp)
        q_n2_pred_clamped = torch.clamp(transformed_q_n2_pred, min=-max_exp, max=max_exp)
        q_co2_gt_clamped = torch.clamp(transformed_q_co2_gt, min=-max_exp, max=max_exp)
        q_n2_gt_clamped = torch.clamp(transformed_q_n2_gt, min=-max_exp, max=max_exp)
        
        q_co2_pred = symlog_inverse(q_co2_pred_clamped, symlog_threshold)
        q_n2_pred = symlog_inverse(q_n2_pred_clamped, symlog_threshold)
        q_co2_gt = symlog_inverse(q_co2_gt_clamped, symlog_threshold)
        q_n2_gt = symlog_inverse(q_n2_gt_clamped, symlog_threshold)
    else:
        # Arcsinh transform: q = sinh(arcsinh_q)
        q_co2_pred = torch.sinh(transformed_q_co2_pred)
        q_n2_pred = torch.sinh(transformed_q_n2_pred)
        q_co2_gt = torch.sinh(transformed_q_co2_gt)
        q_n2_gt = torch.sinh(transformed_q_n2_gt)
    
    # Create valid mask: exclude samples where:
    # 1. Either loading is too small (division instability)
    # 2. Pure component (y_CO2 = 0 or 1, selectivity undefined)
    valid_mask = (
        (q_co2_gt > min_loading) & 
        (q_n2_gt > min_loading) &
        (q_co2_pred > min_loading) &
        (q_n2_pred > min_loading) &
        (y_co2 > eps) & 
        (y_co2 < 1 - eps)
    )
    
    if valid_mask.sum() < 2:
        return zero_loss
    
    # Compute log-selectivity: log(S) = log(q_CO2/q_N2)
    # Note: y_CO2 and y_N2 terms cancel out in MSE(log_S_pred, log_S_gt)
    log_S_pred = torch.log(q_co2_pred[valid_mask] + eps) - torch.log(q_n2_pred[valid_mask] + eps)
    log_S_gt = torch.log(q_co2_gt[valid_mask] + eps) - torch.log(q_n2_gt[valid_mask] + eps)
    
    # MSE loss on log-selectivity
    selectivity_loss = F.mse_loss(log_S_pred, log_S_gt)
    
    return selectivity_loss


def collections_init(pl_module, phase='val'):

    if phase == 'test':
        pl_module.test_logits = [[] for _ in range(len(pl_module.hparams.config["tasks"]))]
        pl_module.test_preds = [[] for _ in range(len(pl_module.hparams.config["tasks"]))]
        pl_module.test_labels = [[] for _ in range(len(pl_module.hparams.config["tasks"]))]
        pl_module.test_cifids = [[] for _ in range(len(pl_module.hparams.config["tasks"]))]

    elif phase == 'val':
        pl_module.val_logits = [[] for _ in range(len(pl_module.hparams.config["tasks"]))]
        pl_module.val_preds = [[] for _ in range(len(pl_module.hparams.config["tasks"]))]
        pl_module.val_labels = [[] for _ in range(len(pl_module.hparams.config["tasks"]))]
        pl_module.val_cifids = [[] for _ in range(len(pl_module.hparams.config["tasks"]))]
    else:
        raise ValueError(f"Unsupported phase: {phase}")

def compute_regression(pl_module, batch, task, infer, phase='train'):

    task_id = list(pl_module.current_tasks.keys()).index(task)
    mask_i = batch["target_mask"][:, task_id]
    if not mask_i.any():
        return {
            f"{task}_cif_id": np.array([]),
            f"{task}_cls_feats": torch.tensor([]),
            f"{task}_loss": torch.tensor(0.0),
            f"{task}_logits": torch.tensor([]),
            f"{task}_labels": torch.tensor([]),
        }
    
    # infer = pl_module.infer(batch)
    head = pl_module.downstream_heads[task_id]
    head.to(infer["cls_feats"].device)
    
    # Check if head requires extra_fea input (Langmuir-gated)
    is_langmuir_head = hasattr(head, 'langmuir_gate')
    # Check if this head skips normalize/denormalize (has non-trivial output activation)
    # skip_normalize_task_indices is on the model (ExTransformerV3/V4)
    skip_normalize = (
        hasattr(pl_module.model, 'skip_normalize_task_indices') and 
        task_id in pl_module.model.skip_normalize_task_indices
    ) or (
        hasattr(head, 'skip_normalize') and head.skip_normalize
    )
    
    if is_langmuir_head:
        # LangmuirGatedRegressionHead: pass extra_fea for pressure-based gating
        extra_fea_device = batch["extra_fea"].to(infer["cls_feats"].device)
        logits = head(infer["cls_feats"], extra_fea_device)[mask_i]  # [B]
    else:
        # Standard RegressionHead
        logits = head(infer["cls_feats"]).squeeze(-1)[mask_i]  # [B]

    # logits = infer[f"{task}_logits"][mask_i]  # [B]
    logits = logits.to(torch.float32)
    extra_fea = batch["extra_fea"][mask_i, :]  # [B, extra_fea_dim]

    if "target" not in batch.keys():
        # For Langmuir head or activated output, result is already in original scale
        if is_langmuir_head or skip_normalize:
            final_logits = logits
        else:
            final_logits = pl_module.denormalize(logits, task)
        return {
            f"{task}_cif_id": np.array(infer["cif_id"])[mask_i.cpu().numpy().tolist()],
            f"{task}_cls_feats": infer["cls_feats"][mask_i],
            # f"{task}_loss": torch.tensor(0.0),
            f"{task}_logits": final_logits,
            # f"{task}_labels": torch.zeros_like(logits),
            f"{task}_extra_fea": extra_fea,
            
        }

    labels = batch["target"][mask_i, task_id].clone().detach()  # [B]
    assert len(labels.shape) == 1

    # For Langmuir head or activated output: compute loss in original scale (no normalization)
    # For standard head: normalize labels and compute loss in normalized scale
    if is_langmuir_head or skip_normalize:
        # Output is already in original scale, so use labels directly
        loss = F.mse_loss(logits, labels)
        final_logits = logits
        final_labels = labels.to(torch.float32)
    else:
        # Standard head: normalize labels for loss calculation
        labels_norm = pl_module.normalize(labels, task)
        loss = F.mse_loss(logits, labels_norm)
        final_logits = pl_module.denormalize(logits, task)
        final_labels = labels.to(torch.float32)  # original scale
    

    ret = {
        f"{task}_cif_id": np.array(infer["cif_id"])[mask_i.cpu().numpy().tolist()],
        f"{task}_cls_feats": infer["cls_feats"][mask_i],
        f"{task}_loss": loss,
        f"{task}_logits": final_logits,
        f"{task}_labels": final_labels,
        f"{task}_extra_fea": extra_fea,
    }

    # call update() loss and acc
    # phase = "train" if pl_module.training else "val"
    loss = getattr(pl_module, f"{phase}_{task}_loss")(ret[f"{task}_loss"])
    mae = getattr(pl_module, f"{phase}_{task}_mae")(
        mean_absolute_error(ret[f"{task}_logits"], ret[f"{task}_labels"])
    )
    
    # MAPE with threshold filter: only compute for samples with GT > threshold
    # This avoids MAPE explosion when GT ≈ 0
    mape_threshold = pl_module.hparams["config"].get("mape_threshold", 0.01)
    gt_above_threshold = ret[f"{task}_labels"] > mape_threshold
    if gt_above_threshold.sum() > 0:
        mape_value = mean_absolute_percentage_error(
            ret[f"{task}_logits"][gt_above_threshold], 
            ret[f"{task}_labels"][gt_above_threshold]
        )
    else:
        mape_value = torch.tensor(0.0)
    mape = getattr(pl_module, f"{phase}_{task}_mape")(mape_value)
    if ret[f"{task}_labels"].shape[0] > 1:
        r2 = getattr(pl_module, f"{phase}_{task}_r2")(
            r2_score(ret[f"{task}_logits"], ret[f"{task}_labels"])
        )
    else:
        r2 = getattr(pl_module, f"{phase}_{task}_r2")(torch.tensor(0.0))
    if pl_module.write_log:
        pl_module.log(f"{task}/{phase}/loss", loss, sync_dist=True)
        pl_module.log(f"{task}/{phase}/mae", mae, sync_dist=True)
        pl_module.log(f"{task}/{phase}/r2", r2, sync_dist=True)
        pl_module.log(f"{task}/{phase}/mape", mape, sync_dist=True)

    return ret

def compute_classification(pl_module, batch, task, infer, phase='train'):

    task_id = list(pl_module.current_tasks.keys()).index(task)
    mask_i = batch["target_mask"][:, task_id]
    if not mask_i.any():
        return {
            f"{task}_cif_id": [],
            f"{task}_cls_feats": torch.tensor([]),
            f"{task}_loss": torch.tensor(0.0),
            f"{task}_logits": torch.tensor([]),
            f"{task}_labels": torch.tensor([]),
        }
    
    # infer = pl_module.infer(batch)
    pl_module.downstream_heads[task_id].to(infer["cls_feats"].device)
    logits, binary = pl_module.downstream_heads[task_id](infer["cls_feats"])  # [B, output_dim]
    logits = logits[mask_i]  # [B, output_dim]

    # logits = infer[f"{task}_logits"][mask_i] # [B, C]
    # binary = infer[f"{task}_binary"]
    extra_fea = batch["extra_fea"][mask_i, :]  # [B, extra_fea_dim]
    if "target" not in batch.keys():
        return {
            f"{task}_cif_id": np.array(infer["cif_id"])[mask_i.cpu().numpy().tolist()],
            f"{task}_cls_feats": infer["cls_feats"][mask_i],
            # f"{task}_loss": torch.tensor(0.0),
            f"{task}_logits": logits,
            # f"{task}_labels": torch.zeros_like(logits),
            f"{task}_extra_fea": extra_fea,
        }
    
    labels = batch["target"][mask_i, task_id].clone().detach().long()  # [B]
    assert len(labels.shape) == 1
    if binary:
        logits = logits.squeeze(dim=-1)
        loss = F.binary_cross_entropy_with_logits(input=logits, target=labels.float())
        logits = torch.sigmoid(logits)
    else:
        loss = F.cross_entropy(logits, labels)
        logits = torch.softmax(logits, dim=-1)

    ret = {
        f"{task}_cif_id": np.array(infer["cif_id"])[mask_i.cpu().numpy().tolist()],
        f"{task}_cls_feats": infer["cls_feats"][mask_i],
        f"{task}_loss": loss,
        f"{task}_logits": logits,
        f"{task}_labels": labels,
        f"{task}_extra_fea": extra_fea,
    }

    # call update() loss and acc
    # phase = "train" if pl_module.training else "val"
    loss = getattr(pl_module, f"{phase}_{task}_loss")(
        ret[f"{task}_loss"]
    )
    acc = getattr(pl_module, f"{phase}_{task}_accuracy")(
        ret[f"{task}_logits"], ret[f"{task}_labels"]
    )

    if pl_module.write_log:
        pl_module.log(f"{task}/{phase}/loss", loss, sync_dist=True)
        pl_module.log(f"{task}/{phase}/accuracy", acc, sync_dist=True)

    return ret

def compute_mpp(pl_module, batch):
    infer = pl_module.infer(batch, mask_grid=True)

    mpp_logits = pl_module.mpp_head(infer["grid_feats"])  # [B, max_image_len+2, bins]
    mpp_logits = mpp_logits[
        :, :-1, :
    ]  # ignore volume embedding, [B, max_image_len+1, bins]
    mpp_labels = infer["grid_labels"]  # [B, max_image_len+1, C=1]

    mask = mpp_labels != -100.0  # [B, max_image_len, 1]

    # masking
    mpp_logits = mpp_logits[mask.squeeze(-1)]  # [mask, bins]
    mpp_labels = mpp_labels[mask].long()  # [mask]

    mpp_loss = F.cross_entropy(mpp_logits, mpp_labels)

    ret = {
        "mpp_loss": mpp_loss,
        "mpp_logits": mpp_logits,
        "mpp_labels": mpp_labels,
    }

    # call update() loss and acc
    phase = "train" if pl_module.training else "val"
    loss = getattr(pl_module, f"{phase}_mpp_loss")(ret["mpp_loss"])
    acc = getattr(pl_module, f"{phase}_mpp_accuracy")(
        ret["mpp_logits"], ret["mpp_labels"]
    )

    if pl_module.write_log:
        pl_module.log(f"mpp/{phase}/loss", loss, sync_dist=True)
        pl_module.log(f"mpp/{phase}/accuracy", acc, sync_dist=True)

    return ret


def compute_mtp(pl_module, batch, infer):
    # infer = pl_module.infer(batch)
    mtp_logits = pl_module.mtp_head(infer["cls_feats"])  # [B, hid_dim]
    mtp_labels = torch.LongTensor(batch["mtp"]).to(mtp_logits.device)  # [B]

    mtp_loss = F.cross_entropy(mtp_logits, mtp_labels)  # [B]

    ret = {
        "mtp_loss": mtp_loss,
        "mtp_logits": mtp_logits,
        "mtp_labels": mtp_labels,
    }

    # call update() loss and acc
    phase = "train" if pl_module.training else "val"
    loss = getattr(pl_module, f"{phase}_mtp_loss")(ret["mtp_loss"])
    acc = getattr(pl_module, f"{phase}_mtp_accuracy")(
        ret["mtp_logits"], ret["mtp_labels"]
    )

    if pl_module.write_log:
        pl_module.log(f"mtp/{phase}/loss", loss, sync_dist=True)
        pl_module.log(f"mtp/{phase}/accuracy", acc, sync_dist=True)

    return ret


def compute_vfp(pl_module, batch, infer):
    # infer = pl_module.infer(batch)

    vfp_logits = pl_module.vfp_head(infer["cls_feats"]).squeeze(-1)  # [B]
    vfp_labels = torch.FloatTensor(batch["vfp"]).to(vfp_logits.device)

    assert len(vfp_labels.shape) == 1

    vfp_loss = F.mse_loss(vfp_logits, vfp_labels)
    ret = {
        "vfp_loss": vfp_loss,
        "vfp_logits": vfp_logits,
        "vfp_labels": vfp_labels,
    }

    # call update() loss and acc
    phase = "train" if pl_module.training else "val"
    loss = getattr(pl_module, f"{phase}_vfp_loss")(ret["vfp_loss"])
    mae = getattr(pl_module, f"{phase}_vfp_mae")(
        mean_absolute_error(ret["vfp_logits"], ret["vfp_labels"])
    )

    if pl_module.write_log:
        pl_module.log(f"vfp/{phase}/loss", loss, sync_dist=True)
        pl_module.log(f"vfp/{phase}/mae", mae, sync_dist=True)

    return ret


def compute_ggm(pl_module, batch):
    pos_len = len(batch["grid"]) // 2
    neg_len = len(batch["grid"]) - pos_len
    ggm_labels = torch.cat([torch.ones(pos_len), torch.zeros(neg_len)]).to(
        pl_module.device
    )

    ggm_images = []
    for i, (bti, bfi) in enumerate(zip(batch["grid"], batch["false_grid"])):
        if ggm_labels[i] == 1:
            ggm_images.append(bti)
        else:
            ggm_images.append(bfi)

    ggm_images = torch.stack(ggm_images, dim=0)

    batch = {k: v for k, v in batch.items()}
    batch["grid"] = ggm_images

    infer = pl_module.infer(batch)
    ggm_logits = pl_module.ggm_head(infer["cls_feats"])  # cls_feats
    ggm_loss = F.cross_entropy(ggm_logits, ggm_labels.long())

    ret = {
        "ggm_loss": ggm_loss,
        "ggm_logits": ggm_logits,
        "ggm_labels": ggm_labels,
    }

    # call update() loss and acc
    phase = "train" if pl_module.training else "val"
    loss = getattr(pl_module, f"{phase}_ggm_loss")(ret["ggm_loss"])
    acc = getattr(pl_module, f"{phase}_ggm_accuracy")(
        ret["ggm_logits"], ret["ggm_labels"]
    )

    if pl_module.write_log:
        pl_module.log(f"ggm/{phase}/loss", loss, sync_dist=True)
        pl_module.log(f"ggm/{phase}/accuracy", acc, sync_dist=True)

    return ret


def compute_moc(pl_module, batch, infer):
    if "bbc" in batch.keys():
        task = "bbc"
    else:
        task = "moc"

    # infer = pl_module.infer(batch)
    moc_logits = pl_module.moc_head(
        infer["graph_feats"][:, 1:, :]
    ).flatten()  # [B, max_graph_len] -> [B * max_graph_len]
    moc_labels = (
        infer["mo_labels"].to(moc_logits).flatten()
    )  # [B, max_graph_len] -> [B * max_graph_len]
    mask = moc_labels != -100

    moc_loss = F.binary_cross_entropy_with_logits(
        input=moc_logits[mask], target=moc_labels[mask]
    )  # [B * max_graph_len]

    ret = {
        "moc_loss": moc_loss,
        "moc_logits": moc_logits,
        "moc_labels": moc_labels,
    }

    # call update() loss and acc
    phase = "train" if pl_module.training else "val"
    loss = getattr(pl_module, f"{phase}_{task}_loss")(ret["moc_loss"])
    acc = getattr(pl_module, f"{phase}_{task}_accuracy")(
        nn.Sigmoid()(ret["moc_logits"]), ret["moc_labels"].long()
    )

    if pl_module.write_log:
        pl_module.log(f"{task}/{phase}/loss", loss, sync_dist=True)
        pl_module.log(f"{task}/{phase}/accuracy", acc, sync_dist=True)

    return ret
