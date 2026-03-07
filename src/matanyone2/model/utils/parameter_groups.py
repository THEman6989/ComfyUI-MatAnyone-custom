"""
Parameter group utilities for MatAnyone2 training.
"""

import logging

log = logging.getLogger()


def get_parameter_groups(model, stage_cfg, print_log=False):
    """
    Assigns different weight decays and learning rates to different parameters.
    Returns parameter groups for optimizer.
    """
    weight_decay = stage_cfg.weight_decay
    embed_weight_decay = stage_cfg.embed_weight_decay
    backbone_lr_ratio = stage_cfg.backbone_lr_ratio
    base_lr = stage_cfg.learning_rate

    backbone_params = []
    embed_params = []
    other_params = []

    embedding_names = ["summary_pos", "query_init", "query_emb", "obj_pe"]
    embedding_names = [e + ".weight" for e in embedding_names]

    memo = set()
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param in memo:
            continue
        memo.add(param)

        if name.startswith("module"):
            name = name[7:]

        inserted = False
        if name.startswith("pixel_encoder."):
            backbone_params.append(param)
            inserted = True
            if print_log:
                log.info("%s counted as a backbone parameter.", name)
        else:
            for e in embedding_names:
                if name.endswith(e):
                    embed_params.append(param)
                    inserted = True
                    if print_log:
                        log.info("%s counted as an embedding parameter.", name)
                    break

        if not inserted:
            other_params.append(param)

    parameter_groups = [
        {
            "params": backbone_params,
            "lr": base_lr * backbone_lr_ratio,
            "weight_decay": weight_decay,
        },
        {"params": embed_params, "lr": base_lr, "weight_decay": embed_weight_decay},
        {"params": other_params, "lr": base_lr, "weight_decay": weight_decay},
    ]

    return parameter_groups
