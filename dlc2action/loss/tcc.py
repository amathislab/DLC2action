#
# Copyright 2020-2022 by A. Mathis Group and contributors. All rights reserved.
#
# This project and all its files are licensed under GNU AGPLv3 or later version. A copy is included in dlc2action/LICENSE.AGPL.
#
#
# Adapted from Temporal Cycle-Consistency Learning by June01
# Adapted from https://github.com/June01/tcc_Temporal_Cycle_Consistency_Loss.pytorch
# Licensed under  Apache License Version 2.0, January 2004
#
""" TCC loss

Adapted from from https://github.com/June01/tcc_Temporal_Cycle_Consistency_Loss.pytorch

"""

import torch
from torch.nn import functional as F
from torch import nn


class _TCCLoss(nn.Module):
    def __init__(
        self,
        loss_type: str = "regression_mse_var",
        variance_lambda: float = 0.001,
        normalize_indices: bool = True,
        normalize_embeddings: bool = False,
        similarity_type: str = "l2",
        num_cycles: int = 20,
        cycle_length: int = 2,
        temperature: float = 0.1,
        label_smoothing: float = 0.1,
    ):
        super().__init__()
        self.loss_type = loss_type
        self.variance_lambda = variance_lambda
        self.normalize_indices = normalize_indices
        self.normalize_embeddings = normalize_embeddings
        self.similarity_type = similarity_type
        self.num_cycles = num_cycles
        self.cycle_length = cycle_length
        self.temperature = temperature
        self.label_smoothing = label_smoothing

    def forward(self, predictions: torch.Tensor, mask: torch.Tensor) -> float:
        real_lens = mask.sum(-1).squeeze().cpu()
        if len(real_lens.shape) == 0:
            return torch.tensor(0)
        return compute_alignment_loss(
            predictions.transpose(-1, -2),
            normalize_embeddings=self.normalize_embeddings,
            normalize_indices=self.normalize_indices,
            loss_type=self.loss_type,
            similarity_type=self.similarity_type,
            num_cycles=self.num_cycles,
            cycle_length=self.cycle_length,
            temperature=self.temperature,
            label_smoothing=self.label_smoothing,
            variance_lambda=self.variance_lambda,
            real_lens=real_lens,
        )


def _align_single_cycle(
    cycle, embs, cycle_length, num_steps, real_len, similarity_type, temperature
):
    # choose from random frame
    n_idx = (torch.rand(1) * real_len).long()[0]
    # n_idx = torch.tensor(8).long()

    # Create labels
    onehot_labels = torch.eye(num_steps)[n_idx]

    # Choose query feats for first frame.
    query_feats = embs[cycle[0], n_idx : n_idx + 1]
    num_channels = query_feats.size(-1)
    for c in range(1, cycle_length + 1):
        candidate_feats = embs[cycle[c]]
        if similarity_type == "l2":
            mean_squared_distance = torch.sum(
                (query_feats.repeat([num_steps, 1]) - candidate_feats) ** 2, dim=1
            )
            similarity = -mean_squared_distance
        elif similarity_type == "cosine":
            similarity = torch.squeeze(
                torch.matmul(candidate_feats, query_feats.transpose(0, 1))
            )
        else:
            raise ValueError("similarity_type can either be l2 or cosine.")

        similarity /= float(num_channels)
        similarity /= temperature

        beta = F.softmax(similarity, dim=0).unsqueeze(1).repeat([1, num_channels])
        query_feats = torch.sum(beta * candidate_feats, dim=0, keepdim=True)

    return similarity.unsqueeze(0), onehot_labels.unsqueeze(0)


def _align(
    cycles,
    embs,
    num_steps,
    real_lens,
    num_cycles,
    cycle_length,
    similarity_type,
    temperature,
    batch_size,
):
    """Align by finding cycles in embs."""
    logits_list = []
    labels_list = []
    for i in range(num_cycles):
        if len(real_lens) == batch_size:
            real_len = int(real_lens[cycles[i][0]])
        else:
            real_len = int(real_lens[cycles[i][0] // batch_size])
        logits, labels = _align_single_cycle(
            cycles[i],
            embs,
            cycle_length,
            num_steps,
            real_len,
            similarity_type,
            temperature,
        )
        logits_list.append(logits)
        labels_list.append(labels)

    logits = torch.cat(logits_list, dim=0)
    labels = torch.cat(labels_list, dim=0).to(embs.device)

    return logits, labels


def gen_cycles(num_cycles, batch_size, cycle_length=2):
    """Generates cycles for alignment.
    Generates a batch of indices to cycle over. For example setting num_cycles=2,
    batch_size=5, cycle_length=3 might return something like this:
    cycles = [[0, 3, 4, 0], [1, 2, 0, 3]]. This means we have 2 cycles for which
    the loss will be calculated. The first cycle starts at sequence 0 of the
    batch, then we find a matching step in sequence 3 of that batch, then we
    find matching step in sequence 4 and finally come back to sequence 0,
    completing a cycle.
    Args:
    num_cycles: Integer, Number of cycles that will be matched in one pass.
    batch_size: Integer, Number of sequences in one batch.
    cycle_length: Integer, Length of the cycles. If we are matching between
      2 sequences (cycle_length=2), we get cycles that look like [0,1,0].
      This means that we go from sequence 0 to sequence 1 then back to sequence
      0. A cycle length of 3 might look like [0, 1, 2, 0].
    Returns:
    cycles: Tensor, Batch indices denoting cycles that will be used for
      calculating the alignment loss.
    """
    sorted_idxes = torch.arange(batch_size).unsqueeze(0).repeat([num_cycles, 1])
    sorted_idxes = sorted_idxes.view([batch_size, num_cycles])
    cycles = sorted_idxes[torch.randperm(len(sorted_idxes))].view(
        [num_cycles, batch_size]
    )
    cycles = cycles[:, :cycle_length]
    cycles = torch.cat([cycles, cycles[:, 0:1]], dim=1)

    return cycles


def compute_stochastic_alignment_loss(
    embs,
    steps,
    seq_lens,
    num_steps,
    batch_size,
    loss_type,
    similarity_type,
    num_cycles,
    cycle_length,
    temperature,
    label_smoothing,
    variance_lambda,
    huber_delta,
    normalize_indices,
    real_lens,
):

    cycles = gen_cycles(num_cycles, batch_size, cycle_length).to(embs.device)
    logits, labels = _align(
        cycles=cycles,
        embs=embs,
        num_steps=num_steps,
        real_lens=real_lens,
        num_cycles=num_cycles,
        cycle_length=cycle_length,
        similarity_type=similarity_type,
        temperature=temperature,
        batch_size=batch_size,
    )

    if "regression" in loss_type:
        steps = steps[cycles[:, 0]]
        seq_lens = seq_lens[cycles[:, 0]]
        loss = regression_loss(
            logits,
            labels,
            num_steps,
            steps,
            seq_lens,
            loss_type,
            normalize_indices,
            variance_lambda,
        )
    else:
        raise ValueError(
            "Unidentified loss type %s. Currently supported loss "
            "types are: regression_mse, regression_mse_var, " % loss_type
        )
    return loss


def compute_alignment_loss(
    embs,
    real_lens,
    steps=None,
    seq_lens=None,
    normalize_embeddings=False,
    loss_type="classification",
    similarity_type="l2",
    num_cycles=20,
    cycle_length=2,
    temperature=0.1,
    label_smoothing=0.1,
    variance_lambda=0.001,
    huber_delta=0.1,
    normalize_indices=True,
):

    # Get the number of timestemps in the sequence embeddings.
    num_steps = embs.shape[1]
    batch_size = embs.shape[0]

    # If steps has not been provided assume sampling has been done uniformly.
    if steps is None:
        steps = (
            torch.arange(0, num_steps)
            .unsqueeze(0)
            .repeat([batch_size, 1])
            .to(embs.device)
        )

    # If seq_lens has not been provided assume is equal to the size of the
    # time axis in the emebeddings.
    if seq_lens is None:
        seq_lens = (
            torch.tensor(num_steps)
            .unsqueeze(0)
            .repeat([batch_size])
            .int()
            .to(embs.device)
        )

    # check if batch_size if consistent with emb etc
    assert num_steps == steps.shape[1]
    assert batch_size == steps.shape[0]

    if normalize_embeddings:
        embs = F.normalize(embs, dim=-1, p=2)

    loss = compute_stochastic_alignment_loss(
        embs=embs,
        steps=steps,
        seq_lens=seq_lens,
        num_steps=num_steps,
        batch_size=batch_size,
        loss_type=loss_type,
        similarity_type=similarity_type,
        num_cycles=num_cycles,
        cycle_length=cycle_length,
        temperature=temperature,
        label_smoothing=label_smoothing,
        variance_lambda=variance_lambda,
        huber_delta=huber_delta,
        normalize_indices=normalize_indices,
        real_lens=real_lens,
    )

    return loss


def regression_loss(
    logits,
    labels,
    num_steps,
    steps,
    seq_lens,
    loss_type,
    normalize_indices,
    variance_lambda,
):
    """Loss function based on regressing to the correct indices.
    In the paper, this is called Cycle-back Regression. There are 3 variants
    of this loss:
    i) regression_mse: MSE of the predicted indices and ground truth indices.
    ii) regression_mse_var: MSE of the predicted indices that takes into account
    the variance of the similarities. This is important when the rate at which
    sequences go through different phases changes a lot. The variance scaling
    allows dynamic weighting of the MSE loss based on the similarities.
    iii) regression_huber: Huber loss between the predicted indices and ground
    truth indices.
    Args:
      logits: Tensor, Pre-softmax similarity scores after cycling back to the
        starting sequence.
      labels: Tensor, One hot labels containing the ground truth. The index where
        the cycle started is 1.
      num_steps: Integer, Number of steps in the sequence embeddings.
      steps: Tensor, step indices/frame indices of the embeddings of the shape
        [N, T] where N is the batch size, T is the number of the timesteps.
      seq_lens: Tensor, Lengths of the sequences from which the sampling was done.
        This can provide additional temporal information to the alignment loss.
      loss_type: String, This specifies the kind of regression loss function.
        Currently supported loss functions: regression_mse, regression_mse_var,
        regression_huber.
      normalize_indices: Boolean, If True, normalizes indices by sequence lengths.
        Useful for ensuring numerical instabilities don't arise as sequence
        indices can be large numbers.
      variance_lambda: Float, Weight of the variance of the similarity
        predictions while cycling back. If this is high then the low variance
        similarities are preferred by the loss while making this term low results
        in high variance of the similarities (more uniform/random matching).
    Returns:
       loss: Tensor, A scalar loss calculated using a variant of regression.
    """
    # Just to be safe, we stop gradients from labels as we are generating labels.
    labels = labels.detach()
    steps = steps.detach()

    if normalize_indices:
        float_seq_lens = seq_lens.float()
        tile_seq_lens = (
            torch.tile(torch.unsqueeze(float_seq_lens, dim=1), [1, num_steps]) + 1e-7
        )
        steps = steps.float() / tile_seq_lens
    else:
        steps = steps.float()

    beta = F.softmax(logits, dim=1)
    true_time = torch.sum(steps * labels, dim=1)
    pred_time = torch.sum(steps * beta, dim=1)

    if loss_type in ["regression_mse", "regression_mse_var"]:
        if "var" in loss_type:
            # Variance aware regression.
            pred_time_tiled = torch.tile(
                torch.unsqueeze(pred_time, dim=1), [1, num_steps]
            )

            pred_time_variance = torch.sum(
                ((steps - pred_time_tiled) ** 2) * beta, dim=1
            )

            # Using log of variance as it is numerically stabler.
            pred_time_log_var = torch.log(pred_time_variance + 1e-7)
            squared_error = (true_time - pred_time) ** 2
            return torch.mean(
                torch.exp(-pred_time_log_var) * squared_error
                + variance_lambda * pred_time_log_var
            )

        else:
            return torch.mean((true_time - pred_time) ** 2)
    else:
        raise ValueError(
            "Unsupported regression loss %s. Supported losses are: "
            "regression_mse, regresstion_mse_var." % loss_type
        )
