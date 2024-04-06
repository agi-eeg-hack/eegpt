from enum import Enum
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torchtyping import TensorType

from poyo.nn import (
    Embedding,
    InfiniteVocabEmbedding,
    PerceiverRotary,
    compute_loss_or_metric,
)
from poyo.models import pad, chain, track_mask, track_batch
from poyo.taxonomy import OutputType

from einops import repeat


class TokenType(StringIntEnum):
    DEFAULT = 0
    START_OF_SEQUENCE = 1
    END_OF_SEQUENCE = 2


def create_start_end_unit_tokens(unit_ids, start, end):
    r"""Creates for each unit a start and end token. Each token is defined by the
    unit index, the token type index and the timestamps.

    Args:
        unit_ids (np.ndarray): List of unit identifiers.
        start (float): The start time of the sequence.
        end (float): The end time of the sequence.
    """
    token_type_index = np.array(
        [TokenType.START_OF_SEQUENCE, TokenType.END_OF_SEQUENCE], dtype=np.int64
    )
    token_type_index = repeat(token_type_index, "u -> (t u)", t=len(unit_ids))

    unit_index = np.arange(len(unit_ids))
    unit_index = repeat(unit_index, "u -> (u t)", t=2)

    timestamps = np.array([start, end], dtype=np.float64)
    timestamps = repeat(timestamps, "u -> (t u)", t=len(unit_ids))
    return token_type_index, unit_index, timestamps


def create_linspace_latent_tokens(start, end, step, num_latents_per_step):
    r"""Creates a sequence of latent tokens. Each token is defined by the
    latent index and the timestamps. The sequence is defined by the start and end
    time and the step size. The group of `num_latents_per_step` latents is repeated
    for each step.

    Args:
        start (float): The start time of the sequence.
        end (float): The end time of the sequence.
        step (float): The step size.
        num_latents_per_step (int): The number of latents per step.
    """
    sequence_len = end - start
    latent_timestamps = np.arange(0, sequence_len, step) + step / 2 + start
    latent_index = np.arange(num_latents_per_step, dtype=np.int64)

    num_timestamps = len(latent_timestamps)
    latent_timestamps = repeat(latent_timestamps, "t -> (t u)", u=len(latent_index))

    latent_index = repeat(latent_index, "u -> (t u)", t=num_timestamps)
    return latent_index, latent_timestamps


class POYO(nn.Module):
    def __init__(
        self,
        *,
        dim=512,
        dim_head=64,
        num_latents=64,
        depth=2,
        cross_heads=1,
        self_heads=8,
        ffn_dropout=0.2,
        lin_dropout=0.4,
        atn_dropout=0.0,
        emb_init_scale=0.02,
    ):
        super().__init__()

        self.unit_emb = InfiniteVocabEmbedding(dim, init_scale=emb_init_scale)
        self.session_emb = InfiniteVocabEmbedding(dim, init_scale=emb_init_scale)
        self.spike_type_emb = Embedding(4, dim, init_scale=emb_init_scale)
        self.latent_emb = Embedding(num_latents, dim, init_scale=emb_init_scale)

        self.perceiver_io = PerceiverRotary(
            dim=dim,
            dim_head=dim_head,
            depth=depth,
            cross_heads=cross_heads,
            self_heads=self_heads,
            ffn_dropout=ffn_dropout,
            lin_dropout=lin_dropout,
            atn_dropout=atn_dropout,
        )

        # Output projections + loss
        self.readout = nn.Linear(dim, 2)

        self.dim = dim

    def forward(
        self,
        *,
        # input sequence
        spike_unit_index,  # (B, N_in)
        spike_timestamps,  # (B, N_in)
        spike_type,  # (B, N_in)
        input_mask=None,  # (B, N_in)
        input_seqlen=None,
        # latent sequence
        latent_index,  # (B, N_latent)
        latent_timestamps,  # (B, N_latent)
        latent_seqlen=None,
        # output sequence
        session_index,  # (B,)
        output_timestamps,  # (B, N_out)
        output_seqlen=None,
        output_batch_index=None,
        output_mask=None,
        output_values: Optional[Dict[str, torch.Tensor]] = None,
        output_weights: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[
        Dict[str, TensorType["batch", "*nqueries", "*nchannelsout"]],
        torch.Tensor,
        Dict[str, torch.Tensor],
    ]:

        # input
        inputs = self.unit_emb(spike_unit_index) + self.spike_type_emb(spike_type)

        # latents
        latents = self.latent_emb(latent_index)

        # outputs
        output_queries = self.session_emb(session_index)

        # feed into perceiver
        output_latents = self.perceiver_io(
            inputs=inputs,
            latents=latents,
            output_queries=output_queries,
            input_timestamps=spike_timestamps,
            latent_timestamps=latent_timestamps,
            output_query_timestamps=output_timestamps,
            input_mask=input_mask,
            input_seqlen=input_seqlen,
            latent_seqlen=latent_seqlen,
            output_query_seqlen=output_seqlen,
        )

        # readout layer
        output_pred = self.readout(output_latents)

        assert output_mask is not None
        loss = compute_loss_or_metric(
            "mse",
            OutputType.CONTINUOUS,
            output_pred[output_mask],
            output_values,
            output_weights,
        )

        output = []
        batch_size = output_latents.shape[0]
        for i in range(batch_size):
            output.append(output[i, output_mask[i]])

        return output, loss


class POYOTokenizer:
    r"""Tokenizer used to tokenize Data for the POYO1 model.

    This tokenizer can be called as a transform. If you are applying multiple
    transforms, make sure to apply this one last.

    Args:
        unit_tokenizer (Callable): Tokenizer for the units.
        session_tokenizer (Callable): Tokenizer for the sessions.
        weight_registry (Dict): Registry of the weights.
        latent_step (float): Step size for generating latent tokens.
        num_latents_per_step (int): Number of latents per step.
    """

    def __init__(
        self,
        unit_tokenizer,
        session_tokenizer,
        latent_step,
        num_latents_per_step,
        using_memory_efficient_attn: bool = True,
        eval=False,
    ):
        self.unit_tokenizer = unit_tokenizer
        self.session_tokenizer = session_tokenizer

        self.latent_step = latent_step
        self.num_latents_per_step = num_latents_per_step

        self.using_memory_efficient_attn = using_memory_efficient_attn
        self.eval = eval

    def __call__(self, data):
        # context window
        start, end = 0, 1.0  # data.domain, data.end

        ### prepare input
        unit_ids = data.units.id
        spike_unit_index = data.spikes.unit_index
        spike_timestamps = data.spikes.timestamps

        # create start and end tokens for each unit
        (
            se_token_type_index,
            se_unit_index,
            se_timestamps,
        ) = create_start_end_unit_tokens(unit_ids, start, end)

        # append start and end tokens to the spike sequence
        spike_token_type_index = np.concatenate(
            [se_token_type_index, np.zeros_like(spike_unit_index)]
        )
        spike_unit_index = np.concatenate([se_unit_index, spike_unit_index])
        spike_timestamps = np.concatenate([se_timestamps, spike_timestamps])

        # unit_index is relative to the recording, so we want it to map it to
        # the global unit index
        local_to_global_map = np.array(self.unit_tokenizer(unit_ids))
        spike_unit_index = local_to_global_map[spike_unit_index]

        ### prepare latents
        latent_index, latent_timestamps = create_linspace_latent_tokens(
            start,
            end,
            step=self.latent_step,
            num_latents_per_step=self.num_latents_per_step,
        )

        ### prepare outputs
        session_index = self.session_tokenizer(data.session)

        output_timestamps = data.cursor.timestamps
        output_values = data.cursor.vel
        output_subtask_index = data.cursor.subtask_index

        # Padding
        batch = {
            # input sequence
            "spike_unit_index": pad(spike_unit_index),
            "spike_timestamps": pad(spike_timestamps),
            "spike_type": pad(spike_token_type_index),
            "input_mask": track_mask(spike_unit_index),
            # latent sequence
            "latent_index": latent_index,
            "latent_timestamps": latent_timestamps,
            # output sequence
            "session_index": pad(np.repeat(session_index, len(output_timestamps))),
            "output_timestamps": pad(output_timestamps),
            "output_values": chain(output_values),
            "output_weights": chain(output_weights),
        }

        if self.eval:
            # we will add a few more fields needed for evaluation
            batch["session_id"] = data.session
            batch["absolute_start"] = data.absolute_start
            batch["output_subtask_index"] = chain(output_subtask_index)

        return batch
