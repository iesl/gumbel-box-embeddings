from typing import Tuple, Dict, Any, Union
from torch import Tensor
from .max_margin_models import MaxMarginBoxModel
from .base import BaseBoxModel
from typing import Optional
from allennlp.models import Model
from allennlp.training.metrics import Average
from boxes.box_wrapper import SigmoidBoxTensor, BoxTensor
from boxes.modules import BoxEmbedding
from boxes.utils import log1mexp
from allennlp.modules.token_embedders.embedding import Embedding
from allennlp.modules.token_embedders import Embedding
import torch
import numpy as np
from ..metrics import HitsAt10, F1WithThreshold
from allennlp.training.metrics import F1Measure, FBetaMeasure


@Model.register('max-margin-conditional-Torbox-model')
class MaxMarginConditionalModelTorus(MaxMarginBoxModel):
    def _get_triple_score(self, head: BoxTensor, tail: BoxTensor,
                          relation: BoxTensor, triple_type: str = 'pos') -> torch.Tensor:
        """ Gets score using conditionals.

        :: note: We do not need to worry about the dimentions of the boxes. If
                it can sensibly broadcast it will.
            """
        
        triple = torch.stack((head.data, tail.data), dim =-3)
        int_lengths = head.per_dim_int_length(triple)
        head_tail_box_vol = torch.sum(torch.log(int_lengths.clamp_min(0)+1e-8), dim=-1)
        # score = tail_head_relation_box_vol - tail_relation_box.log_soft_volume(
        #    temp=self.softbox_temp)
        tail_data = tail.data.view(-1, 1, 2, self.embedding_dim)
        score = head_tail_box_vol - tail._intersection_volume(tail_data, True)
        if triple_type == 'pos':
            triple = torch.stack((head.data, tail.data), dim =-3)
            target_probs = torch.ones_like(score)
            if not self.is_eval():
                self.surr_loss = self.pull_loss(triple, target_probs, int_lengths)
            else:
                self.surr_loss = 0
        return score

    def get_scores(self,
                   embeddings: Dict) -> Tuple[torch.Tensor, torch.Tensor]:

        p_s = self._get_triple_score(embeddings['p_h'], embeddings['p_t'],
                                     embeddings['p_r'], triple_type='pos')

        n_s = self._get_triple_score(embeddings['n_h'], embeddings['n_t'],
                                     embeddings['n_r'], triple_type='neg')

        return (p_s, n_s)

    def pull_loss(self, boxes: Tensor, target_probs: Tensor, per_dim_int_lengths: Tensor) -> Tensor:
        """
        :param boxes:  Tensor with shape (..., int_dim, min/delta, embedding_dim) representing boxes on
            the torus, where int_dim is the axes where boxes which are to be intersected are stored
        :param target_probs: Target probabilities, or other Tensor with shape (...) which is > 0 if and only if those
            boxes should overlap
        :param per_dim_int_lengths: Tensor with shape (..., embedding_dim) which has the intersection length in each dimension
            (should be actual intersection length, not in log space)
        """
        first_box = boxes[...,[0],:,:]
        other_boxes = boxes[..., 1:,:,:]
        other_boxes[...,0,:] -= first_box[...,0,:]
        other_boxes[...,0,:] %= 1
        first_case = other_boxes[...,0,:] - first_box[...,1,:] # should be negative
        second_case = 1 - (other_boxes[...,0,:] + other_boxes[...,1,:]) # should be negative
        return (((per_dim_int_lengths < 1e-8) & (target_probs > 0.0)[...,None])[...,None,:] * torch.min(first_case, second_case)**2).mean()


@Model.register('max-margin-conditional-classification-Torbox-model')
class MaxMarginConditionalClassificationModel(MaxMarginConditionalModelTorus):
    def __init__(
            self,
            num_entities: int,
            num_relations: int,
            embedding_dim: int,
            box_type: str = 'MinDeltaBoxesOnTorus',
            single_box: bool = False,
            softbox_temp: float = 10.,
            margin: float = 0.0,
            number_of_negative_samples: int = 0,
            debug: bool = False,
            regularization_weight: float = 0,
            init_interval_center: float = 0.25,
            init_interval_delta: float = 0.1,
            # adversarial_negative: bool = False,
            # adv_neg_softmax_temp: float = 0.8
    ) -> None:
        super().__init__(
            num_entities, num_relations, embedding_dim, box_type, single_box,
            softbox_temp, margin, number_of_negative_samples, debug,
            regularization_weight, init_interval_center, init_interval_delta)
        self.train_f1 = FBetaMeasure(average='micro')
        #self.valid_f1 = FBetaMeasure(average='micro')
        self.threshold_with_f1 = F1WithThreshold(flip_sign=True)

    def get_ranks(self, embeddings: Dict[str, BoxTensor]) -> Any:
        s = self._get_triple_score(embeddings['h'], embeddings['t'],
                                   embeddings['r'])
        # preds = torch.stack((p_s, n_s), dim=1)  # shape = (batch, 2)
        #self.valid_f1(preds, labels)
        labels = embeddings['label']
        # upate the metrics
        self.threshold_with_f1(s, labels)

        return {}

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        if self.is_eval():
            metrics = self.threshold_with_f1.get_metric(reset)
        else:
            metrics = self.train_f1.get_metric(reset)
            metrics[
                'regularization_loss'] = self.regularization_loss.get_metric(
                    reset)

        return metrics

    def get_box_embeddings_val(self, h: torch.Tensor, t: torch.Tensor,
                               r: torch.Tensor,
                               label: torch.tensor) -> Dict[str, BoxTensor]:

        return BaseBoxModel.get_box_embeddings_val(
            self, h=h, t=t, r=r, label=label)

    def get_loss(self, scores: Tuple[torch.Tensor, torch.Tensor],
                 label: torch.Tensor) -> torch.Tensor:
        # max margin loss expects label to be float
        label = label.to(scores[0].dtype)
        loss = self.loss_f(*scores, label) + self.regularization_weight * self.surr_loss
        # metrics require 0,1 labels

        if not self.is_eval():
            with torch.no_grad():
                labels = torch.zeros_like(scores[0]).reshape(
                    -1)  # shape = (batch)
                preds = torch.stack(scores, dim=1)
                self.train_f1(preds, labels)

        return loss

@Model.register('BCE-Torbox-model')
class BCEBoxModel(MaxMarginConditionalModelTorus):
    def __init__(self,
                 num_entities: int,
                 num_relations: int,
                 embedding_dim: int,
                 box_type: str = 'MinDeltaBoxesOnTorus',
                 single_box: bool = False,
                 softbox_temp: float = 10.,
                 number_of_negative_samples: int = 0,
                 debug: bool = False,
                 regularization_weight: float = 0,
                 init_interval_center: float = 0.25,
                 init_interval_delta: float = 0.1) -> None:
        super().__init__(
            num_entities,
            num_relations,
            embedding_dim,
            box_type=box_type,
            single_box=single_box,
            softbox_temp=softbox_temp,
            margin=0.0,
            number_of_negative_samples=number_of_negative_samples,
            debug=debug,
            regularization_weight=regularization_weight,
            init_interval_center=init_interval_center,
            init_interval_delta=init_interval_delta)

        self.loss_f = torch.nn.NLLLoss(reduction='mean')

    def get_box_embeddings_training(  # type:ignore
            self,
            h: torch.Tensor,
            r: torch.Tensor,
            t: torch.Tensor,  # type:ignore
            label: torch.Tensor,
            **kwargs) -> Dict[str, BoxTensor]:  # type: ignore

        return {
            'h': self.h(h),
            't': self.t(t),
            'r': self.r(r),
            'label': label,
        }

    def _get_triple_score(self, head: BoxTensor, tail: BoxTensor,
                          relation: BoxTensor, label: Tensor) -> torch.Tensor:
        """ Gets score using conditionals.

        :: note: We do not need to worry about the dimentions of the boxes. If
                it can sensibly broadcast it will.
            """

        head.data[...,0,:] = head.data[...,0,:] % 1
        head.data[...,1,:] = torch.sigmoid(head.data[..., 1, :])
        tail.data[...,0,:] = tail.data[...,0,:] % 1
        tail.data[...,1,:] = torch.sigmoid(tail.data[..., 1, :])
        
        triple = torch.stack((head.data, tail.data), dim =-3)
        int_lengths = head.per_dim_int_length(triple)
        triple = torch.stack((head.data, tail.data), dim =-3)
        head_tail_box_vol = head._intersection_volume(triple, True, eps=1e-20)
        # score = tail_head_relation_box_vol - tail_relation_box.log_soft_volume(
        #    temp=self.softbox_temp)
        tail_data = tail.data.view(-1, 1, 2, self.embedding_dim)
        int_lengths_tail = tail.per_dim_int_length(tail_data, True)
        tail_volume = torch.sum(torch.log(int_lengths_tail.clamp_min(1e-20)), dim=-1)
        score = head_tail_box_vol - tail_volume

        triple = torch.stack((head.data, tail.data), dim =-3)
        target_probs = label
        if not self.is_eval():
            self.surr_loss = self.pull_loss(triple, target_probs, int_lengths)
        else:
            self.surr_loss = 0
        return score

    def get_scores(self, embeddings: Dict) -> torch.Tensor:
        p = self._get_triple_score(embeddings['h'], embeddings['t'],
                                   embeddings['r'], embeddings['label'])

        return p

    def get_loss(self, scores: torch.Tensor,
                 label: torch.Tensor) -> torch.Tensor:
        log_p = scores
        log1mp = log1mexp(log_p)
        logits = torch.stack([log1mp, log_p], dim=-1)
        loss = self.loss_f(logits, label) + self.regularization_weight * self.surr_loss

        return loss

    def batch_with_negative_samples(self, **kwargs) -> Dict[str, torch.Tensor]:
        if self.number_of_negative_samples <= 0:
            return kwargs
        head_name, head = self.get_expected_head(kwargs)
        tail_name, tail = self.get_expected_tail(kwargs)
        rel_name, rel = self.get_expected_relation(kwargs)
        label = kwargs.pop('label', None)

        if label is None:
            raise ValueError("Training without labels!")
        # create the tensors for negatives
        #size = self.get_negaive_sample_tensorsize(head.size())
        # for Classification model, we will do it inplace
        multiplier = int(self.number_of_negative_samples / 2)
        size = head.size()[-1]
        head = self.repeat(head, 2 * multiplier + 1)
        tail = self.repeat(tail, 2 * multiplier + 1)
        rel = self.repeat(rel, 2 * multiplier + 1)
        label = self.repeat(label, 2 * multiplier + 1)

        # fill in the random
        self.fill_random_entities_(head[size:size + size * multiplier])
        label[size:size * multiplier + size] = 0
        self.fill_random_entities_(tail[size * (1 + multiplier):])
        label[size * (1 + multiplier):] = 0

        return {'h': head, 't': tail, 'r': rel, 'label': label}


@Model.register('BCE-classification-Torbox-model')
class BCEBoxClassificationModel(BCEBoxModel):
    def __init__(self,
                 num_entities: int,
                 num_relations: int,
                 embedding_dim: int,
                 box_type: str = 'MinDeltaBoxesOnTorus',
                 single_box: bool = False,
                 softbox_temp: float = 10.,
                 number_of_negative_samples: int = 0,
                 debug: bool = False,
                 regularization_weight: float = 0,
                 init_interval_center: float = 0.25,
                 init_interval_delta: float = 0.1,
                 margin: float = 6.0) -> None:
        super().__init__(
            num_entities,
            num_relations,
            embedding_dim,
            box_type=box_type,
            single_box=single_box,
            softbox_temp=softbox_temp,
            number_of_negative_samples=number_of_negative_samples,
            debug=debug,
            regularization_weight=regularization_weight,
            init_interval_center=init_interval_center,
            init_interval_delta=init_interval_delta)
        self.train_f1 = FBetaMeasure(average='micro')
        #self.valid_f1 = FBetaMeasure(average='micro')
        self.threshold_with_f1 = F1WithThreshold(flip_sign=True)

        self.loss_f = torch.nn.NLLLoss(reduction='mean')

    def get_box_embeddings_val(self, h: torch.Tensor, t: torch.Tensor,
                               r: torch.Tensor,
                               label: torch.Tensor) -> Dict[str, BoxTensor]:

        return BaseBoxModel.get_box_embeddings_val(
            self, h=h, t=t, r=r, label=label)

    def get_loss(self, scores: torch.Tensor,
                 label: torch.Tensor) -> torch.Tensor:
        log_p = scores
        log1mp = log1mexp(log_p)
        logits = torch.stack([log1mp, log_p], dim=-1)
        loss = self.loss_f(logits, label) + self.regularization_weight * self.surr_loss

        if not self.is_eval():
            with torch.no_grad():
                self.train_f1(logits, label)

        return loss

    def get_ranks(self, embeddings: Dict[str, BoxTensor]) -> Any:
        s = self._get_triple_score(embeddings['h'], embeddings['t'],
                                   embeddings['r'], embeddings['label'])
        # preds = torch.stack((p_s, n_s), dim=1)  # shape = (batch, 2)
        #self.valid_f1(preds, labels)
        labels = embeddings['label']
        # upate the metrics
        self.threshold_with_f1(s, labels)

        return {}

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        if self.is_eval():
            metrics = self.threshold_with_f1.get_metric(reset)
        else:
            metrics = self.train_f1.get_metric(reset)
            metrics[
                'regularization_loss'] = self.surr_loss.item()

        return metrics
