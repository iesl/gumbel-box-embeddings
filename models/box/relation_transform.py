from typing import Tuple, Dict, Any, Union
from .base import BaseBoxModel
from .max_margin_models import MaxMarginBoxModel, MaxMarginConditionalModel, MaxMarginConditionalClassificationModel
from .bce_models import BCEBoxClassificationSplitNegVolPenaltyModel
from typing import Optional
from allennlp.models import Model
from allennlp.training.metrics import Average
from boxes.box_wrapper import SigmoidBoxTensor, BoxTensor, DeltaBoxTensor
from boxes.modules import BoxEmbedding
from boxes.utils import log1mexp
from allennlp.modules.token_embedders import Embedding
import torch
import torch.nn as nn
import numpy as np
from ..metrics import HitsAt10, F1WithThreshold
from allennlp.training.metrics import F1Measure, FBetaMeasure

@Model.register('BCE-classification-relation_transform-box-model')
class BCEBoxClassificationRelationTransformModel(BCEBoxClassificationSplitNegVolPenaltyModel):
    box_types = {'SigmoidBoxTensor': SigmoidBoxTensor,
                 'DeltaBoxTensor': DeltaBoxTensor,
                 'BoxTensor': BoxTensor
                 }

    def __init__(self,
                 num_entities: int,
                 num_relations: int,
                 embedding_dim: int,
                 box_type: str = 'SigmoidBoxTensor',
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
            number_of_negative_samples_head=number_of_negative_samples,
            number_of_negative_samples_tail=number_of_negative_samples,
            debug=debug,
            regularization_weight=regularization_weight,
            init_interval_center=init_interval_center,
            init_interval_delta=init_interval_delta)

        self.embedding_dim = embedding_dim
        self.get_relation_embeddings(num_relations, embedding_dim)
        try:
            self.box = self.box_types[box_type]
        except KeyError as ke:
            raise ValueError("Invalid box type {}".format(box_type)) from ke


    def batch_with_negative_samples(self, **kwargs) -> Dict[str, torch.Tensor]:
        if self.number_of_negative_samples <= 0:
            return kwargs
        head_name, head = self.get_expected_head(kwargs)
        tail_name, tail = self.get_expected_tail(kwargs)
        rel_head_name = 'r_h'
        rel_tail_name = 'r_t'
        rel_head = kwargs[rel_head_name]
        rel_tail = kwargs[rel_tail_name]
        label = kwargs.pop('label', None)

        if label is None:
            raise ValueError("Training without labels!")
        # create the tensors for negatives
        #size = self.get_negaive_sample_tensorsize(head.size())
        # for Classification model, we will do it inplace
        multiplier = int(self.number_of_negative_samples)
        size = head.size()[-1]
        head = self.repeat(head, multiplier + 1)
        tail = self.repeat(tail, multiplier + 1)
        rel_head = self.repeat(rel_head, multiplier + 1)
        rel_tail = self.repeat(rel_tail, multiplier + 1)
        label = self.repeat(label, multiplier + 1)

        # fill in the random
        head_multiplier = int(self.number_of_negative_samples_head)
        self.fill_random_entities_(head[size:size + size * head_multiplier])
        tail_multiplier = int(self.number_of_negative_samples_tail)

        if tail_multiplier > 0:
            self.fill_random_entities_(
                tail[size + size * head_multiplier:size +
                     size * head_multiplier + size * tail_multiplier])
        label[size:size * multiplier + size] = 0

        return {'h': head, 't': tail, 'r_h': rel_head, 'r_t': rel_tail, 'label': label}

    def get_box_embeddings_training(  # type:ignore
            self,
            h: torch.Tensor,
            r_h: torch.Tensor,
            r_t: torch.Tensor,
            t: torch.Tensor,  # type:ignore

            label: torch.Tensor,
            **kwargs) -> Dict[str, BoxTensor]:  # type: ignore

        return {
            'h': self.h(h),
            't': self.t(t),
            'r_h': r_h,
            'r_t': r_t,
            'label': label,
        }

    def get_box_embeddings_val(self,
            h: torch.Tensor,
            r_h: torch.Tensor,
            r_t: torch.Tensor,
            t: torch.Tensor,  # type:ignore

            label: torch.Tensor,
            **kwargs) -> Dict[str, BoxTensor]: # type: ignore

        return {
            'h': self.h(h),
            't': self.t(t),
            'r_h': r_h,
            'r_t': r_t,
            'label': label,
        }

    def get_ranks(self, embeddings: Dict[str, BoxTensor]) -> Any:
        if self.is_test():
            return self.get_test(embeddings)

        s = self._get_triple_score(embeddings['h'], embeddings['t'],
                                   embeddings['r_h'], embeddings['r_t'])
        # preds = torch.stack((p_s, n_s), dim=1)  # shape = (batch, 2)
        #self.valid_f1(preds, labels)
        labels = embeddings['label']
        # upate the metrics
        self.threshold_with_f1(s, labels)

        return {}

    def get_test(self, embeddings: Dict[str, BoxTensor]) -> Any:
        if self.test_threshold is None:
            raise RuntimeError("test_threshold should be set")
        s = self._get_triple_score(embeddings['h'], embeddings['t'],
                                   embeddings['r_h'], embeddings['r_t'])
        labels = embeddings['label']
        pos_prediction = (s > self.test_threshold).float()
        neg_prediction = 1.0 - pos_prediction
        predictions = torch.stack((neg_prediction, pos_prediction), -1)
        self.test_f1(predictions, labels)

        return {}

    def get_relation_embeddings(self, num_embeddings: int, embedding_dim: int):
        self.relation_delta_weight = nn.Embedding(num_embeddings=num_embeddings,
                              embedding_dim=embedding_dim,
                              sparse=False
                              )
        self.relation_delta_bias = nn.Embedding(num_embeddings=num_embeddings,
                                   embedding_dim=embedding_dim,
                                   sparse= False)
        self.relation_min_weight = nn.Embedding(num_embeddings=num_embeddings,
                              embedding_dim=embedding_dim,
                              sparse=False
                              )
        self.relation_min_bias = nn.Embedding(num_embeddings=num_embeddings,
                                   embedding_dim=embedding_dim,
                                   sparse= False)
        nn.init.xavier_uniform_(self.relation_delta_weight.weight.data)
        nn.init.xavier_uniform_(self.relation_delta_bias.weight.data)
        nn.init.xavier_uniform_(self.relation_min_weight.weight.data)
        nn.init.xavier_uniform_(self.relation_min_bias.weight.data)

    def get_relation_transform(self, box: BoxTensor, relation: torch.Tensor):
        transformed_box = self.box_types[self.box_type](box.data.clone())
        weight_delta = self.relation_delta_weight(relation)
        weight_min = self.relation_min_weight(relation)
        bias_delta = self.relation_delta_bias(relation)
        bias_min = self.relation_min_bias(relation)
        
        min_point = transformed_box.data[:,0,:].clone()
        delta = transformed_box.data[:,1,:].clone()
        transformed_box.data[:,0,:] = min_point * weight_min + bias_min
        transformed_box.data[:,1,:] = nn.functional.softplus(delta * weight_delta) + bias_delta
        return transformed_box

    def _get_triple_score(self, head: BoxTensor, tail: BoxTensor,
                          relation_head: torch.Tensor, relation_tail: torch.Tensor) -> torch.Tensor:
        head = self.get_relation_transform(box=head, relation=relation_head)
        tail = self.get_relation_transform(tail, relation_tail)
        head_tail_box_vol = head.intersection_log_soft_volume(
            tail, temp=self.softbox_temp)
        score = head_tail_box_vol - tail.log_soft_volume(temp=self.softbox_temp)
        return score

    def get_scores(self, embeddings: Dict) -> torch.Tensor:
        p = self._get_triple_score(embeddings['h'], embeddings['t'],
                                   embeddings['r_h'], embeddings['r_t'])

        return p


@Model.register('BCE-classification-relation_transform-gumbel-box-model')
class BCEBoxClassificationRelationTransformGumbelModel(BCEBoxClassificationRelationTransformModel):
    box_types = {'SigmoidBoxTensor': SigmoidBoxTensor,
                 'DeltaBoxTensor': DeltaBoxTensor,
                 'BoxTensor': BoxTensor
                 }

    def __init__(self,
                 num_entities: int,
                 num_relations: int,
                 embedding_dim: int,
                 box_type: str = 'SigmoidBoxTensor',
                 single_box: bool = False,
                 softbox_temp: float = 10.,
                 number_of_negative_samples: int = 0,
                 debug: bool = False,
                 regularization_weight: float = 0,
                 init_interval_center: float = 0.25,
                 init_interval_delta: float = 0.1,
                 gumbel_beta: float = 1.0) -> None:
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
        self.gumbel_beta = gumbel_beta

    def _get_triple_score(self, head: BoxTensor, tail: BoxTensor,
                          relation_head: torch.Tensor, relation_tail: torch.Tensor) -> torch.Tensor:
        head = self.get_relation_transform(box=head, relation=relation_head)
        tail = self.get_relation_transform(tail, relation_tail)
        intersection_box = head.gumbel_intersection(tail, gumbel_beta=self.gumbel_beta)
        intersection_vol = intersection_box._log_soft_volume_adjusted(intersection_box.z,
            intersection_box.Z, temp=self.softbox_temp, gumbel_beta=self.gumbel_beta)
        tail_vol = tail._log_soft_volume_adjusted(tail.z, tail.Z,
            temp=self.softbox_temp, gumbel_beta=self.gumbel_beta)
        score = intersection_vol - tail_vol
        return score

@Model.register('re-BCE-classification-relation_transform-gumbel-box-model')
class BCEBoxClassificationRelationTransformGumbelRevisedModel(BCEBoxClassificationRelationTransformGumbelModel):
    box_types = {'SigmoidBoxTensor': SigmoidBoxTensor,
                 'DeltaBoxTensor': DeltaBoxTensor,
                 'BoxTensor': BoxTensor
                 }

    def _get_triple_score(self, head: BoxTensor, tail: BoxTensor,
                          relation_head: torch.Tensor, relation_tail: torch.Tensor) -> torch.Tensor:
        head_transformed = self.get_relation_transform(box=head, relation=relation_head)
        tail_transformed = self.get_relation_transform(tail, relation_tail)
        intersection_box = head_transformed.gumbel_intersection(tail_transformed, gumbel_beta=self.gumbel_beta)
        intersection_vol = intersection_box._log_soft_volume_adjusted(intersection_box.z,
            intersection_box.Z, temp=self.softbox_temp, gumbel_beta=self.gumbel_beta)
        tail_vol = tail_transformed._log_soft_volume_adjusted(tail_transformed.z, tail_transformed.Z,
            temp=self.softbox_temp, gumbel_beta=self.gumbel_beta)
        score = intersection_vol - tail_vol
        return score
