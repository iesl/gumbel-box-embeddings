from typing import Tuple, Dict, Any, Union
from .base import BaseBoxModel
from .bce_models import BCEBoxClassificationSplitNegVolPenaltyModel
from .relation_transform import BCEBoxClassificationRelationTransformGumbelModel
from typing import Optional
from allennlp.models import Model
from allennlp.training.metrics import Average
from boxes.box_wrapper import SigmoidBoxTensor, BoxTensor, DeltaBoxTensor, DeltaExpBoxTensor
from boxes.modules import BoxEmbedding
from boxes.utils import log1mexp
from allennlp.modules.token_embedders import Embedding
import torch
import torch.nn as nn
import numpy as np
from ..metrics import HitsAt10, F1WithThreshold
from allennlp.training.metrics import F1Measure, FBetaMeasure

@Model.register('BCE-classification-box-to-box-gumbel-box-model')
class BCEBoxClassificationBoxToBoxGumbelModel(BCEBoxClassificationRelationTransformGumbelModel):
    box_types = {'SigmoidBoxTensor': SigmoidBoxTensor,
                 'DeltaBoxTensor': DeltaBoxTensor,
                 'BoxTensor': BoxTensor,
                 'DeltaExpBoxTensor': DeltaExpBoxTensor
                 }

    def __init__(self,
                 num_entities: int,
                 num_relations: int,
                 embedding_dim: int,
                 box_type: str = 'DeltaExpBoxTensor',
                 shared: float = 1.,
                 global_param: float = 1.,
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
        self.global_param = global_param
        self.embedding_dim = embedding_dim
        self.shared = shared
        self.get_relation_embeddings(num_relations, embedding_dim)
        self.get_relation_embeddings_per_entity(num_entities, embedding_dim)

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
            'h_id': h,
            't_id': t,
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
            'h_id': h,
            't_id': t,
            'label': label,
        }

    def get_scores(self, embeddings: Dict) -> torch.Tensor:
        p = self._get_triple_score(embeddings['h'], embeddings['t'],
                                   embeddings['r_h'], embeddings['r_t'],
                                   embeddings['h_id'], embeddings['t_id'])

        return p


    def get_ranks(self, embeddings: Dict[str, BoxTensor]) -> Any:
        if self.is_test():
            return self.get_test(embeddings)

        s = self._get_triple_score(embeddings['h'], embeddings['t'],
                                   embeddings['r_h'], embeddings['r_t'],
                                   embeddings['h_id'], embeddings['t_id'])
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
                                   embeddings['r_h'], embeddings['r_t'],
                                   embeddings['h_id'], embeddings['t_id'])
        labels = embeddings['label']
        pos_prediction = (s > self.test_threshold).float()
        neg_prediction = 1.0 - pos_prediction
        predictions = torch.stack((neg_prediction, pos_prediction), -1)
        self.test_f1(predictions, labels)

        return {}

    def _get_triple_score(self, head: BoxTensor, tail: BoxTensor,
                          relation_head: torch.Tensor, relation_tail: torch.Tensor,
                          head_id: torch.Tensor, tail_id: torch.Tensor) -> torch.Tensor:
        
        head = self.get_relation_transform(head, head_id, relation_head)
        tail = self.get_relation_transform(tail, tail_id, relation_tail)
        intersection_box = head.gumbel_intersection(tail, gumbel_beta=self.gumbel_beta)
        intersection_vol = intersection_box._log_soft_volume_adjusted(intersection_box.z,
            intersection_box.Z, temp=self.softbox_temp, gumbel_beta=self.gumbel_beta)
        tail_vol = tail._log_soft_volume_adjusted(tail.z, tail.Z,
            temp=self.softbox_temp, gumbel_beta=self.gumbel_beta)
        score = intersection_vol - tail_vol
        return score

    def get_relation_embeddings_per_entity(self, num_embeddings: int, embedding_dim: int):
        self.entity_delta_weight = nn.Embedding(num_embeddings=num_embeddings,
                                     embedding_dim=embedding_dim,
                                     sparse=False
                                    )

        self.entity_min_weight = nn.Embedding(num_embeddings=num_embeddings,
                              embedding_dim=embedding_dim,
                              sparse=False
                              )
        nn.init.xavier_uniform_(self.entity_delta_weight.weight.data)
        nn.init.xavier_uniform_(self.entity_min_weight.weight.data)


    def get_relation_embeddings(self, num_embeddings: int, embedding_dim: int):
        self.relation_delta_weight = nn.Embedding(num_embeddings=num_embeddings,
                                     embedding_dim=embedding_dim,
                                     sparse=False
                                    )
        self.relation_min_weight = nn.Embedding(num_embeddings=num_embeddings,
                              embedding_dim=embedding_dim,
                              sparse=False
                              )
        nn.init.xavier_uniform_(self.relation_delta_weight.weight.data)
        nn.init.xavier_uniform_(self.relation_min_weight.weight.data)

    def get_relation_transform(self, box: BoxTensor, entity_id: torch.Tensor, relation: torch.Tensor):
        transformed_box = self.box_types[self.box_type](box.data.clone())
        global_delta = self.relation_delta_weight(relation)
        global_min = self.relation_min_weight(relation)
        per_entity_delta = self.entity_delta_weight(entity_id)
        per_entity_min = self.entity_min_weight(entity_id)
        min_point = transformed_box.data[:,0,:].clone()
        delta = transformed_box.data[:,1,:].clone()
        relation = relation.repeat(self.embedding_dim).view(-1 ,self.embedding_dim)
        transformed_box.data[:,0,:] = min_point + relation * (per_entity_min * (1 - self.shared) + global_min * self.global_param)
        transformed_box.data[:,1,:] = delta + relation * (per_entity_delta * (1 - self.shared) + global_delta * self.global_param)
        
        return transformed_box
