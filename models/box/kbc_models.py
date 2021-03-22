from typing import Tuple, Dict, Any, Union
from .base import BaseBoxModel
from .bce_models import BCEBoxModel, BCEDeltaPenaltyModel
from .max_margin_models import MaxMarginBoxModel, MaxMarginConditionalModel
from .gumbel_bce_box import BCEBesselApproxModel, BCEBesselApproxClassificationModel
from typing import Optional
from allennlp.models import Model
from allennlp.training.metrics import Average
from boxes.box_wrapper import SigmoidBoxTensor, BoxTensor, DeltaBoxTensor
from boxes.modules import BoxEmbedding
from boxes.utils import log1mexp
from allennlp.modules.token_embedders import Embedding
import torch
from torch import nn
from torch.distributions.gumbel import Gumbel
from ..metrics import HitsAt10, F1WithThreshold
from allennlp.training.metrics import F1Measure, FBetaMeasure
from copy import deepcopy


@Model.register('BCE-sym-gumbel-kbc-model')
class BCESymmetricGumbelKbcModel(BCEBesselApproxModel):
    def _get_triple_score(self,
                          head: BoxTensor,
                          tail: BoxTensor,
                          relation: BoxTensor) -> torch.Tensor:
        if self.is_eval():
            if len(head.data.shape) > len(tail.data.shape):
                tail.data = torch.cat(head.data.shape[-3]*[tail.data])
                relation.data = torch.cat(head.data.shape[-3]*[relation.data])
            elif len(head.data.shape) < len(tail.data.shape):
                head.data = torch.cat(tail.data.shape[-3]*[head.data])
                relation.data = torch.cat(tail.data.shape[-3]*[relation.data])

        head_relation_box = relation.gumbel_intersection(head, gumbel_beta=self.gumbel_beta)
        tail_relation_box = relation.gumbel_intersection(tail, gumbel_beta=self.gumbel_beta)
        tail_head_relation_box = tail_relation_box.gumbel_intersection(head, gumbel_beta=self.gumbel_beta)
       
        tail_head_relation_box_vol = tail_head_relation_box._log_soft_volume_adjusted(tail_head_relation_box.z,
        	tail_head_relation_box.Z, temp=self.softbox_temp, gumbel_beta=self.gumbel_beta)
         
        tail_relation_box_vol = tail_relation_box._log_soft_volume_adjusted(tail_relation_box.z,
        	tail_relation_box.Z, temp=self.softbox_temp, gumbel_beta=self.gumbel_beta)
        score_head = tail_head_relation_box_vol - tail_relation_box_vol
        
        head_relation_box_vol = head_relation_box._log_soft_volume_adjusted(head_relation_box.z,
        	head_relation_box.Z, temp=self.softbox_temp, gumbel_beta=self.gumbel_beta)
        score_tail = tail_head_relation_box_vol - head_relation_box_vol

        return (score_head + score_tail)/2


@Model.register('BCE-asym-gumbel-kbc-model')
class BCEAsymmetricGumbelKbcModel(BCEBesselApproxModel):
    def _get_triple_score(self,
                          head: BoxTensor,
                          tail: BoxTensor,
                          relation: BoxTensor) -> torch.Tensor:
        if self.is_eval():
            if len(head.data.shape) > len(tail.data.shape):
                tail.data = torch.cat(head.data.shape[-3]*[tail.data])
                relation.data = torch.cat(head.data.shape[-3]*[relation.data])
            elif len(head.data.shape) < len(tail.data.shape):
                head.data = torch.cat(tail.data.shape[-3]*[head.data])
                relation.data = torch.cat(tail.data.shape[-3]*[relation.data])

        tail_relation_box = relation.gumbel_intersection(tail, gumbel_beta=self.gumbel_beta)
        tail_head_relation_box = tail_relation_box.gumbel_intersection(head, gumbel_beta=self.gumbel_beta)
       
        tail_head_relation_box_vol = tail_head_relation_box._log_soft_volume_adjusted(tail_head_relation_box.z,
            tail_head_relation_box.Z, temp=self.softbox_temp, gumbel_beta=self.gumbel_beta)
        tail_vol = tail._log_soft_volume_adjusted(tail.z, tail.Z, temp=self.softbox_temp, gumbel_beta=self.gumbel_beta) 
        score_head = tail_head_relation_box_vol - tail_vol

        return score_head

@Model.register('BCE-asym-gumbel-classification-kbc-model')
class BCEAsymmetricGumbelKbcModel(BCEBesselApproxClassificationModel):
    def _get_triple_score(self,
                          head: BoxTensor,
                          tail: BoxTensor,
                          relation: BoxTensor) -> torch.Tensor:
        if self.is_eval():
            if len(head.data.shape) > len(tail.data.shape):
                tail.data = torch.cat(head.data.shape[-3]*[tail.data])
                relation.data = torch.cat(head.data.shape[-3]*[relation.data])
            elif len(head.data.shape) < len(tail.data.shape):
                head.data = torch.cat(tail.data.shape[-3]*[head.data])
                relation.data = torch.cat(tail.data.shape[-3]*[relation.data])

        tail_relation_box = relation.gumbel_intersection(tail, gumbel_beta=self.gumbel_beta)
        tail_head_relation_box = tail_relation_box.gumbel_intersection(head, gumbel_beta=self.gumbel_beta)
       
        tail_head_relation_box_vol = tail_head_relation_box._log_soft_volume_adjusted(tail_head_relation_box.z,
            tail_head_relation_box.Z, temp=self.softbox_temp, gumbel_beta=self.gumbel_beta)
        tail_vol = tail._log_soft_volume_adjusted(tail.z, tail.Z, temp=self.softbox_temp, gumbel_beta=self.gumbel_beta) 
        score_head = tail_head_relation_box_vol - tail_vol

        return score_head 


@Model.register('MaxMargin-sym-gumbel-kbc-model')
class MaxMarginSymmetricGumbelKbcModel(MaxMarginBoxModel):
    def __init__(self,
                 num_entities: int,
                 num_relations: int,
                 embedding_dim: int,
                 box_type: str = 'SigmoidBoxTensor',
                 margin: float = 5.0,
                 single_box: bool = False,
                 softbox_temp: float = 10.,
                 number_of_negative_samples: int = 0,
                 debug: bool = False,
                 regularization_weight: float = 0,
                 init_interval_center: float = 0.25,
                 init_interval_delta: float = 0.1,
                 neg_samples_in_dataset_reader: int = 0,
                 gumbel_beta: float = 0.1) -> None:
        self.gumbel_beta = gumbel_beta
        super().__init__(
            num_entities,
            num_relations,
            embedding_dim,
            box_type=box_type,
            single_box=single_box,
            softbox_temp=softbox_temp,
            margin=margin,
            number_of_negative_samples=number_of_negative_samples,
            debug=debug,
            regularization_weight=regularization_weight,
            init_interval_center=init_interval_center,
            init_interval_delta=init_interval_delta)

    def _get_triple_score(self, head: BoxTensor, tail: BoxTensor, relation: BoxTensor) -> torch.Tensor:
        if self.is_eval():
            if len(head.data.shape) > len(tail.data.shape):
                tail.data = torch.cat(head.data.shape[-3]*[tail.data])
                relation.data = torch.cat(head.data.shape[-3]*[relation.data])
            elif len(head.data.shape) < len(tail.data.shape):
                head.data = torch.cat(tail.data.shape[-3]*[head.data])
                relation.data = torch.cat(tail.data.shape[-3]*[relation.data])

        head_relation_box = relation.gumbel_intersection(head, gumbel_beta=self.gumbel_beta)
        tail_relation_box = relation.gumbel_intersection(tail, gumbel_beta=self.gumbel_beta)
        tail_head_relation_box = tail_relation_box.gumbel_intersection(head, gumbel_beta=self.gumbel_beta)
   
        tail_head_relation_box_vol = tail_head_relation_box._log_soft_volume_adjusted(tail_head_relation_box.z,
            tail_head_relation_box.Z, temp=self.softbox_temp, gumbel_beta=self.gumbel_beta)

        return tail_head_relation_box_vol


@Model.register('MaxMargin-asym-gumbel-kbc-model')
class MaxMarginAsymmetricGumbelKbcModel(MaxMarginSymmetricGumbelKbcModel):
    def _get_triple_score(self, head: BoxTensor, tail: BoxTensor, relation: BoxTensor) -> torch.Tensor:
        if self.is_eval():
            if len(head.data.shape) > len(tail.data.shape):
                tail.data = torch.cat(head.data.shape[-3]*[tail.data])
                relation.data = torch.cat(head.data.shape[-3]*[relation.data])
            elif len(head.data.shape) < len(tail.data.shape):
                head.data = torch.cat(tail.data.shape[-3]*[head.data])
                relation.data = torch.cat(tail.data.shape[-3]*[relation.data])

        head_relation_box = relation.gumbel_intersection(head, gumbel_beta=self.gumbel_beta)
        tail_relation_box = relation.gumbel_intersection(tail, gumbel_beta=self.gumbel_beta)
        tail_head_relation_box = tail_relation_box.gumbel_intersection(head, gumbel_beta=self.gumbel_beta)
       
        tail_head_relation_box_vol = tail_head_relation_box._log_soft_volume_adjusted(tail_head_relation_box.z,
            tail_head_relation_box.Z, temp=self.softbox_temp, gumbel_beta=self.gumbel_beta)
         
        tail_relation_box_vol = tail_relation_box._log_soft_volume_adjusted(tail_relation_box.z,
            tail_relation_box.Z, temp=self.softbox_temp, gumbel_beta=self.gumbel_beta)
        score_head = tail_head_relation_box_vol - tail_relation_box_vol

        return score_head

@Model.register('BCE-affine-transform-gumbel-kbc-model')
class BCEAffineTransformGumbelKbcModel(BCEBesselApproxModel):
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
                 neg_samples_in_dataset_reader: int = 0,
                 gumbel_beta: float = 1.0,
                 n_samples: int=10) -> None:
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
            init_interval_delta=init_interval_delta,
            neg_samples_in_dataset_reader=neg_samples_in_dataset_reader,
            gumbel_beta=gumbel_beta,
            n_samples=n_samples)
        self.get_relation_embeddings(num_relations, embedding_dim)

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
        weight_delta = self.relation_delta_weight(relation)
        weight_min = self.relation_min_weight(relation)
        bias_delta = self.relation_delta_bias(relation)
        bias_min = self.relation_min_bias(relation)
        if len(box.data.shape) == 3:
           box.data[:,0,:] = box.data[:,0,:].clone() * weight_min + bias_min
           box.data[:,1,:] = nn.functional.softplus(box.data[:,1,:].clone() * weight_delta + bias_delta)
        else:
           box.data[:,:,0,:] = box.data[:,:,0,:].clone() * weight_min + bias_min
           box.data[:,:,1,:] = nn.functional.softplus(box.data[:,:,1,:].clone() * weight_delta + bias_delta)
        return box
    
    def _get_triple_score(self,
                          head: BoxTensor,
                          tail: BoxTensor,
                          relation: BoxTensor) -> torch.Tensor:
        
        transformed_box = self.get_relation_transform(head, relation)

        if self.is_eval():
            if len(head.data.shape) > len(tail.data.shape):
                tail.data = torch.cat(head.data.shape[-3]*[tail.data])
            elif len(head.data.shape) < len(tail.data.shape):
                transformed_box.data = torch.cat(tail.data.shape[-3]*[transformed_box.data])

        intersection_box = transformed_box.gumbel_intersection(tail, gumbel_beta=self.gumbel_beta)
        intersection_vol = intersection_box._log_soft_volume_adjusted(intersection_box.z,
            intersection_box.Z, temp=self.softbox_temp, gumbel_beta=self.gumbel_beta)
        tail_vol = tail._log_soft_volume_adjusted(tail.z, tail.Z,
            temp=self.softbox_temp, gumbel_beta=self.gumbel_beta)

        score = intersection_vol - tail_vol
        return score

    def get_box_embeddings_training(  # type:ignore
            self,
            h: torch.Tensor,
            r: torch.Tensor,
            t: torch.Tensor,  # type:ignore
            label: torch.Tensor,
            **kwargs) -> Dict[str, Union[BoxTensor, torch.Tensor]]:  # type: ignore

        return {
            'h': self.h(h),
            't': self.t(t),
            'r': r,
            'label': label,
        }
    def get_box_embeddings_val(self, hr_t: torch.Tensor, hr_r: torch.Tensor,
                               hr_e: torch.Tensor, tr_h: torch.Tensor,
                               tr_r: torch.Tensor,
                               tr_e: torch.Tensor) -> Dict[str, Union[BoxTensor, torch.Tensor]]:

        if not self.is_eval():
            raise RuntimeError("get_box_embeddings_val called during training")
        with torch.no_grad():
            embs = {
                'hr_t': self.t(hr_t),  # shape=(batch_size, 2, emb_dim)
                'hr_r': hr_r,
                'hr_e': self.h(hr_e),  # shape=(batch_size, *,2,emb_dim)
                'tr_h': self.h(tr_h),
                'tr_r': tr_r,
                'tr_e': self.t(tr_e)  # shape=(*,2,emb_dim)
            }  # batch_size is assumed to be 1 during rank validation

        return embs


@Model.register('BCE-affine-transform-head-tail-gumbel-kbc-model')
class BCEAffineTransformHeadTailGumbelKbcModel(BCEAffineTransformGumbelKbcModel):
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

        self.relation_delta_weight_tail = nn.Embedding(num_embeddings=num_embeddings,
                                                  embedding_dim=embedding_dim,
                                                  sparse=False
                                                )
        self.relation_delta_bias_tail = nn.Embedding(num_embeddings=num_embeddings,
                                                embedding_dim=embedding_dim,
                                                sparse= False)
        self.relation_min_weight_tail = nn.Embedding(num_embeddings=num_embeddings,
                                                embedding_dim=embedding_dim,
                                                sparse=False
                                                )
        self.relation_min_bias_tail = nn.Embedding(num_embeddings=num_embeddings,
                                              embedding_dim=embedding_dim,
                                              sparse= False)
        nn.init.xavier_uniform_(self.relation_delta_weight_tail.weight.data)
        nn.init.xavier_uniform_(self.relation_delta_bias_tail.weight.data)
        nn.init.xavier_uniform_(self.relation_min_weight_tail.weight.data)
        nn.init.xavier_uniform_(self.relation_min_bias_tail.weight.data)

    def get_relation_transform_tail(self, box: BoxTensor, relation: torch.Tensor):
        weight_delta = self.relation_delta_weight_tail(relation)
        weight_min = self.relation_min_weight_tail(relation)
        bias_delta = self.relation_delta_bias_tail(relation)
        bias_min = self.relation_min_bias_tail(relation)
        if len(box.data.shape) == 3:
           box.data[:,0,:] = box.data[:,0,:].clone() * weight_min + bias_min
           box.data[:,1,:] = nn.functional.softplus(box.data[:,1,:].clone() * weight_delta + bias_delta)
        else:
           box.data[:,:,0,:] = box.data[:,:,0,:].clone() * weight_min + bias_min
           box.data[:,:,1,:] = nn.functional.softplus(box.data[:,:,1,:].clone() * weight_delta + bias_delta)
        return box

    def _get_triple_score(self,
                          head: BoxTensor,
                          tail: BoxTensor,
                          relation: BoxTensor) -> torch.Tensor:
        
        transformed_box = self.get_relation_transform(head, relation)
        transformed_box_tail = self.get_relation_transform_tail(tail, relation)

        if self.is_eval():
            if len(head.data.shape) > len(tail.data.shape):
                transformed_box_tail.data = torch.cat(head.data.shape[-3]*[transformed_box_tail.data])
            elif len(head.data.shape) < len(tail.data.shape):
                transformed_box.data = torch.cat(tail.data.shape[-3]*[transformed_box.data])

        intersection_box = transformed_box.gumbel_intersection(transformed_box_tail, gumbel_beta=self.gumbel_beta)
        intersection_vol = intersection_box._log_soft_volume_adjusted(intersection_box.z,
            intersection_box.Z, temp=self.softbox_temp, gumbel_beta=self.gumbel_beta)
        tail_vol = transformed_box_tail._log_soft_volume_adjusted(transformed_box_tail.z, transformed_box_tail.Z,
            temp=self.softbox_temp, gumbel_beta=self.gumbel_beta)

        score = intersection_vol - tail_vol
        return score

@Model.register('BCE-affine-transform-gumbel-prob-kbc-model')
class BCEAffineTransformHeadTailGumbelProbKbcModel(BCEAffineTransformHeadTailGumbelKbcModel):
    def get_box_embeddings_val(self, h: torch.Tensor, t: torch.Tensor,
                              r: torch.Tensor, label: torch.Tensor) -> Dict[str, BoxTensor]:

          if not self.is_eval():
              raise RuntimeError("get_box_embeddings_val called during training")
          with torch.no_grad():
              embs = {
                  'h': self.h(h),  # shape=(batch_size, 2, emb_dim)
                  'r': r,
                  't': self.t(torch.arange(self.num_entities).cuda()),  # shape=(batch_size, *,2,emb_dim
                  't_act': t
              }  # batch_size is assumed to be 1 during rank validation
          return embs

    def _get_triple_score(self,
                          head: BoxTensor,
                          tail: BoxTensor,
                          relation: torch.Tensor) -> torch.Tensor:
        
        transformed_box = self.get_relation_transform(head, relation)
        transformed_box_tail = self.get_relation_transform_tail(tail, relation)

        if self.is_eval():
            if head.data.shape[-3] > tail.data.shape[-3]:
                transformed_box_tail.data = torch.cat(head.data.shape[-3]*[transformed_box_tail.data])
            elif head.data.shape[-3] < tail.data.shape[-3]:
                transformed_box.data = torch.cat(tail.data.shape[-3]*[transformed_box.data])

        intersection_box = transformed_box.gumbel_intersection(transformed_box_tail, gumbel_beta=self.gumbel_beta)
        intersection_vol = intersection_box._log_soft_volume_adjusted(intersection_box.z,
            intersection_box.Z, temp=self.softbox_temp, gumbel_beta=self.gumbel_beta)
        tail_vol = transformed_box_tail._log_soft_volume_adjusted(transformed_box_tail.z, transformed_box_tail.Z,
            temp=self.softbox_temp, gumbel_beta=self.gumbel_beta)

        score = intersection_vol - tail_vol
        return score

    def get_ranks(self, embeddings: Dict[str, BoxTensor]) -> Any:
        if not self.is_eval():
            raise RuntimeError("get_ranks called during training")
        with torch.no_grad():
            # hr_scores = self._get_hr_score(embeddings)\

            tr_scores = self._get_triple_score(embeddings['h'], embeddings['t'],
                                 embeddings['r'])
            higher_elements = torch.sum(tr_scores[0:-1] > tr_scores[embeddings['t_act']])
            ties = torch.sum(tr_scores[0:-1] == tr_scores[embeddings['t_act']])

            # find the spot of zeroth element in the sorted array
            # hr_rank = (
            #     torch.argsort(
            #         hr_scores,
            #         descending=True) == hr_scores.shape[0] - 1  # type:ignore
            # ).nonzero().reshape(-1).item()  # type:ignore
            tr_rank = (higher_elements + ties/2 + 1.0).item()
            # self.head_replacement_rank_avg(hr_rank)
            self.tail_replacement_rank_avg(tr_rank)
            # avg_rank = (hr_rank + tr_rank) / 2.
            avg_rank = tr_rank
            self.avg_rank(avg_rank)
            # self.hitsat10(hr_rank)
            self.hitsat10(tr_rank)
            # self.head_hitsat3(hr_rank)
            self.tail_hitsat3(tr_rank)
            # self.head_hitsat1(hr_rank)
            self.tail_hitsat1(tr_rank)
            # hr_mrr = (1. / hr_rank)
            tr_mrr = (1. / tr_rank)
            # mrr = (hr_mrr + tr_mrr) / 2.
            mrr = tr_mrr
            # self.head_replacement_mrr(hr_mrr)
            self.tail_replacement_mrr(tr_mrr)
            self.mrr(mrr)

            return {
                'hr_rank': 0,
                'tr_rank': tr_rank,
                'avg_rank': avg_rank,
                'hr_mrr': 0,
                'tr_mrr': tr_mrr,
                'int_vol': 0,
                'mrr': mrr
            }


@Model.register('re-BCE-affine-transform-gumbel-kbc-model')
class ReBCEAffineTransformGumbelKbcModel(BCEAffineTransformGumbelKbcModel):
    box_types = {
        'SigmoidBoxTensor': SigmoidBoxTensor,
        'DeltaBoxTensor': DeltaBoxTensor,
        'BoxTensor': BoxTensor
    }
    def get_relation_transform(self, box: BoxTensor, relation: torch.Tensor):

        transformed_box = self.box_types[self.box_type](box.data.clone())
        weight_delta = self.relation_delta_weight(relation)
        weight_min = self.relation_min_weight(relation)
        bias_delta = self.relation_delta_bias(relation)
        bias_min = self.relation_min_bias(relation)
        if len(box.data.shape) == 3:
           transformed_box.data[:,0,:] = transformed_box.data[:,0,:].clone() * weight_min + bias_min
           transformed_box.data[:,1,:] = nn.functional.softplus(transformed_box.data[:,1,:].clone() * weight_delta + bias_delta)
        else:
           transformed_box.data[:,:,0,:] = transformed_box.data[:,:,0,:].clone() * weight_min + bias_min
           transformed_box.data[:,:,1,:] = nn.functional.softplus(transformed_box.data[:,:,1,:].clone() * weight_delta + bias_delta)
        return transformed_box

@Model.register('re-BCE-affine-transform-head-tail-gumbel-kbc-model')
class ReBCEAffineTransformHeadTailGumbelKbcModel(BCEAffineTransformHeadTailGumbelKbcModel):
    box_types = {
        'SigmoidBoxTensor': SigmoidBoxTensor,
        'DeltaBoxTensor': DeltaBoxTensor,
        'BoxTensor': BoxTensor
    }
    def get_relation_transform(self, box: BoxTensor, relation: torch.Tensor):

        transformed_box = self.box_types[self.box_type](box.data.clone())
        weight_delta = self.relation_delta_weight(relation)
        weight_min = self.relation_min_weight(relation)
        bias_delta = self.relation_delta_bias(relation)
        bias_min = self.relation_min_bias(relation)
        if len(box.data.shape) == 3:
           transformed_box.data[:,0,:] = transformed_box.data[:,0,:].clone() * weight_min + bias_min
           transformed_box.data[:,1,:] = nn.functional.softplus(transformed_box.data[:,1,:].clone() * weight_delta + bias_delta)
        else:
           transformed_box.data[:,:,0,:] = transformed_box.data[:,:,0,:].clone() * weight_min + bias_min
           transformed_box.data[:,:,1,:] = nn.functional.softplus(transformed_box.data[:,:,1,:].clone() * weight_delta + bias_delta)
        return transformed_box
    
    def get_relation_transform_tail(self, box: BoxTensor, relation: torch.Tensor):
        transformed_box = self.box_types[self.box_type](box.data.clone())
        weight_delta = self.relation_delta_weight_tail(relation)
        weight_min = self.relation_min_weight_tail(relation)
        bias_delta = self.relation_delta_bias_tail(relation)
        bias_min = self.relation_min_bias_tail(relation)
        if len(box.data.shape) == 3:
           transformed_box.data[:,0,:] = transformed_box.data[:,0,:].clone() * weight_min + bias_min
           transformed_box.data[:,1,:] = nn.functional.softplus(transformed_box.data[:,1,:].clone() * weight_delta + bias_delta)
        else:
           transformed_box.data[:,:,0,:] = transformed_box.data[:,:,0,:].clone() * weight_min + bias_min
           transformed_box.data[:,:,1,:] = nn.functional.softplus(transformed_box.data[:,:,1,:].clone() * weight_delta + bias_delta)
        return transformed_box

@Model.register('BCE-simplE-gumbel-kbc-model')
class BCESimplEGumbelKbcModel(BCEBesselApproxModel):
    def _get_triple_score(self,
                          head: BoxTensor,
                          tail: BoxTensor,
                          relation: BoxTensor,
                          head_rev: BoxTensor,
                          tail_rev: BoxTensor,
                          relation_rev: BoxTensor) -> torch.Tensor:
        if self.is_eval():
            if len(head.data.shape) > len(tail.data.shape):
                tail.data = torch.cat(head.data.shape[-3]*[tail.data])
                relation.data = torch.cat(head.data.shape[-3]*[relation.data])
                relation_rev.data = torch.cat(head.data.shape[-3]*[relation_rev.data])
                head_rev.data =  torch.cat(head.data.shape[-3]*[head_rev.data])
            elif len(head.data.shape) < len(tail.data.shape):
                head.data = torch.cat(tail.data.shape[-3]*[head.data])
                relation.data = torch.cat(tail.data.shape[-3]*[relation.data])
                relation_rev.data = torch.cat(tail.data.shape[-3]*[relation_rev.data])
                tail_rev.data = torch.cat(tail.data.shape[-3]*[tail_rev.data])

        tail_relation_box = relation.gumbel_intersection(tail, gumbel_beta=self.gumbel_beta)
        tail_head_relation_box = tail_relation_box.gumbel_intersection(head, gumbel_beta=self.gumbel_beta)
       
        tail_head_relation_box_vol = tail_head_relation_box._log_soft_volume_adjusted(tail_head_relation_box.z,
            tail_head_relation_box.Z, temp=self.softbox_temp, gumbel_beta=self.gumbel_beta)
        tail_vol = tail._log_soft_volume_adjusted(tail.z, tail.Z, temp=self.softbox_temp, gumbel_beta=self.gumbel_beta) 
        score_fwd = tail_head_relation_box_vol - tail_vol

        tail_relation_box_rev = relation_rev.gumbel_intersection(tail_rev, gumbel_beta=self.gumbel_beta)
        tail_head_relation_box_rev = tail_relation_box_rev.gumbel_intersection(head_rev, gumbel_beta=self.gumbel_beta)
       
        tail_head_relation_box_rev_vol = tail_head_relation_box_rev._log_soft_volume_adjusted(tail_head_relation_box_rev.z,
            tail_head_relation_box_rev.Z, temp=self.softbox_temp, gumbel_beta=self.gumbel_beta)
        tail_rev_vol = tail_rev._log_soft_volume_adjusted(tail_rev.z, tail_rev.Z, temp=self.softbox_temp, gumbel_beta=self.gumbel_beta) 
        score_rev = tail_head_relation_box_rev_vol - tail_rev_vol


        return 0.5*(score_fwd + score_rev)

    def get_box_embeddings_val(self, hr_t: torch.Tensor, hr_r: torch.Tensor,
                               hr_e: torch.Tensor, tr_h: torch.Tensor,
                               tr_r: torch.Tensor,
                               tr_e: torch.Tensor) -> Dict[str, BoxTensor]:

        if not self.is_eval():
            raise RuntimeError("get_box_embeddings_val called during training")
        with torch.no_grad():
            embs = {
                'hr_t': self.t(hr_t),  # shape=(batch_size, 2, emb_dim)
                'hr_r': self.r(hr_r),
                'hr_e': self.h(hr_e),  # shape=(batch_size, *,2,emb_dim)
                'tr_h': self.h(tr_h),
                'tr_r': self.r(tr_r),
                'tr_e': self.t(tr_e),  # shape=(*,2,emb_dim)
                'hr_t_rev': self.t(hr_e),  # shape=(batch_size, 2, emb_dim)
                'hr_r_rev': self.r(hr_r + int(self.num_relations/2)),
                'hr_e_rev': self.h(hr_t),  # shape=(batch_size, *,2,emb_dim)
                'tr_h_rev': self.h(tr_e),
                'tr_r_rev': self.r(tr_r + int(self.num_relations/2)),
                'tr_e_rev': self.t(tr_h),
            }  # batch_size is assumed to be 1 during rank validation

        return embs

    def _get_hr_score(self, embeddings: Dict[str, BoxTensor]) -> torch.Tensor:
        with torch.no_grad():
            b = embeddings
            hr_scores = self._get_triple_score(b['hr_e'], b['hr_t'], b['hr_r'],
                                    b['hr_e_rev'], b['hr_t_rev'], b['hr_r_rev'])

            return hr_scores.reshape(-1)  # flatten

    def _get_tr_score(self, embeddings: Dict[str, BoxTensor]) -> torch.Tensor:
        b = embeddings
        tr_scores = self._get_triple_score(b['tr_h'], b['tr_e'], b['tr_r'],
            b['tr_h_rev'], b['tr_e_rev'], b['tr_r_rev'])

        return tr_scores.reshape(-1)  # flatten

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
            'h_rev': self.h(t),
            't_rev': self.t(h),
            'r_rev': self.r(r + int(self.num_relations/2))
        }
    def get_scores(self, embeddings: Dict) -> torch.Tensor:
        p = self._get_triple_score(embeddings['h'], embeddings['t'], embeddings['r'],
                                    embeddings['h_rev'], embeddings['t_rev'], embeddings['r_rev'])

        return p

@Model.register('BCE-nonlinear-transform-gumbel-kbc-model')
class BCENonLinearTransformGumbelKbcModel(BCEAffineTransformGumbelKbcModel):

    def get_relation_embeddings(self, num_embeddings: int, embedding_dim: int):
        self.relation_delta_weight_1 = nn.Embedding(num_embeddings=num_embeddings,
                                                  embedding_dim=embedding_dim,
                                                  sparse=False
                                                )
        self.relation_delta_bias_1 = nn.Embedding(num_embeddings=num_embeddings,
                                                embedding_dim=embedding_dim,
                                                sparse= False)
        self.relation_min_weight_1 = nn.Embedding(num_embeddings=num_embeddings,
                                                embedding_dim=embedding_dim,
                                                sparse=False
                                                )
        self.relation_min_bias_1 = nn.Embedding(num_embeddings=num_embeddings,
                                              embedding_dim=embedding_dim,
                                              sparse= False)

        self.relation_delta_weight_2 = nn.Embedding(num_embeddings=num_embeddings,
                                                  embedding_dim=embedding_dim,
                                                  sparse=False
                                                )
        self.relation_delta_bias_2 = nn.Embedding(num_embeddings=num_embeddings,
                                                embedding_dim=embedding_dim,
                                                sparse= False)
        self.relation_min_weight_2 = nn.Embedding(num_embeddings=num_embeddings,
                                                embedding_dim=embedding_dim,
                                                sparse=False
                                                )
        self.relation_min_bias_2 = nn.Embedding(num_embeddings=num_embeddings,
                                              embedding_dim=embedding_dim,
                                              sparse= False)
        nn.init.xavier_uniform_(self.relation_delta_weight_1.weight.data)
        nn.init.xavier_uniform_(self.relation_delta_bias_1.weight.data)
        nn.init.xavier_uniform_(self.relation_min_weight_1.weight.data)
        nn.init.xavier_uniform_(self.relation_min_bias_1.weight.data)
        nn.init.xavier_uniform_(self.relation_delta_weight_2.weight.data)
        nn.init.xavier_uniform_(self.relation_delta_bias_2.weight.data)
        nn.init.xavier_uniform_(self.relation_min_weight_2.weight.data)
        nn.init.xavier_uniform_(self.relation_min_bias_2.weight.data)
    
    def get_relation_transform(self, box: BoxTensor, relation: torch.Tensor):
        weight_delta_1 = self.relation_delta_weight_1(relation)
        weight_min_1 = self.relation_min_weight_1(relation)
        bias_delta_1 = self.relation_delta_bias_1(relation)
        bias_min_1 = self.relation_min_bias_1(relation)

        weight_delta_2 = self.relation_delta_weight_2(relation)
        weight_min_2 = self.relation_min_weight_2(relation)
        bias_delta_2 = self.relation_delta_bias_2(relation)
        bias_min_2 = self.relation_min_bias_2(relation)
        
        if len(box.data.shape) == 3:
           box.data[:,0,:] = nn.functional.softplus(box.data[:,0,:].clone() * weight_min_1 + bias_min_1)
           box.data[:,1,:] = nn.functional.softplus(box.data[:,1,:].clone() * weight_delta_1 + bias_delta_1)
           box.data[:,0,:] = box.data[:,0,:].clone() * weight_min_2 + bias_min_2
           box.data[:,1,:] = nn.functional.softplus(box.data[:,1,:].clone() * weight_delta_2 + bias_delta_2)
        else:
           box.data[:,:,0,:] = nn.functional.softplus(box.data[:,:,0,:].clone() * weight_min_1 + bias_min_1)
           box.data[:,:,1,:] = nn.functional.softplus(box.data[:,:,1,:].clone() * weight_delta_1 + bias_delta_1)
           box.data[:,:,0,:] = box.data[:,:,0,:].clone() * weight_min_2 + bias_min_2
           box.data[:,:,1,:] = nn.functional.softplus(box.data[:,:,1,:].clone() * weight_delta_2 + bias_delta_2)
        return box
