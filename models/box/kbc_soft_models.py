from typing import Tuple, Dict, Any, Union

import torch
from allennlp.models import Model

from boxes.box_wrapper import SigmoidBoxTensor, BoxTensor, DeltaBoxTensor
from .kbc_models import BCEAffineTransformGumbelKbcModel, BCESimplEGumbelKbcModel, BCEAffineTransformHeadTailGumbelKbcModel
from .kbc_models import ReBCEAffineTransformHeadTailGumbelKbcModel
from .kbc_models import BCENonLinearTransformGumbelKbcModel


@Model.register('BCE-affine-transform-soft-kbc-model')
class BCEAffineTransformSoftKbcModel(BCEAffineTransformGumbelKbcModel):
    def _get_triple_score(self, head: BoxTensor, tail: BoxTensor,
                          relation: BoxTensor) -> torch.Tensor:
        """ Gets score using conditionals.

        :: note: We do not need to worry about the dimentions of the boxes. If
                it can sensibly broadcast it will.
            """
        transformed_box = self.get_relation_transform(head, relation)
        head_tail_box_vol = transformed_box.intersection_log_soft_volume(
            tail, temp=self.softbox_temp)
        score = head_tail_box_vol - tail.log_soft_volume(
            temp=self.softbox_temp)

        return score

@Model.register('BCE-affine-transform-head-tail-soft-kbc-model')
class BCEAffineTransformHeadTailSoftKbcModel(BCEAffineTransformHeadTailGumbelKbcModel):
    def _get_triple_score(self, head: BoxTensor, tail: BoxTensor,
                          relation: BoxTensor) -> torch.Tensor:
        transformed_box = self.get_relation_transform(head, relation)
        transformed_box_tail = self.get_relation_transform_tail(tail, relation)
        
        head_tail_box_vol = transformed_box.intersection_log_soft_volume(
            transformed_box_tail, temp=self.softbox_temp)
        score = head_tail_box_vol - transformed_box_tail.log_soft_volume(
            temp=self.softbox_temp)

        return score

@Model.register('BCE-affine-transform-soft-prob-kbc-model')
class BCEAffineTransformHeadTailSoftProbKbcModel(BCEAffineTransformHeadTailSoftKbcModel):
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

@Model.register('BCE-nonlinear-transform-soft-kbc-model')
class BCENonLinearTransformSoftKbcModel(BCENonLinearTransformGumbelKbcModel):
    def _get_triple_score(self, head: BoxTensor, tail: BoxTensor,
                          relation: BoxTensor) -> torch.Tensor:
        """ Gets score using conditionals.

        :: note: We do not need to worry about the dimentions of the boxes. If
                it can sensibly broadcast it will.
            """
        transformed_box = self.get_relation_transform(head, relation)
        head_tail_box_vol = transformed_box.intersection_log_soft_volume(
            tail, temp=self.softbox_temp)
        score = head_tail_box_vol - tail.log_soft_volume(
            temp=self.softbox_temp)

        return score

@Model.register('re-BCE-affine-transform-head-tail-soft-kbc-model')
class ReBCEAffineTransformHeadTailSoftKbcModel(ReBCEAffineTransformHeadTailGumbelKbcModel):
    def _get_triple_score(self, head: BoxTensor, tail: BoxTensor,
                          relation: BoxTensor) -> torch.Tensor:
        transformed_box = self.get_relation_transform(head, relation)
        transformed_box_tail = self.get_relation_transform_tail(tail, relation)
        
        head_tail_box_vol = transformed_box.intersection_log_soft_volume(
            transformed_box_tail, temp=self.softbox_temp)
        score = head_tail_box_vol - transformed_box_tail.log_soft_volume(
            temp=self.softbox_temp)

        return score


@Model.register('BCE-simplE-Soft-kbc-model')
class BCESimplESoftKbcModel(BCESimplEGumbelKbcModel):
    def _get_triple_score(self,
                          head: BoxTensor,
                          tail: BoxTensor,
                          relation: BoxTensor,
                          head_rev: BoxTensor,
                          tail_rev: BoxTensor,
                          relation_rev: BoxTensor) -> torch.Tensor:


        tail_relation_box = relation.intersection(tail)
        tail_head_relation_box_vol = tail_relation_box.intersection_log_soft_volume(
            head, temp=self.softbox_temp)
        tail_vol = tail.log_soft_volume(temp=self.softbox_temp) 
        score_fwd = tail_head_relation_box_vol - tail_vol

        tail_relation_box_rev = relation_rev.intersection(tail_rev)
        tail_head_relation_box_rev_vol = tail_relation_box_rev.intersection_soft_volume(head_rev, temp=self.softbox_temp)

        tail_rev_vol = tail_rev.log_soft_volume(temp=self.softbox_temp)
        score_rev = tail_head_relation_box_rev_vol - tail_rev_vol


        return 0.5*(score_fwd + score_rev)
