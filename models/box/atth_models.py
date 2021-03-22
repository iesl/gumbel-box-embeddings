from typing import Tuple, Dict, Any
from .base import BaseBoxModel
from .bce_models import BCEBoxClassificationModel, BCEDeltaPenaltyModel
from typing import Optional
from allennlp.models import Model
from allennlp.training.metrics import Average
from boxes.box_wrapper import SigmoidBoxTensor, BoxTensor
from allennlp.modules.token_embedders import Embedding
from boxes.utils import log1mexp
import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from ..metrics import HitsAt10
from ..utils.atth_utils import *

@Model.register('atth-model')
class AttHModel(BCEBoxClassificationModel):
    def __init__(
            self,
            num_entities: int,
            num_relations: int,
            embedding_dim: int,
            regularization_weight: float = 0.0,
            number_of_negative_samples: int = 3,
            add_bias: bool = False,
            margin: float = 0.0,
            vocab: Optional[None] = None,
            debug: bool = False ) -> None:

        super().__init__(num_entities, num_relations, embedding_dim, 
                         regularization_weight=regularization_weight,
                         number_of_negative_samples=number_of_negative_samples,
                         )
        self.add_bias = add_bias
        self.entity = nn.Embedding(num_embeddings=num_entities, embedding_dim=embedding_dim)
        self.rel = nn.Embedding(num_embeddings=num_relations, embedding_dim=2*embedding_dim)
        self.entity.weight.data = torch.randn((num_entities, embedding_dim))
        self.rel.weight.data = torch.randn((num_relations, 2 * embedding_dim))
        self.rel_diag = nn.Embedding(num_relations, 2 * embedding_dim)
        self.rel_diag.weight.data = 2 * torch.rand((num_relations, 2 * embedding_dim)) - 1.0
        self.context_vec = nn.Embedding(num_relations, embedding_dim)
        self.context_vec.weight.data = torch.randn((num_relations, embedding_dim))
        self.act = nn.Softmax(dim=1)
        self.scale = torch.Tensor([1. / np.sqrt(embedding_dim)]).cuda()
        c_init = torch.ones((num_relations, 1))
        self.c = nn.Parameter(c_init, requires_grad=True)
        self.bh = nn.Embedding(num_entities, 1)
        self.bh.weight.data = torch.zeros((num_entities, 1))
        self.bt = nn.Embedding(num_entities, 1)
        self.bt.weight.data = torch.zeros((num_entities, 1))
        self.loss_f = nn.CrossEntropyLoss(reduction='mean')

    def get_box_embeddings_training(  # type:ignore
            self,
            h: torch.Tensor,
            r: torch.Tensor,
            t: torch.Tensor,  # type:ignore
            label: torch.Tensor,
            **kwargs) -> Dict[str, BoxTensor]:  # type: ignore

        return {
            'h': self.entity(h),
            't': self.entity(t),
            'r': self.rel(r),
            'c': F.softplus(self.c[r]),
            'rel_diag': self.rel_diag(r),
            'context_vec': self.context_vec(r),
            'bh': self.bh(h),
            'bt': self.bt(t),
            'label': label
        }

    def get_box_embeddings_val(  # type:ignore
            self,
            h: torch.Tensor,
            r: torch.Tensor,
            t: torch.Tensor,  # type:ignore
            label: torch.Tensor,
            **kwargs) -> Dict[str, BoxTensor]:  # type: ignore

        return {
            'h': self.entity(h),
            't': self.entity(t),
            'r': self.rel(r),
            'c': F.softplus(self.c[r]),
            'rel_diag': self.rel_diag(r),
            'context_vec': self.context_vec(r),
            'bh': self.bh(h),
            'bt': self.bt(t),
            'label': label
        }

    def get_scores(self, embeddings: Dict) -> torch.Tensor:
        p = self._get_triple_score(embeddings['h'], embeddings['t'], embeddings['r'],
                                   embeddings['c'], embeddings['rel_diag'],
                                   embeddings['context_vec'], embeddings['bh'],
                                   embeddings['bt'])

        if self.regularization_weight > 0:
            self.reg_loss = self.get_regularization_penalty_vector(
                                                    embeddings['h'], 
                                                    embeddings['t'],
                                                    embeddings['r'])
        else:
            self.reg_loss = 0.0
        return p

    def get_regularization_penalty_vector(self, h, t, r):
        regul = (torch.mean(h ** 2) + torch.mean(t ** 2) + torch.mean(r ** 2)) / 3
        return regul

    def _get_triple_score(self, head: torch.Tensor, tail: torch.Tensor, r: torch.Tensor,
                          c: torch.Tensor, rel_diag: torch.Tensor,
                          context_vec: torch.Tensor, bh: torch.Tensor,
                          bt: torch.Tensor) -> torch.Tensor:

        rot_mat, ref_mat = torch.chunk(rel_diag, 2, dim=1)
        rot_q = givens_rotations(rot_mat, head).view((-1, 1, self.embedding_dim))
        ref_q = givens_reflection(ref_mat, head).view((-1, 1, self.embedding_dim))
        cands = torch.cat([ref_q, rot_q], dim=1)
        context_vec = context_vec.view((-1, 1, self.embedding_dim))
        att_weights = torch.sum(context_vec * cands * self.scale, dim=-1, keepdim=True)
        att_weights = self.act(att_weights)
        att_q = torch.sum(att_weights * cands, dim=1)
        lhs = expmap0(att_q, c)
        rel, _ = torch.chunk(r, 2, dim=1)
        rel = expmap0(rel, c)
        res = project(mobius_add(lhs, rel, c), c)
        similarity_score = - (hyp_distance_multi_c(res, tail, c) ** 2)

        if not self.add_bias:
            return torch.squeeze(similarity_score)

        return torch.squeeze(similarity_score + bt + bh)

    def get_loss(self, scores: Tuple[torch.Tensor, torch.Tensor],
                 label: torch.Tensor) -> torch.Tensor:
        # max margin loss expects label to be float
        log_p = F.logsigmoid(scores)
        log1mp = log1mexp(log_p)
        if not self.is_eval():
            with torch.no_grad():
                self.train_f1(torch.stack([log1mp, log_p], dim=-1), label)
        label = (2.0 * label) - 1
        return - F.logsigmoid(label * scores).mean() + self.regularization_weight*self.reg_loss


    def get_ranks(self, embeddings: Dict[str, BoxTensor]) -> Any:
        if self.is_test():
            return self.get_test(embeddings)

        s = self._get_triple_score(embeddings['h'], embeddings['t'], embeddings['r'],
                                   embeddings['c'], embeddings['rel_diag'],
                                   embeddings['context_vec'], embeddings['bh'],
                                   embeddings['bt'])
        # preds = torch.stack((p_s, n_s), dim=1)  # shape = (batch, 2)
        #self.valid_f1(preds, labels)
        labels = embeddings['label']
        # upate the metrics
        self.threshold_with_f1(s, labels)

        return {}

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        if self.is_eval():
            if not self.istest:
                metrics = self.threshold_with_f1.get_metric(reset)
            else:
                p, r, f = self.test_f1.get_metric(reset)
                metrics = {'precision': p, 'recall': r, 'fscore': f}
        else:
            metrics = self.train_f1.get_metric(reset)
            # metrics = {}

        return metrics


@Model.register('roth-model')
class RotHModel(AttHModel):
    def __init__(
            self,
            num_entities: int,
            num_relations: int,
            embedding_dim: int,
            regularization_weight: float = 0.0,
            number_of_negative_samples: int = 3,
            add_bias: bool = False,
            margin: float = 0.0,
            vocab: Optional[None] = None,
            debug: bool = False ) -> None:

        super().__init__(num_entities, num_relations, embedding_dim, 
                         regularization_weight=regularization_weight,
                         number_of_negative_samples=number_of_negative_samples,
                         )
        self.rel_diag = nn.Embedding(num_relations, embedding_dim)
        self.rel_diag.weight.data = 2 * torch.rand((num_relations, embedding_dim)) - 1.0
    def _get_triple_score(self, head: torch.Tensor, tail: torch.Tensor, r: torch.Tensor,
                          c: torch.Tensor, rel_diag: torch.Tensor,
                          context_vec: torch.Tensor, bh: torch.Tensor,
                          bt: torch.Tensor) -> torch.Tensor:
        head = expmap0(head, c)
        rel1, rel2 = torch.chunk(r, 2, dim=1)
        rel1 = expmap0(rel1, c)
        rel2 = expmap0(rel2, c)
        lhs = project(mobius_add(head, rel1, c), c)
        res1 = givens_rotations(rel_diag, lhs)
        res2 = mobius_add(res1, rel2, c)
        similarity_score = - (hyp_distance_multi_c(res2, tail, c) ** 2)

        if not self.add_bias:
            return torch.squeeze(similarity_score)

        return torch.squeeze(similarity_score + bt + bh)

@Model.register('rote-model')
class RotEModel(RotHModel):
    def __init__(
            self,
            num_entities: int,
            num_relations: int,
            embedding_dim: int,
            regularization_weight: float = 0.0,
            number_of_negative_samples: int = 3,
            add_bias: bool = False,
            margin: float = 0.0,
            vocab: Optional[None] = None,
            debug: bool = False ) -> None:

        super().__init__(num_entities, num_relations, embedding_dim, 
                         regularization_weight=regularization_weight,
                         number_of_negative_samples=number_of_negative_samples,
                         )
        self.rel = nn.Embedding(num_relations, embedding_dim)
        self.rel.weight.data = torch.randn((num_relations, embedding_dim))
    def _get_triple_score(self, head: torch.Tensor, tail: torch.Tensor, r: torch.Tensor,
                          c: torch.Tensor, rel_diag: torch.Tensor,
                          context_vec: torch.Tensor, bh: torch.Tensor,
                          bt: torch.Tensor) -> torch.Tensor:
        lhs = givens_rotations(rel_diag, head) + r
        similarity_score = - euc_sqdistance(lhs, tail)

        if not self.add_bias:
            return  torch.squeeze(similarity_score)

        return  torch.squeeze(similarity_score + bt + bh)
