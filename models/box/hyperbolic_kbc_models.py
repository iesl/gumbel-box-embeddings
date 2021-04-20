from typing import Tuple, Dict, Any
from .base import BaseBoxModel
from .bce_models import BCEBoxClassificationModel, BCEDeltaPenaltyModel
from typing import Optional
from allennlp.models import Model
from allennlp.training.metrics import Average
from boxes.box_wrapper import SigmoidBoxTensor, BoxTensor
from allennlp.modules.token_embedders import Embedding
import torch.nn as nn
import torch
import numpy as np
from ..metrics import HitsAt10
from ..utils.mpe_utils import *

@Model.register('MuRP-model')
class MuRPModel(BCEBoxClassificationModel):
    def __init__(
            self,
            num_entities: int,
            num_relations: int,
            embedding_dim: int,
            single_vec: bool = True,
            add_bias: bool = False,
            regularization_weight: float = 0.0,
            number_of_negative_samples: int = 3,
            margin: float = 0.0,
            vocab: Optional[None] = None,
            debug: bool = False ) -> None:

        super().__init__(num_entities, num_relations, embedding_dim, 
                         regularization_weight=regularization_weight,
                         number_of_negative_samples=number_of_negative_samples,
                         )
        self.add_bias = add_bias
        self.Eh = nn.Embedding(
            num_embeddings=num_entities,
            embedding_dim=embedding_dim)
        self.Eh.weight.data = (1e-3 * torch.randn((num_entities, embedding_dim), dtype=torch.double))
        self.Wu = torch.nn.Parameter(torch.tensor(np.random.uniform(-1, 1, (num_relations, 
                                      embedding_dim)), dtype=torch.double, requires_grad=True))
        self.rvh = torch.nn.Embedding(num_relations, embedding_dim, padding_idx=0)
        self.rvh.weight.data = (1e-3 * torch.randn((num_relations, embedding_dim), dtype=torch.double))
        self.bs = torch.nn.Parameter(torch.zeros(num_entities, dtype=torch.double, requires_grad=True))
        self.bo = torch.nn.Parameter(torch.zeros(num_entities, dtype=torch.double, requires_grad=True))
        self.loss_f = torch.nn.BCEWithLogitsLoss()

    def get_box_embeddings_training(  # type:ignore
            self,
            h: torch.Tensor,
            r: torch.Tensor,
            t: torch.Tensor,  # type:ignore
            label: torch.Tensor,
            **kwargs) -> Dict[str, BoxTensor]:  # type: ignore

        return {
            'h': self.Eh(h),
            't': self.Eh(t),
            'Ru': self.Wu[r],
            'rvh': self.rvh(r),
            'bs': self.bs[h],
            'bo':self.bo[t],
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
            'h': self.Eh(h),
            't': self.Eh(t),
            'Ru': self.Wu[r],
            'rvh': self.rvh(r),
            'bs': self.bs[h],
            'bo':self.bo[t],
            'label': label
        }

    def get_scores(self, embeddings: Dict) -> torch.Tensor:
        p = self._get_triple_score(embeddings['h'], embeddings['t'],
                                   embeddings['Ru'], embeddings['rvh'],
                                   embeddings['bs'], embeddings['bo'])

        return p

    def _get_triple_score(self, u: torch.Tensor, v: torch.Tensor,
                          Ru: torch.Tensor, rvh: torch.Tensor,
                          bs: torch.Tensor, bo: torch.Tensor) -> torch.Tensor:

        u = torch.where(torch.norm(u, 2, dim=-1, keepdim=True) >= 1, 
                        u/(torch.norm(u, 2, dim=-1, keepdim=True)-1e-5), u)
        v = torch.where(torch.norm(v, 2, dim=-1, keepdim=True) >= 1, 
                        v/(torch.norm(v, 2, dim=-1, keepdim=True)-1e-5), v)
        rvh = torch.where(torch.norm(rvh, 2, dim=-1, keepdim=True) >= 1, 
                          rvh/(torch.norm(rvh, 2, dim=-1, keepdim=True)-1e-5), rvh) 
        u_e = p_log_map(u)
        u_W = u_e * Ru
        u_m = p_exp_map(u_W)
        v_m = p_sum(v, rvh)
        u_m = torch.where(torch.norm(u_m, 2, dim=-1, keepdim=True) >= 1, 
                          u_m/(torch.norm(u_m, 2, dim=-1, keepdim=True)-1e-5), u_m)
        v_m = torch.where(torch.norm(v_m, 2, dim=-1, keepdim=True) >= 1, 
                          v_m/(torch.norm(v_m, 2, dim=-1, keepdim=True)-1e-5), v_m)
        
        sqdist = (2.*artanh(torch.clamp(torch.norm(p_sum(-u_m, v_m), 2, dim=-1), 1e-10, 1-1e-5)))**2

        if not self.add_bias:
            return - sqdist

        return -sqdist + bs + bo

    def get_loss(self, scores: Tuple[torch.Tensor, torch.Tensor],
                 label: torch.Tensor) -> torch.Tensor:
        # max margin loss expects label to be float
        label = label.to(scores[0].dtype)
        return self.loss_f(scores, label)

    def get_ranks(self, embeddings: Dict[str, BoxTensor]) -> Any:
        if self.is_test():
            return self.get_test(embeddings)

        s = self._get_triple_score(embeddings['h'], embeddings['t'],
                                   embeddings['Ru'], embeddings['rvh'],
                                   embeddings['bs'], embeddings['bo'])
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
            # metrics = self.train_f1.get_metric(reset)
            metrics = {}

        return metrics
