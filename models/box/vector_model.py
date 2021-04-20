from typing import Tuple, Dict, Any
from .base import BaseBoxModel
from .max_margin_models import MaxMarginConditionalClassificationModel, MaxMarginConditionalModel
from typing import Optional
from allennlp.models import Model
from allennlp.training.metrics import Average
from boxes.box_wrapper import SigmoidBoxTensor, BoxTensor
from allennlp.modules.token_embedders import Embedding
import torch.nn as nn
import torch
import numpy as np
from ..metrics import HitsAt10
from ..utils.metrics import single_rank


@Model.register('transE-model')
class TransEModel(MaxMarginConditionalClassificationModel):
    def get_r_vol(self) -> torch.Tensor:
        with torch.no_grad():
            v = 0

        return v
    def get_h_vol(self) -> torch.Tensor:
        with torch.no_grad():
            v = 0

        return v
    def get_t_vol(self) -> torch.Tensor:
        with torch.no_grad():
            v = 0

        return v
    def create_embeddings_layer(self, num_entities: int, num_relations: int,
                                embedding_dim: int, single_vec: bool,
                                entities_init_interval_center: float,
                                entities_init_interval_delta: float,
                                relations_init_interval_center: float,
                                relations_init_interval_delta: float) -> None:
        self.h = nn.Embedding(
            num_embeddings=num_entities,
            embedding_dim=embedding_dim)

        if not single_vec:
            self.t = nn.Embedding(
                num_embeddings=num_entities,
                embedding_dim=embedding_dim)
        else:
            self.t = self.h

        self.r = nn.Embedding(
            num_embeddings=num_relations,
            embedding_dim=embedding_dim)
        nn.init.xavier_uniform_(self.h.weight.data)
        nn.init.xavier_uniform_(self.t.weight.data)
        nn.init.xavier_uniform_(self.r.weight.data)

        self.appropriate_emb = {
            'p_h': self.h,
            'n_h': self.h,
            'h': self.h,
            'tr_h': self.h,
            'hr_e': self.h,
            'p_t': self.t,
            'n_t': self.t,
            't': self.t,
            'hr_t': self.t,
            'tr_e': self.t,
            'p_r': self.r,
            'n_r': self.r,
            'r': self.r,
            'hr_r': self.r,
            'tr_r': self.r,
            'label': (lambda x: x)
        }


    def __init__(
            self,
            num_entities: int,
            num_relations: int,
            embedding_dim: int,
            single_vec: bool = True,
            regularization_weight: float = 0.0,
            number_of_negative_samples: int = 3,
            margin: float = 0.0,
            vocab: Optional[None] = None,
            debug: bool = False
            # we don't need vocab but some api relies on its presence as an argument
    ) -> None:
        super().__init__(num_entities, num_relations, embedding_dim, 
                         regularization_weight=regularization_weight,
                         number_of_negative_samples=number_of_negative_samples)
        self.create_embeddings_layer(num_entities,
                                     num_relations,
                                     embedding_dim, single_vec, 0.5, 0.5, 0.5, 0.5)
        self.loss_f: torch.nn.modules._Loss = torch.nn.MarginRankingLoss(  # type: ignore
            margin=margin, reduction='mean')
        self.margin = margin

    def get_scores(self,
                   embeddings: Dict) -> Tuple[torch.Tensor, torch.Tensor]:

        p_s = self._get_triple_score(embeddings['p_h'], embeddings['p_t'],
                                     embeddings['p_r'])

        n_s = self._get_triple_score(embeddings['n_h'], embeddings['n_t'],
                                     embeddings['n_r'])
        if self.regularization_weight > 0:
            self.reg_loss = self.get_regularization_penalty_vector(
                                                    embeddings['p_h'], 
                                                    embeddings['p_t'],
                                                    embeddings['p_r'])
        else:
            self.reg_loss = torch.tensor(0)
        return (p_s, n_s)

    def get_regularization_penalty(self):
        return self.regularization_weight*self.reg_loss

    def _get_triple_score(self, head: torch.Tensor, tail: torch.Tensor,
                          relation: torch.Tensor) -> torch.Tensor:
        """ Gets score using three way intersection

        We do not need to worry about the dimentions of the boxes. If
            it can sensibly broadcast it will.
        """
        return -torch.norm(head + relation - tail, p='fro', dim=-1)

    def get_loss(self, scores: Tuple[torch.Tensor, torch.Tensor],
                 label: torch.Tensor) -> torch.Tensor:
        # max margin loss expects label to be float
        label = label.to(scores[0].dtype)
        return self.loss_f(*scores, label) + self.regularization_weight*self.reg_loss

    def get_regularization_penalty_vector(self, h, t, r):
        regul = (torch.mean(h ** 2) + torch.mean(t ** 2) + torch.mean(r ** 2)) / 3
        return regul

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
            metrics[
                'regularization_loss'] = self.reg_loss.item()

        return metrics

@Model.register('transE-model-ranking')
class TransEModelRanking(MaxMarginConditionalModel):
    def get_r_vol(self) -> torch.Tensor:
        with torch.no_grad():
            v = 0

        return v
    def get_h_vol(self) -> torch.Tensor:
        with torch.no_grad():
            v = 0

        return v
    def get_t_vol(self) -> torch.Tensor:
        with torch.no_grad():
            v = 0

        return v
    def create_embeddings_layer(self, num_entities: int, num_relations: int,
                                embedding_dim: int, single_vec: bool,
                                entities_init_interval_center: float,
                                entities_init_interval_delta: float,
                                relations_init_interval_center: float,
                                relations_init_interval_delta: float) -> None:
        self.h = nn.Embedding(
            num_embeddings=num_entities,
            embedding_dim=embedding_dim)

        if not single_vec:
            self.t = nn.Embedding(
                num_embeddings=num_entities,
                embedding_dim=embedding_dim)
        else:
            self.t = self.h

        self.r = nn.Embedding(
            num_embeddings=num_relations,
            embedding_dim=embedding_dim)
        nn.init.xavier_uniform_(self.h.weight.data)
        nn.init.xavier_uniform_(self.t.weight.data)
        nn.init.xavier_uniform_(self.r.weight.data)

        self.appropriate_emb = {
            'p_h': self.h,
            'n_h': self.h,
            'h': self.h,
            'tr_h': self.h,
            'hr_e': self.h,
            'p_t': self.t,
            'n_t': self.t,
            't': self.t,
            'hr_t': self.t,
            'tr_e': self.t,
            'p_r': self.r,
            'n_r': self.r,
            'r': self.r,
            'hr_r': self.r,
            'tr_r': self.r,
            'label': (lambda x: x)
        }


    def __init__(
            self,
            num_entities: int,
            num_relations: int,
            embedding_dim: int,
            single_vec: bool = True,
            regularization_weight: float = 0.0,
            number_of_negative_samples: int = 3,
            margin: float = 0.0,
            vocab: Optional[None] = None,
            debug: bool = False
            # we don't need vocab but some api relies on its presence as an argument
    ) -> None:
        super().__init__(num_entities, num_relations, embedding_dim, 
                         regularization_weight=regularization_weight,
                         number_of_negative_samples=number_of_negative_samples)
        self.create_embeddings_layer(num_entities,
                                     num_relations,
                                     embedding_dim, single_vec, 0.5, 0.5, 0.5, 0.5)
        self.loss_f: torch.nn.modules._Loss = torch.nn.MarginRankingLoss(  # type: ignore
            margin=margin, reduction='mean')
        self.margin = margin

    def get_scores(self,
                   embeddings: Dict) -> Tuple[torch.Tensor, torch.Tensor]:

        p_s = self._get_triple_score(embeddings['p_h'], embeddings['p_t'],
                                     embeddings['p_r'])

        n_s = self._get_triple_score(embeddings['n_h'], embeddings['n_t'],
                                     embeddings['n_r'])
        if self.regularization_weight > 0:
            self.reg_loss = self.get_regularization_penalty_vector(
                                                    embeddings['p_h'], 
                                                    embeddings['p_t'],
                                                    embeddings['p_r'])
        else:
            self.reg_loss = 0
        return (p_s, n_s)

    def get_regularization_penalty(self):
        return self.regularization_weight*self.reg_loss

    def _get_triple_score(self, head: torch.Tensor, tail: torch.Tensor,
                          relation: torch.Tensor) -> torch.Tensor:
        """ Gets score using three way intersection

        We do not need to worry about the dimentions of the boxes. If
            it can sensibly broadcast it will.
        """
        return -torch.norm(head + relation - tail, p='fro', dim=-1) 


    def get_loss(self, scores: Tuple[torch.Tensor, torch.Tensor],
                 label: torch.Tensor) -> torch.Tensor:
        # max margin loss expects label to be float
        label = label.to(scores[0].dtype)
        return self.loss_f(*scores, label) + self.regularization_weight*self.reg_loss

    def get_regularization_penalty_vector(self, h, t, r):
        regul = (torch.mean(h ** 2) + torch.mean(t ** 2) + torch.mean(r ** 2)) / 3
        return regul


@Model.register('complex-model')
class ComplexModel(TransEModel):
    def create_embeddings_layer(self, num_entities: int, num_relations: int,
                                embedding_dim: int, single_vec: bool,
                                entities_init_interval_center: float,
                                entities_init_interval_delta: float,
                                relations_init_interval_center: float,
                                relations_init_interval_delta: float) -> None:
        self.h_re = nn.Embedding(
            num_embeddings=num_entities,
            embedding_dim=embedding_dim)
        self.h_im = nn.Embedding(
            num_embeddings=num_entities,
            embedding_dim=embedding_dim)

        if not single_vec:
            self.t_re = nn.Embedding(
                num_embeddings=num_entities,
                embedding_dim=embedding_dim)
            self.t_im = nn.Embedding(
                num_embeddings=num_entities,
                embedding_dim=embedding_dim)
        else:
            self.t_re = self.h_re
            self.t_im = self.t_im

        self.r_re = nn.Embedding(
            num_embeddings=num_relations,
            embedding_dim=embedding_dim)
        self.r_im = nn.Embedding(
            num_embeddings=num_relations,
            embedding_dim=embedding_dim)

        nn.init.xavier_uniform_(self.h_re.weight.data)
        nn.init.xavier_uniform_(self.h_im.weight.data)
        nn.init.xavier_uniform_(self.t_re.weight.data)
        nn.init.xavier_uniform_(self.t_im.weight.data)
        nn.init.xavier_uniform_(self.r_re.weight.data)
        nn.init.xavier_uniform_(self.r_im.weight.data)

    def __init__(
           self,
            num_entities: int,
            num_relations: int,
            embedding_dim: int,
            single_vec: bool = True,
            regularization_weight: float = 0.0,
            number_of_negative_samples: int = 3,
            margin: float = 0.0,
            vocab: Optional[None] = None,
            debug: bool = False
            # we don't need vocab but some api relies on its presence as an argument
    
            # we don't need vocab but some api relies on its presence as an argument
    ) -> None:
        super().__init__(num_entities, num_relations, embedding_dim, 
                         regularization_weight=regularization_weight,
                         number_of_negative_samples=number_of_negative_samples,
                         )
        self.create_embeddings_layer(num_entities,
                                     num_relations,
                                     embedding_dim, single_vec, 0.5, 0.5, 0.5, 0.5)
        self.loss_f: torch.nn.modules._Loss = torch.nn.MarginRankingLoss(  # type: ignore
            margin=margin, reduction='mean')
        self.margin = margin

    def _get_triple_score(self, h_re, h_im, t_re, t_im, r_re, r_im) -> torch.Tensor:
        """ Gets score using three way intersection

        We do not need to worry about the dimentions of the boxes. If
            it can sensibly broadcast it will.
        """
        return torch.sum(
            h_re * t_re * r_re
            + h_im * t_im * r_re
            + h_re * t_im * r_im
            - h_im * t_re * r_im,
            -1
            )

    def get_scores(self,
                   embeddings: Dict) -> Tuple[torch.Tensor, torch.Tensor]:

        p_s = self._get_triple_score(embeddings['p_h_re'],
                                     embeddings['p_h_im'],
                                     embeddings['p_t_re'],
                                     embeddings['p_t_im'],
                                     embeddings['p_r_re'],
                                     embeddings['p_r_im'])


        n_s = self._get_triple_score(embeddings['n_h_re'],
                                     embeddings['n_h_im'],
                                     embeddings['n_t_re'],
                                     embeddings['n_t_im'],
                                     embeddings['n_r_re'],
                                     embeddings['n_r_im'])
        if self.regularization_weight > 0:
            self.reg_loss = self.get_regularization_penalty_vector(
                                                    embeddings['p_h_re'],
                                                    embeddings['p_h_im'],
                                                    embeddings['p_t_re'],
                                                    embeddings['p_t_im'],
                                                    embeddings['p_r_re'],
                                                    embeddings['p_r_im'])

        return (p_s, n_s)

    def get_regularization_penalty_vector(self, h_re, h_im, t_re, t_im, r_re, r_im):
        regul = (torch.mean(h_re ** 2) +
                 torch.mean(h_im ** 2) +
                 torch.mean(t_re ** 2) +
                 torch.mean(t_im ** 2) +
                 torch.mean(r_re ** 2) +
                 torch.mean(r_im ** 2)) / 6
        return regul

    def get_box_embeddings_training(  # type:ignore
            self,
            p_h: torch.Tensor,
            p_r: torch.Tensor,
            p_t: torch.Tensor,  # type:ignore
            n_h: torch.Tensor,
            n_r: torch.Tensor,
            n_t: torch.Tensor,
            **kwargs) -> Dict[str, BoxTensor]:  # type: ignore
        return {
            'p_h_re': self.h_re(p_h),
            'n_h_re': self.h_re(n_h),
            'p_h_im': self.h_im(p_h),
            'n_h_im': self.h_im(n_h),
            'p_t_re': self.t_re(p_t),
            'n_t_re': self.t_re(n_t),
            'p_t_im': self.t_im(p_t),
            'n_t_im': self.t_im(n_t),
            'p_r_re': self.r_re(p_r),
            'n_r_re': self.r_re(n_r),
            'p_r_im': self.r_im(p_r),
            'n_r_im': self.r_im(n_r)
        }

    def get_box_embeddings_val(  # type:ignore
            self,
            h: torch.Tensor,
            r: torch.Tensor,
            t: torch.Tensor,
            label : torch.Tensor
            ) -> Dict[str, BoxTensor]:  # type: ignore
        return {
            'h_re': self.h_re(h),
            'h_im': self.h_im(h),
            't_re': self.t_re(t),
            't_im': self.t_im(t),
            'r_re': self.r_re(r),
            'r_im': self.r_im(r),
            'label': label
        }
    def get_test(self, embeddings: Dict[str, torch.Tensor]) -> Any:
        if self.test_threshold is None:
            raise RuntimeError("test_threshold should be set")
        s = self._get_triple_score(embeddings['h_re'],
                                   embeddings['h_im'],
                                   embeddings['t_re'],
                                   embeddings['t_im'],
                                   embeddings['r_re'],
                                   embeddings['r_im'])
        labels = embeddings['label']
        pos_prediction = (s > self.test_threshold).float()
        neg_prediction = 1.0 - pos_prediction
        predictions = torch.stack((neg_prediction, pos_prediction), -1)
        self.test_f1(predictions, labels)

        return {}

    def get_ranks(self, embeddings: Dict[str, torch.Tensor]) -> Any:
        if self.is_test():
            return self.get_test(embeddings)
        s = self._get_triple_score(embeddings['h_re'],
                                   embeddings['h_im'],
                                   embeddings['t_re'],
                                   embeddings['t_im'],
                                   embeddings['r_re'],
                                   embeddings['r_im'])
        # preds = torch.stack((p_s, n_s), dim=1)  # shape = (batch, 2)
        #self.valid_f1(preds, labels)
        labels = embeddings['label']
        # upate the metrics
        self.threshold_with_f1(s, labels)

        return {}

@Model.register('complexE-model-ranking')
class complexEModelRanking(TransEModelRanking):
    def create_embeddings_layer(self, num_entities: int, num_relations: int,
                                embedding_dim: int, single_vec: bool,
                                entities_init_interval_center: float,
                                entities_init_interval_delta: float,
                                relations_init_interval_center: float,
                                relations_init_interval_delta: float) -> None:
        self.h_re = nn.Embedding(
            num_embeddings=num_entities,
            embedding_dim=embedding_dim)
        self.h_im = nn.Embedding(
            num_embeddings=num_entities,
            embedding_dim=embedding_dim)

        if not single_vec:
            self.t_re = nn.Embedding(
                num_embeddings=num_entities,
                embedding_dim=embedding_dim)
            self.t_im = nn.Embedding(
                num_embeddings=num_entities,
                embedding_dim=embedding_dim)
        else:
            self.t_re = self.h_re
            self.t_im = self.t_im

        self.r_re = nn.Embedding(
            num_embeddings=num_relations,
            embedding_dim=embedding_dim)
        self.r_im = nn.Embedding(
            num_embeddings=num_relations,
            embedding_dim=embedding_dim)

        nn.init.xavier_uniform_(self.h_re.weight.data)
        nn.init.xavier_uniform_(self.h_im.weight.data)
        nn.init.xavier_uniform_(self.t_re.weight.data)
        nn.init.xavier_uniform_(self.t_im.weight.data)
        nn.init.xavier_uniform_(self.r_re.weight.data)
        nn.init.xavier_uniform_(self.r_im.weight.data)


    def __init__(
            self,
            num_entities: int,
            num_relations: int,
            embedding_dim: int,
            single_vec: bool = True,
            regularization_weight: float = 0.0,
            number_of_negative_samples: int = 3,
            margin: float = 0.0,
            vocab: Optional[None] = None,
            debug: bool = False
            # we don't need vocab but some api relies on its presence as an argument
    ) -> None:
        super().__init__(num_entities, num_relations, embedding_dim, 
                         regularization_weight=regularization_weight,
                         number_of_negative_samples=number_of_negative_samples)
        self.create_embeddings_layer(num_entities,
                                     num_relations,
                                     embedding_dim, single_vec, 0.5, 0.5, 0.5, 0.5)
        self.loss_f: torch.nn.modules._Loss = torch.nn.MarginRankingLoss(  # type: ignore
            margin=margin, reduction='mean')
        self.margin = margin

    def _get_triple_score(self, h_re, h_im, t_re, t_im, r_re, r_im) -> torch.Tensor:
        """ Gets score using three way intersection

        We do not need to worry about the dimentions of the boxes. If
            it can sensibly broadcast it will.
        """
        return torch.sum(
            h_re * t_re * r_re
            + h_im * t_im * r_re
            + h_re * t_im * r_im
            - h_im * t_re * r_im,
            -1
            )

    def get_box_embeddings_val(  # type:ignore
            self,
            h: torch.Tensor,
            r: torch.Tensor,
            t: torch.Tensor,
            label : torch.Tensor
            ) -> Dict[str, BoxTensor]:  # type: ignore
        return {
            'h_re': self.h_re(h),
            'h_im': self.h_im(h),
            't_re': self.t_re(t),
            't_im': self.t_im(t),
            'r_re': self.r_re(r),
            'r_im': self.r_im(r),
            'label': label
        }

    def get_scores(self,
                   embeddings: Dict) -> Tuple[torch.Tensor, torch.Tensor]:

        p_s = self._get_triple_score(embeddings['p_h_re'],
                                     embeddings['p_h_im'],
                                     embeddings['p_t_re'],
                                     embeddings['p_t_im'],
                                     embeddings['p_r_re'],
                                     embeddings['p_r_im'])


        n_s = self._get_triple_score(embeddings['n_h_re'],
                                     embeddings['n_h_im'],
                                     embeddings['n_t_re'],
                                     embeddings['n_t_im'],
                                     embeddings['n_r_re'],
                                     embeddings['n_r_im'])
        if self.regularization_weight > 0:
            self.reg_loss = self.get_regularization_penalty_vector(
                                                    embeddings['p_h_re'],
                                                    embeddings['p_h_im'],
                                                    embeddings['p_t_re'],
                                                    embeddings['p_t_im'],
                                                    embeddings['p_r_re'],
                                                    embeddings['p_r_im'])
        else:
            self.reg_loss = 0

        return (p_s, n_s)

    def get_regularization_penalty_vector(self, h_re, h_im, t_re, t_im, r_re, r_im):
        regul = (torch.mean(h_re ** 2) +
                 torch.mean(h_im ** 2) +
                 torch.mean(t_re ** 2) +
                 torch.mean(t_im ** 2) +
                 torch.mean(r_re ** 2) +
                 torch.mean(r_im ** 2)) / 6
        return regul

    def get_box_embeddings_training(  # type:ignore
            self,
            p_h: torch.Tensor,
            p_r: torch.Tensor,
            p_t: torch.Tensor,  # type:ignore
            n_h: torch.Tensor,
            n_r: torch.Tensor,
            n_t: torch.Tensor,
            **kwargs) -> Dict[str, BoxTensor]:  # type: ignore
        return {
            'p_h_re': self.h_re(p_h),
            'n_h_re': self.h_re(n_h),
            'p_h_im': self.h_im(p_h),
            'n_h_im': self.h_im(n_h),
            'p_t_re': self.t_re(p_t),
            'n_t_re': self.t_re(n_t),
            'p_t_im': self.t_im(p_t),
            'n_t_im': self.t_im(n_t),
            'p_r_re': self.r_re(p_r),
            'n_r_re': self.r_re(n_r),
            'p_r_im': self.r_im(p_r),
            'n_r_im': self.r_im(n_r)
        }

    def _get_hr_score(self, embeddings: Dict[str, BoxTensor]) -> torch.Tensor:
        with torch.no_grad():
            b = embeddings
            hr_scores = self._get_triple_score(b['hr_e_re'], b['hr_e_im'],  b['hr_t_re'], b['hr_t_im'], b['hr_r_re'], b['hr_r_im'])

            return hr_scores.reshape(-1)  # flatten

    def _get_tr_score(self, embeddings: Dict[str, BoxTensor]) -> torch.Tensor:
        b = embeddings
        tr_scores = self._get_triple_score(b['tr_h_re'], b['tr_h_im'], b['tr_e_re'], b['tr_e_im'], b['tr_r_re'], b['tr_r_im'])

        return tr_scores.reshape(-1)  # flatten

    def get_box_embeddings_val(self, hr_t: torch.Tensor, hr_r: torch.Tensor,
                               hr_e: torch.Tensor, tr_h: torch.Tensor,
                               tr_r: torch.Tensor,
                               tr_e: torch.Tensor) -> Dict[str, BoxTensor]:

        if not self.is_eval():
            raise RuntimeError("get_box_embeddings_val called during training")
        with torch.no_grad():
            embs = {
                'hr_t_re': self.t_re(hr_t),  # shape=(batch_size, 2, emb_dim)
                'hr_r_re': self.r_re(hr_r),
                'hr_e_re': self.h_re(hr_e),  # shape=(batch_size, *,2,emb_dim)
                'tr_h_re': self.h_re(tr_h),
                'tr_r_re': self.r_re(tr_r),
                'tr_e_re': self.t_re(tr_e),  # shape=(*,2,emb_dim)
                'hr_t_im': self.t_im(hr_t),  # shape=(batch_size, 2, emb_dim)
                'hr_r_im': self.r_im(hr_r),
                'hr_e_im': self.h_im(hr_e),  # shape=(batch_size, *,2,emb_dim)
                'tr_h_im': self.h_im(tr_h),
                'tr_r_im': self.r_im(tr_r),
                'tr_e_im': self.t_im(tr_e)  # shape=(*,2,emb_dim)
            }  # batch_size is assumed to be 1 during rank validation

        return embs

@Model.register('rotatE-model')
class RotatEModel(TransEModel):
    def __init__(
            self,
            num_entities: int,
            num_relations: int,
            embedding_dim: int,
            single_vec: bool = True,
            regularization_weight: float = 0.0,
            number_of_negative_samples: int = 3,
            margin: float = 0.5,
            epsilon: float = 2.0,
            vocab: Optional[None] = None,
            debug: bool = False
            # we don't need vocab but some api relies on its presence as an argument
    ) -> None:
        super(RotatEModel, self).__init__(num_entities, num_relations, embedding_dim, 
                         regularization_weight=regularization_weight,
                         number_of_negative_samples=number_of_negative_samples)

        self.loss_f: torch.nn.modules._Loss = torch.nn.LogSigmoid()
        self.create_embeddings_layer_(num_entities,
                                     num_relations,
                                     embedding_dim, single_vec, margin, epsilon)

    def create_embeddings_layer_(self, num_entities: int, num_relations: int,
                                embedding_dim: int, single_vec: bool, margin: float, epsilon: float) -> None:
        self.h = nn.Embedding(
            num_embeddings=num_entities,
            embedding_dim=embedding_dim*2)

        if not single_vec:
            self.t = nn.Embedding(
                num_embeddings=num_entities,
                embedding_dim=embedding_dim*2)
        else:
            self.t = self.h

        self.r = nn.Embedding(
            num_embeddings=num_relations,
            embedding_dim=embedding_dim)

        nn.init.xavier_uniform_(self.h.weight.data)
        nn.init.xavier_uniform_(self.t.weight.data)
        nn.init.xavier_uniform_(self.r.weight.data)

        self.ent_embedding_range = nn.Parameter(
            torch.Tensor([(margin + epsilon) / (self.embedding_dim*2)]), 
            requires_grad=False
        )

        nn.init.uniform_(
            tensor = self.h.weight.data, 
            a=-self.ent_embedding_range.item(), 
            b=self.ent_embedding_range.item()
        )
        nn.init.uniform_(
            tensor = self.t.weight.data, 
            a=-self.ent_embedding_range.item(), 
            b=self.ent_embedding_range.item()
        )

        self.rel_embedding_range = nn.Parameter(
            torch.Tensor([(margin + epsilon) / self.embedding_dim]), 
            requires_grad=False
        )

        nn.init.uniform_(
            tensor = self.r.weight.data, 
            a=-self.rel_embedding_range.item(), 
            b=self.rel_embedding_range.item()
        )

        self.margin = margin

        self.appropriate_emb = {
            'p_h': self.h,
            'n_h': self.h,
            'h': self.h,
            'tr_h': self.h,
            'hr_e': self.h,
            'p_t': self.t,
            'n_t': self.t,
            't': self.t,
            'hr_t': self.t,
            'tr_e': self.t,
            'p_r': self.r,
            'n_r': self.r,
            'r': self.r,
            'hr_r': self.r,
            'tr_r': self.r,
            'label': (lambda x: x)
        }

    def _get_triple_score(self, h: torch.Tensor, t: torch.Tensor,
                          r: torch.Tensor) -> torch.Tensor:
        pi = 3.14159265358979323846
        re_head, im_head = torch.chunk(h, 2, dim=-1)
        re_tail, im_tail = torch.chunk(t, 2, dim=-1)

        phase_relation = r / (self.rel_embedding_range.item() / pi)

        re_relation = torch.cos(phase_relation)
        im_relation = torch.sin(phase_relation)

        re_score = re_head * re_relation - im_head * im_relation
        im_score = re_head * im_relation + im_head * re_relation
        re_score = re_score - re_tail
        im_score = im_score - im_tail

        score = torch.stack([re_score, im_score], dim = 0)
        score = score.norm(dim = 0).sum(dim = -1)

        return self.margin - score


    def get_loss(self, scores: Tuple[torch.Tensor, torch.Tensor],
                 label: torch.Tensor) -> torch.Tensor:
        # max margin loss expects label to be float
        p_score = scores[0]
        n_score = scores[1]
        loss = -(self.loss_f(p_score).mean() + self.loss_f(-n_score).mean()) / 2
        return loss + self.regularization_weight*self.reg_loss


@Model.register('rotatE-model-ranking')
class RotatEModelRanking(TransEModelRanking):
    def __init__(
            self,
            num_entities: int,
            num_relations: int,
            embedding_dim: int,
            single_vec: bool = True,
            regularization_weight: float = 0.0,
            number_of_negative_samples: int = 3,
            margin: float = 0.5,
            epsilon: float = 2.0,
            vocab: Optional[None] = None,
            debug: bool = False
            # we don't need vocab but some api relies on its presence as an argument
    ) -> None:

        
        super(RotatEModelRanking, self).__init__(num_entities, num_relations, embedding_dim, 
                         regularization_weight=regularization_weight,
                         number_of_negative_samples=number_of_negative_samples)

        self.loss_f: torch.nn.modules._Loss = torch.nn.LogSigmoid()
        self.create_embeddings_layer_(num_entities,
                                     num_relations,
                                     embedding_dim, single_vec, margin, epsilon)
    def create_embeddings_layer_(self, num_entities: int, num_relations: int,
                                     embedding_dim: int, single_vec: bool, 
                                    margin: float, epsilon: float) -> None:
            self.h = nn.Embedding(
                num_embeddings=num_entities,
                embedding_dim=embedding_dim*2)

            if not single_vec:
                self.t = nn.Embedding(
                    num_embeddings=num_entities,
                    embedding_dim=embedding_dim*2)
            else:
                self.t = self.h

            self.r = nn.Embedding(
                num_embeddings=num_relations,
                embedding_dim=embedding_dim)

            nn.init.xavier_uniform_(self.h.weight.data)
            nn.init.xavier_uniform_(self.t.weight.data)
            nn.init.xavier_uniform_(self.r.weight.data)

            self.ent_embedding_range = nn.Parameter(
                torch.Tensor([(margin + epsilon) / (self.embedding_dim*2)]), 
                requires_grad=False
            )

            nn.init.uniform_(
                tensor = self.h.weight.data, 
                a=-self.ent_embedding_range.item(), 
                b=self.ent_embedding_range.item()
            )
            nn.init.uniform_(
                tensor = self.t.weight.data, 
                a=-self.ent_embedding_range.item(), 
                b=self.ent_embedding_range.item()
            )

            self.rel_embedding_range = nn.Parameter(
                torch.Tensor([(margin + epsilon) / self.embedding_dim]), 
                requires_grad=False
            )

            nn.init.uniform_(
                tensor = self.r.weight.data, 
                a=-self.rel_embedding_range.item(), 
                b=self.rel_embedding_range.item()
            )

            self.margin = margin

            self.appropriate_emb = {
                'p_h': self.h,
                'n_h': self.h,
                'h': self.h,
                'tr_h': self.h,
                'hr_e': self.h,
                'p_t': self.t,
                'n_t': self.t,
                't': self.t,
                'hr_t': self.t,
                'tr_e': self.t,
                'p_r': self.r,
                'n_r': self.r,
                'r': self.r,
                'hr_r': self.r,
                'tr_r': self.r,
                'label': (lambda x: x)
            }

    def get_box_embeddings_val(self, h: torch.Tensor, t: torch.Tensor,
                            r: torch.Tensor, label: torch.Tensor) -> Dict[str, BoxTensor]:

        if not self.is_eval():
            raise RuntimeError("get_box_embeddings_val called during training")
        with torch.no_grad():
            embs = {
                'h': self.h(h),  # shape=(batch_size, 2, emb_dim)
                'r': self.r(r),
                't': self.t(torch.arange(self.num_entities)),  # shape=(batch_size, *,2,emb_dim
                't_act': t
            }  # batch_size is assumed to be 1 during rank validation

        return embs

    def get_ranks(self, embeddings: Dict[str, BoxTensor]) -> Any:
        if not self.is_eval():
            raise RuntimeError("get_ranks called during training")
        with torch.no_grad():
            # hr_scores = self._get_hr_score(embeddings)
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

    def _get_triple_score(self, h: torch.Tensor, t: torch.Tensor,
                          r: torch.Tensor) -> torch.Tensor:
        pi = 3.14159265358979323846
        re_head, im_head = torch.chunk(h, 2, dim=-1)
        re_tail, im_tail = torch.chunk(t, 2, dim=-1)

        phase_relation = r / (self.rel_embedding_range.item() / pi)

        re_relation = torch.cos(phase_relation)
        im_relation = torch.sin(phase_relation)

        re_score = re_head * re_relation - im_head * im_relation
        im_score = re_head * im_relation + im_head * re_relation
        re_score = re_score - re_tail
        im_score = im_score - im_tail

        score = torch.stack([re_score, im_score], dim = 0)
        score = score.norm(dim = 0).sum(dim = -1)

        return self.margin - score

    def get_loss(self, scores: Tuple[torch.Tensor, torch.Tensor],
                 label: torch.Tensor) -> torch.Tensor:
        # max margin loss expects label to be float
        p_score = scores[0]
        n_score = scores[1]

        loss = -(self.loss_f(p_score).mean() + self.loss_f(-n_score).mean()) / 2
        return loss + self.regularization_weight*self.reg_loss

