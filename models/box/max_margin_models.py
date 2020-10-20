from typing import Tuple, Dict, Any, Union
from .base import BaseBoxModel
from typing import Optional
from allennlp.models import Model
from allennlp.training.metrics import Average
from boxes.box_wrapper import SigmoidBoxTensor, BoxTensor
from boxes.modules import BoxEmbedding
from allennlp.modules.token_embedders.embedding import Embedding
from allennlp.modules.token_embedders import Embedding
import torch
import numpy as np
from ..metrics import HitsAt10, F1WithThreshold
from allennlp.training.metrics import F1Measure, FBetaMeasure


@Model.register('max-margin-box-model')
class MaxMarginBoxModel(BaseBoxModel):
    def __init__(
            self,
            num_entities: int,
            num_relations: int,
            embedding_dim: int,
            box_type: str = 'SigmoidBoxTensor',
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
        # we don't need vocab but some api relies on its presence as an argument
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
        self.loss_f: torch.nn.modules._Loss = torch.nn.MarginRankingLoss(  # type: ignore
            margin=margin, reduction='mean')
        self.margin = margin
        # used only during eval
        # metrics defined in base model
        # self.adversarial_negative = adversarial_negative
        # self.adv_neg_softmax_temp = adv_neg_softmax_temp

    def batch_with_negative_samples(self, **kwargs) -> Dict[str, torch.Tensor]:
        if self.number_of_negative_samples <= 0:
            return kwargs
        label = kwargs.get('label')

        if label is None:
            raise ValueError
        head_name, head = self.get_expected_head(kwargs)

        tail_name, tail = self.get_expected_tail(kwargs)
        rel_name, rel = self.get_expected_relation(kwargs)

        # create the tensors for negatives
        # size = self.get_negaive_sample_tensorsize(head.size())
        size = head.size()[-1]
        multiplier = int(self.number_of_negative_samples / 2)
        head = self.repeat(head, multiplier * 2)  # l = (multiplier*2)*size
        tail = self.repeat(tail, multiplier * 2)
        rel = self.repeat(rel, multiplier * 2)
        label = self.repeat(label, multiplier * 2)
        # create copy for negatives
        neg_head = head.clone()
        neg_tail = tail.clone()

        # fill with random neg
        self.fill_random_entities_(neg_head[0:size * multiplier])
        self.fill_random_entities_(neg_tail[size * multiplier:])

        # create the dict
        batch = {
            'p_h': head,
            'p_r': rel,
            'p_t': tail,
            'n_h': neg_head,
            'n_t': neg_tail,
            'n_r': rel,
            'label': label
        }

        return batch

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
            'p_h': self.h(p_h),
            'p_t': self.t(p_t),
            'p_r': self.r(p_r),
            'n_h': self.h(n_h),
            'n_t': self.t(n_t),
            'n_r': self.r(n_r)
        }

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
                'tr_e': self.t(tr_e)  # shape=(*,2,emb_dim)
            }  # batch_size is assumed to be 1 during rank validation

        return embs

    def _get_triple_score(self, head: BoxTensor, tail: BoxTensor,
                          relation: BoxTensor) -> torch.Tensor:
        """ Gets score using three way intersection

        We do not need to worry about the dimentions of the boxes. If
            it can sensibly broadcast it will.
        """
        head_relation_box = relation.intersection(head)
        tail_relation_box = relation.intersection(tail)
        score = head_relation_box.intersection_log_soft_volume(
            tail_relation_box, temp=self.softbox_temp)

        return score

    def _get_hr_score(self, embeddings: Dict[str, BoxTensor]) -> torch.Tensor:
        with torch.no_grad():
            b = embeddings
            hr_scores = self._get_triple_score(b['hr_e'], b['hr_t'], b['hr_r'])

            return hr_scores.reshape(-1)  # flatten

    def _get_tr_score(self, embeddings: Dict[str, BoxTensor]) -> torch.Tensor:
        b = embeddings
        tr_scores = self._get_triple_score(b['tr_h'], b['tr_e'], b['tr_r'])

        return tr_scores.reshape(-1)  # flatten

    def get_ranks(self, embeddings: Dict[str, BoxTensor]) -> Any:
        if not self.is_eval():
            raise RuntimeError("get_ranks called during training")
        with torch.no_grad():
            hr_scores = self._get_hr_score(embeddings)
            tr_scores = self._get_tr_score(embeddings)

            self.int_volume_dev(hr_scores[-1].item())

            # find the spot of zeroth element in the sorted array
            hr_rank = (
                torch.argsort(
                    hr_scores,
                    descending=True) == hr_scores.shape[0] - 1  # type:ignore
            ).nonzero().reshape(-1).item()  # type:ignore
            tr_rank = (
                torch.argsort(  # type:ignore
                    tr_scores, descending=True) == tr_scores.shape[0] -
                1).nonzero().reshape(-1).item()
            self.head_replacement_rank_avg(hr_rank)
            self.tail_replacement_rank_avg(tr_rank)
            avg_rank = (hr_rank + tr_rank) / 2.
            self.avg_rank(avg_rank)
            self.hitsat10(hr_rank)
            self.hitsat10(tr_rank)
            self.head_hitsat3(hr_rank)
            self.tail_hitsat3(tr_rank)
            self.head_hitsat1(hr_rank)
            self.tail_hitsat1(tr_rank)
            hr_mrr = (1. / (hr_rank + 1))
            tr_mrr = (1. / (tr_rank + 1))
            mrr = (hr_mrr + tr_mrr) / 2.
            self.head_replacement_mrr(hr_mrr)
            self.tail_replacement_mrr(tr_mrr)
            self.mrr(mrr)

            return {
                'hr_rank': hr_rank,
                'tr_rank': tr_rank,
                'avg_rank': avg_rank,
                'hr_mrr': hr_mrr,
                'tr_mrr': tr_mrr,
                'int_vol': (hr_scores[-1]).item(),
                'mrr': mrr
            }

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:

        return {
            'hr_rank': self.head_replacement_rank_avg.get_metric(reset),
            'tr_rank': self.tail_replacement_rank_avg.get_metric(reset),
            'avg_rank': self.avg_rank.get_metric(reset),
            'hitsat10': self.hitsat10.get_metric(reset),
            'hr_mrr': self.head_replacement_mrr.get_metric(reset),
            'tr_mrr': self.tail_replacement_mrr.get_metric(reset),
            'int_volume_train': self.int_volume_train.get_metric(reset),
            'int_volume_dev': self.int_volume_dev.get_metric(reset),
            'regularization_loss': self.regularization_loss.get_metric(reset),
            'hr_hitsat1': self.head_hitsat1.get_metric(reset),
            'tr_hitsat1': self.tail_hitsat1.get_metric(reset),
            'hr_hitsat3': self.head_hitsat3.get_metric(reset),
            'tr_hitsat3': self.tail_hitsat3.get_metric(reset),
            'mrr': self.mrr.get_metric(reset)
        }

    def get_scores(self,
                   embeddings: Dict) -> Tuple[torch.Tensor, torch.Tensor]:

        p_s = self._get_triple_score(embeddings['p_h'], embeddings['p_t'],
                                     embeddings['p_r'])

        self.int_volume_train(torch.mean(p_s).item())

        n_s = self._get_triple_score(embeddings['n_h'], embeddings['n_t'],
                                     embeddings['n_r'])

        # if self.adversarial_negative:
        # with torch.no_grad():
        # weights = torch.nn.functional.log_softmax(
        # n_s * self.adv_neg_softmax_temp, dim=-1).detach()
        # n_s = weights + n_s

        return (p_s, n_s)

    def get_loss(self, scores: Tuple[torch.Tensor, torch.Tensor],
                 label: torch.Tensor) -> torch.Tensor:
        # max margin loss expects label to be float
        label = label.to(scores[0].dtype)
        loss = self.loss_f(*scores, label) + self.get_regularization_penalty()

        return loss


@Model.register('max-margin-conditional-box-model')
class MaxMarginConditionalModel(MaxMarginBoxModel):
    def _get_triple_score(self, head: BoxTensor, tail: BoxTensor,
                          relation: BoxTensor) -> torch.Tensor:
        """ Gets score using conditionals.

        :: note: We do not need to worry about the dimentions of the boxes. If
                it can sensibly broadcast it will.
            """
        #head_relation_box = relation.intersection(head)
        #tail_relation_box = relation.intersection(tail)
        # tail_head_relation_box_vol = tail.intersection_log_soft_volume(
        #    head_relation_box, temp=self.softbox_temp)
        # score = tail_head_relation_box_vol - head_relation_box.log_soft_volume(
        #    temp=self.softbox_temp)
        head_tail_box_vol = head.intersection_log_soft_volume(
            tail, temp=self.softbox_temp)
        # score = tail_head_relation_box_vol - tail_relation_box.log_soft_volume(
        #    temp=self.softbox_temp)
        score = head_tail_box_vol - tail.log_soft_volume(
            temp=self.softbox_temp)

        return score


@Model.register('max-margin-conditional-classification-box-model')
class MaxMarginConditionalClassificationModel(MaxMarginConditionalModel):
    def __init__(
            self,
            num_entities: int,
            num_relations: int,
            embedding_dim: int,
            box_type: str = 'SigmoidBoxTensor',
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
        self.istest = False
        self.test_threshold = None
        self.test_f1 = F1Measure(positive_label=1)

    def is_test(self) -> bool:
        if (not self.is_eval()) and self.test:
            raise RuntimeError("test flag is true but eval is false")

        return self.is_eval() and self.istest

    def test(self) -> None:
        if not self.is_eval():
            raise RuntimeError("test flag is true but eval is false")
        self.istest = True

    def get_ranks(self, embeddings: Dict[str, BoxTensor]) -> Any:
        if self.is_test():
            return self.get_test(embeddings)

        s = self._get_triple_score(embeddings['h'], embeddings['t'],
                                   embeddings['r'])
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
                                   embeddings['r'])
        labels = embeddings['label']
        pos_prediction = (s > self.test_threshold).float()
        neg_prediction = 1.0 - pos_prediction
        predictions = torch.stack((neg_prediction, pos_prediction), -1)
        self.test_f1(predictions, labels)

        return {}

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        if self.is_eval():
            if not self.test:
                metrics = self.threshold_with_f1.get_metric(reset)
            else:
                p, r, f = self.test_f1.get_metric(reset)
                metrics = {'precision': p, 'recall': r, 'fscore': f}
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
        loss = self.loss_f(*scores, label) + self.get_regularization_penalty()
        # metrics require 0,1 labels

        if not self.is_eval():
            with torch.no_grad():
                labels = torch.zeros_like(scores[0]).reshape(
                    -1)  # shape = (batch)
                preds = torch.stack(scores, dim=1)
                self.train_f1(preds, labels)

        return loss


@Model.register('dim-wise-max-margin-conditional-box-model')
class DimWiseMaxMarginConditionalModel(MaxMarginBoxModel):
    def __init__(
            self,
            num_entities: int,
            num_relations: int,
            embedding_dim: int,
            box_type: str = 'SigmoidBoxTensor',
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

    def create_embeddings_layer(self, num_entities: int, num_relations: int,
                                embedding_dim: int, single_box: bool,
                                entities_init_interval_center: float,
                                entities_init_interval_delta: float,
                                relations_init_interval_center: float,
                                relations_init_interval_delta: float) -> None:
        self.h = BoxEmbedding(
            num_embeddings=num_entities,
            box_embedding_dim=embedding_dim,
            box_type=self.box_type,
            sparse=False,
            init_interval_center=entities_init_interval_center,
            init_interval_delta=entities_init_interval_delta)

        if not single_box:
            self.t = BoxEmbedding(
                num_embeddings=num_entities,
                box_embedding_dim=embedding_dim,
                box_type=self.box_type,
                sparse=False,
                init_interval_center=entities_init_interval_center,
                init_interval_delta=entities_init_interval_delta)
        else:
            self.t = self.h

        self.r = Embedding(num_relations, embedding_dim)
        # Also create common name mapping
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

    def _get_triple_score(self, head: BoxTensor, tail: BoxTensor,
                          relation: torch.Tensor) -> torch.Tensor:
        """ Gets score using conditionals.

        :: note: We do not need to worry about the dimentions of the boxes. If
                it can sensibly broadcast it will.
            """
        # get conditionals interval intersection scores

        numerators = head.dimension_wise_intersection_soft_volume(
            tail, temp=self.softbox_temp).clamp_min(1e-38)
        denominators = head.dimension_wise_soft_volume(
            temp=self.softbox_temp).clamp_min(1e-38)

        probs = numerators / denominators  # shape = (batch, num_dims)

        weighted_probs = (torch.nn.functional.softmax(relation,
                                                      dim=-1)) * probs
        score = torch.sum(weighted_probs, dim=-1)

        return score

    def get_regularization_penalty(self) -> Union[float, torch.Tensor]:

        if self.regularization_weight > 0:
            entity_penalty = self.h.get_bounding_box().log_soft_volume(
                temp=self.softbox_temp)
            # don't penalize if bb has volume 1 or less

            if (entity_penalty < 0).all():
                entity_penalty = entity_penalty * 0

            if not self.single_box:
                entity_penalty_t = self.t.get_bounding_box().log_soft_volume(
                    temp=self.softbox_temp)

                if (entity_penalty_t < 0).all():
                    pass
                else:
                    entity_penalty += entity_penalty_t

            reg_loss = (self.regularization_weight * (entity_penalty))
            # track the reg loss
            self.regularization_loss(reg_loss.item())

            return reg_loss
        else:
            return 0.0

    def get_histograms_to_log(self) -> Dict[str, torch.Tensor]:

        return {
            "relation_weights": self.r.weight.cpu().data.numpy().flatten(),
            "head_entity_volume_historgram": self.get_h_vol(),
            "tail_entity_volume_historgram": self.get_t_vol()
        }


@Model.register('dim-wise-max-margin-conditional-box-model-2')
class DimWiseMaxMarginConditionalModel2(DimWiseMaxMarginConditionalModel):
    def _get_triple_score(self, head: BoxTensor, tail: BoxTensor,
                          relation: torch.Tensor) -> torch.Tensor:
        """ Gets score using conditionals.

        :: note: We do not need to worry about the dimentions of the boxes. If
                it can sensibly broadcast it will.
            """
        # get conditionals interval intersection scores

        numerators = head.dimension_wise_intersection_soft_volume(
            tail, temp=self.softbox_temp).clamp_min(1e-38)
        denominators = head.dimension_wise_soft_volume(
            temp=self.softbox_temp).clamp_min(1e-38)

        probs = numerators / denominators  # shape = (batch, num_dims)

        weighted_probs = relation * probs
        score = torch.sum(weighted_probs, dim=-1)

        return score


@Model.register('max-margin-conditional-inside-relation-box-model')
class MaxMarginConditionalModel2(MaxMarginConditionalModel):
    def _get_triple_score(self, head: BoxTensor, tail: BoxTensor,
                          relation: BoxTensor) -> torch.Tensor:
        head_relation_box = relation.intersection(head)
        tail_relation_box = relation.intersection(tail)
        head_tail_relation_intersection_vol = tail_relation_box.intersection_log_soft_volume(
            head_relation_box, temp=self.softbox_temp)
        relation_box_vol = relation.log_soft_volume(temp=self.softbox_temp)
        score = head_tail_relation_intersection_vol - relation_box_vol

        return score


@Model.register('dim-wise-max-margin-box-model')
class DimWiseMaxMarginModel(DimWiseMaxMarginConditionalModel):
    def _get_triple_score(self, head: BoxTensor, tail: BoxTensor,
                          relation: torch.Tensor) -> torch.Tensor:
        """ Gets score using conditionals.

        :: note: We do not need to worry about the dimentions of the boxes. If
                it can sensibly broadcast it will.
            """
        # get conditionals interval intersection scores

        numerators = head.dimension_wise_intersection_soft_volume(
            tail, temp=self.softbox_temp).clamp_min(1e-38)

        probs = numerators  # shape = (batch, num_dims)
        weighted_probs = (torch.nn.functional.softmax(relation,
                                                      dim=-1)) * probs

        score = torch.sum(weighted_probs, dim=-1)

        return score
