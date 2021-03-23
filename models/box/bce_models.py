from typing import Tuple, Dict, Any, Union
from .base import BaseBoxModel
from .max_margin_models import MaxMarginBoxModel, MaxMarginConditionalModel, MaxMarginConditionalClassificationModel
from typing import Optional
from allennlp.models import Model
from allennlp.training.metrics import Average
from boxes.box_wrapper import SigmoidBoxTensor, BoxTensor
from boxes.modules import BoxEmbedding
from boxes.utils import log1mexp
from allennlp.modules.token_embedders import Embedding
import torch
from torch import nn
from torch.distributions.multivariate_normal import MultivariateNormal
import numpy as np
from ..metrics import HitsAt10, F1WithThreshold
from allennlp.training.metrics import F1Measure, FBetaMeasure


@Model.register('BCE-box-model')
class BCEBoxModel(MaxMarginConditionalModel):
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
                 neg_samples_in_dataset_reader: int = 0) -> None:
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

        if number_of_negative_samples > 0 and neg_samples_in_dataset_reader > 0:
            raise ValueError(
                "Negative sampling cannot be done in both model and dataset reader"
            )
        neg_samples = number_of_negative_samples if number_of_negative_samples > 0 else neg_samples_in_dataset_reader

        if (neg_samples > 0):
            neg_weight = 1.0 / neg_samples
        else:
            neg_weight = 1.0
        self.loss_f = torch.nn.NLLLoss(
            weight=torch.tensor([neg_weight, 1.0]), reduction='mean')

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

    def get_scores(self, embeddings: Dict) -> torch.Tensor:
        p = self._get_triple_score(embeddings['h'], embeddings['t'],
                                   embeddings['r'])

        return p

    def get_loss(self, scores: torch.Tensor,
                 label: torch.Tensor) -> torch.Tensor:
        log_p = scores
        log1mp = log1mexp(log_p)
        logits = torch.stack([log1mp, log_p], dim=-1)
        loss = self.loss_f(logits, label) + self.get_regularization_penalty()

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
        # size = self.get_negaive_sample_tensorsize(head.size())
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


@Model.register('BCE-classification-box-model')
class BCEBoxClassificationModel(BCEBoxModel):
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
                 neg_samples_in_dataset_reader: int = 0) -> None:
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
            neg_samples_in_dataset_reader=neg_samples_in_dataset_reader)
        self.train_f1 = FBetaMeasure(average='micro')
        # self.valid_f1 = FBetaMeasure(average='micro')
        self.threshold_with_f1 = F1WithThreshold(flip_sign=True)

        self.istest = False
        self.test_threshold = None
        # self.test_f1 = FBetaMeasure(average='macro')
        self.test_f1 = F1Measure(positive_label=1)

    def is_test(self) -> bool:
        if (not self.is_eval()) and self.test:
            raise RuntimeError("test flag is true but eval is false")

        return self.is_eval() and self.istest

    def test(self) -> None:
        if not self.is_eval():
            raise RuntimeError("test flag is true but eval is false")
        self.istest = True

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
        
        loss = self.loss_f(logits, label) + self.get_regularization_penalty()
        if torch.isnan(loss).any():
            breakpoint()
        if not self.is_eval():
            with torch.no_grad():
                self.train_f1(logits, label)

        return loss

    def get_ranks(self, embeddings: Dict[str, BoxTensor]) -> Any:
        if self.is_test():
            return self.get_test(embeddings)

        s = self._get_triple_score(embeddings['h'], embeddings['t'],
                                   embeddings['r'])
        # preds = torch.stack((p_s, n_s), dim=1)  # shape = (batch, 2)
        # self.valid_f1(preds, labels)
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
            if not self.istest:
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

    def get_regularization_penalty(self) -> Union[float, torch.Tensor]:

        if self.is_eval():
            return 0.0

        if self.regularization_weight > 0:
            all_ = self.h.all_boxes
            deltas = all_.Z - all_.z
            with torch.no_grad():
                assert (deltas >= 0.0).all()
            penalty = self.regularization_weight * torch.sum(deltas)
            # track the reg loss
            self.regularization_loss(penalty.item())

            return penalty
        else:
            return 0.0


@Model.register('BCE-classification-min-delta-penalty-box-model')
class BCEBoxClassificationMinDeltaPenaltyModel(BCEBoxClassificationModel):
    def get_regularization_penalty(self) -> Union[float, torch.Tensor]:

        if self.is_eval():
            return 0.0

        if self.regularization_weight > 0:
            #vols = self.h.get_volumes(temp=self.softbox_temp)
            # don't penalize if box has very less vol
            all_ = self.h.all_boxes
            deltas = all_.Z - all_.z
            with torch.no_grad():
                assert (deltas >= 0.0).all()

            mask = (deltas > (0.001))
            deltas_to_penalize = deltas[mask]
            penalty = self.regularization_weight * torch.sum(
                deltas_to_penalize)

            # track the reg loss
            self.regularization_loss(penalty.item())

            return penalty
        else:
            return 0.0


@Model.register('BCE-classification-min-delta-penalty-box-model-2')
class BCEBoxClassificationMinDeltaPenaltyModel2(BCEBoxClassificationModel):
    def get_regularization_penalty(self) -> Union[float, torch.Tensor]:

        if self.is_eval():
            return 0.0

        if self.regularization_weight > 0:
            vols = torch.exp(self.h.get_volumes(temp=self.softbox_temp))
            # don't penalize if box has very less vol
            mask = (vols > (0.01)**self.embedding_dim)
            vols = vols[mask]
            penalty = self.regularization_weight * torch.sum(vols)

            # track the reg loss
            self.regularization_loss(penalty.item())

            return penalty
        else:
            return 0.0


@Model.register('BCE-classification-split-neg-box-model')
class BCEBoxClassificationSplitNegModel(BCEBoxClassificationModel):
    def __init__(self,
                 num_entities: int,
                 num_relations: int,
                 embedding_dim: int,
                 box_type: str = 'SigmoidBoxTensor',
                 single_box: bool = False,
                 softbox_temp: float = 10.,
                 number_of_negative_samples_head: int = 0,
                 number_of_negative_samples_tail: int = 0,
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
            number_of_negative_samples=number_of_negative_samples_head +
            number_of_negative_samples_tail,
            debug=debug,
            regularization_weight=regularization_weight,
            init_interval_center=init_interval_center,
            init_interval_delta=init_interval_delta)
        self.number_of_negative_samples_head = number_of_negative_samples_head
        self.number_of_negative_samples_tail = number_of_negative_samples_tail

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
        # size = self.get_negaive_sample_tensorsize(head.size())
        # for Classification model, we will do it inplace
        multiplier = int(self.number_of_negative_samples)
        size = head.size()[-1]
        head = self.repeat(head, multiplier + 1)
        tail = self.repeat(tail, multiplier + 1)
        rel = self.repeat(rel, multiplier + 1)
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

        return {'h': head, 't': tail, 'r': rel, 'label': label}


@Model.register('BCE-delta-penalty-box-model')
class BCEDeltaPenaltyModel(BCEBoxModel):
    def get_regularization_penalty(self) -> Union[float, torch.Tensor]:

        if self.is_eval():
            return 0.0

        if self.regularization_weight > 0:
            all_ = self.h.all_boxes
            deltas = all_.Z - all_.z
            with torch.no_grad():
                assert (deltas >= 0.0).all()
            penalty = self.regularization_weight * torch.sum(deltas)
            # track the reg loss
            self.regularization_loss(penalty.item())

            return penalty
        else:
            return 0.0


@Model.register('BCE-delta-l1-penalty-box-model')
class BCEDeltaL1PenaltyModel(BCEDeltaPenaltyModel):
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
                 l1_regularization_weight: float = 0) -> None:
        super().__init__(
            num_entities=num_entities,
            num_relations=num_relations,
            embedding_dim=embedding_dim,
            box_type=box_type,
            single_box=single_box,
            softbox_temp=softbox_temp,
            number_of_negative_samples=number_of_negative_samples,
            debug=debug,
            regularization_weight=regularization_weight,
            init_interval_center=init_interval_center,
            init_interval_delta=init_interval_delta,
        )
        self.l1_regularization_weight = l1_regularization_weight
        self.l1 = torch.nn.L1Loss(reduce=False, reduction=None)

    def get_scores(self, embeddings: Dict) -> torch.Tensor:
        p = self._get_triple_score(embeddings['h'], embeddings['t'],
                                   embeddings['r'])

        if self.is_eval():

            return p
        else:
            h_centers = (embeddings['h'].Z + embeddings['h'].z) / 2.0
            t_centers = (embeddings['t'].Z + embeddings['t'].z) / 2.0
            l1 = torch.sum(
                self.l1(h_centers, t_centers),  # shape = (batch, emb_dim)
                dim=-1)  # final shape = (batch,)

            return (p, l1)

    def get_loss(self, scores: Tuple[torch.Tensor, torch.Tensor],
                 label: torch.Tensor) -> torch.Tensor:
        log_p = scores[0]
        log1mp = log1mexp(log_p)
        logits = torch.stack([log1mp, log_p], dim=-1)
        loss = self.loss_f(logits, label) + \
            self.get_regularization_penalty()
        l1 = scores[1]
        # reduce l1 for pos, increase for the neg
        pos = label.type(torch.bool)
        neg = ~pos
        pos_l1s = l1[pos]
        neg_l1s = -1 * l1[neg]
        # protect against empty tensors as mean produces nan

        if l1[pos].nelement():
            l1_loss_pos = torch.mean(pos_l1s)
        else:
            l1_loss_pos = 0.0

        if l1[neg].nelement():
            l1_loss_neg = torch.mean(neg_l1s)
        else:
            l1_loss_neg = 0.0
        l1_loss = l1_loss_pos + l1_loss_neg

        return loss + self.l1_regularization_weight * l1_loss


@Model.register('BCE-delta-min-penalty-box-model')
class BCEDeltaMinPenaltyModel(BCEBoxModel):
    def get_regularization_penalty(self) -> Union[float, torch.Tensor]:

        if self.is_eval():
            return 0.0

        if self.regularization_weight > 0:
            all_ = self.h.all_boxes
            deltas = all_.Z - all_.z
            with torch.no_grad():
                assert (deltas >= 0.0).all()
            penalty_delta = self.regularization_weight * torch.sum(deltas)
            penalty_min = self.regularization_weight * torch.sum(torch.abs(all_.z))
            penalty = penalty_min + penalty_delta
            # track the reg loss
            self.regularization_loss(penalty.item())

            return penalty
        else:
            return 0.0


@Model.register('BCE-classification-split-neg-vol-penalty-box-model')
class BCEBoxClassificationSplitNegVolPenaltyModel(
        BCEBoxClassificationSplitNegModel):
    def get_regularization_penalty(self) -> Union[float, torch.Tensor]:

        if self.is_eval():
            return 0.0

        if self.regularization_weight > 0:
            vols = torch.exp(self.h.get_volumes(temp=self.softbox_temp))
            min_vol = 0.01
            # don't penalize if box has very less vol
            # deviation = (vols - 0.5)**2
            # large_mask = (vols > (0.01)**self.embedding_dim)
            small_mask = (vols < min_vol)
            vols = vols[small_mask]
            diff = min_vol - vols
            # penalty = self.regularization_weight * torch.sum(vols)
            penalty = self.regularization_weight * torch.sum(diff)
            # track the reg loss
            self.regularization_loss(penalty.item())

            return penalty
        else:
            return 0.0


@Model.register('BCE-classification-split-neg-vol-penalty-gumbel-box-model')
class BCEBoxClassificationSplitNegVolPenaltyGumbelModel(BCEBoxClassificationSplitNegVolPenaltyModel):
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
                 gumbel_beta=1.0) -> None:
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

        self.gumbel_beta = gumbel_beta

    def _get_triple_score(self, head: BoxTensor, tail: BoxTensor,
                          relation: BoxTensor) -> torch.Tensor:
        intersection_box = head.gumbel_intersection(tail, gumbel_beta=self.gumbel_beta)
        intersection_vol = intersection_box._log_soft_volume_adjusted(intersection_box.z,
            intersection_box.Z, temp=self.softbox_temp, gumbel_beta=self.gumbel_beta)
        tail_vol = tail._log_soft_volume_adjusted(tail.z, tail.Z,
            temp=self.softbox_temp, gumbel_beta=self.gumbel_beta)
        score = intersection_vol - tail_vol
        return score 


@Model.register('BCE-min-vol-reg-box-model')
class BCEBoxMinVolumeRegModel(BCEBoxModel):
    def get_regularization_penalty(self) -> Union[float, torch.Tensor]:

        if self.is_eval():
            return 0.0

        if self.regularization_weight > 0:
            vols = torch.exp(self.h.get_volumes(temp=self.softbox_temp))
            min_vol = 0.01
            # don't penalize if box has very less vol
            # deviation = (vols - 0.5)**2
            # large_mask = (vols > (0.01)**self.embedding_dim)
            small_mask = (vols < min_vol)
            vols = vols[small_mask]
            diff = min_vol - vols
            # penalty = self.regularization_weight * torch.sum(vols)
            penalty = self.regularization_weight * torch.sum(diff)
            # track the reg loss
            self.regularization_loss(penalty.item())

            return penalty
        else:
            return 0.0


@Model.register('BCE-max-vol-reg-box-model')
class BCEBoxMaxVolumeRegModel(BCEBoxModel):
    def get_regularization_penalty(self) -> Union[float, torch.Tensor]:

        if self.is_eval():
            return 0.0

        if self.regularization_weight > 0:
            vols = torch.exp(self.h.get_volumes(temp=self.softbox_temp))
            min_vol = 0.01
            # don't penalize if box has very less vol
            # deviation = (vols - 0.5)**2
            large_mask = (vols > (0.01)**self.embedding_dim)
            # small_mask = (vols < min_vol)
            vols = vols[large_mask]
            # diff = min_vol - vols
            penalty = self.regularization_weight * torch.sum(vols)
            # penalty = self.regularization_weight * torch.sum(diff)
            # track the reg loss
            self.regularization_loss(penalty.item())

            return penalty
        else:
            return 0.0


@Model.register('BCE-centre-distance-model')
class BCECentreDistanceBoxModel(BCEBoxModel):
    def __init__(self,
                 num_entities: int,
                 num_relations: int,
                 embedding_dim: int,
                 box_type: str = 'SigmoidBoxTensor',
                 single_box: bool = False,
                 softbox_temp: float = 10.,
                 margin: float = 1.,
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
            number_of_negative_samples=number_of_negative_samples,
            debug=debug,
            regularization_weight=regularization_weight,
            init_interval_center=init_interval_center,
            init_interval_delta=init_interval_delta)
        self.number_of_negative_samples = number_of_negative_samples
        self.centre_loss_metric = Average()
        self.loss_f_centre: torch.nn.modules._Loss = torch.nn.MarginRankingLoss(  # type: ignore
            margin=margin, reduction='mean')

    def get_scores(self, embeddings: Dict) -> torch.Tensor:
        p = self._get_triple_score(embeddings['h'], embeddings['t'],
                                   embeddings['r'])

        diff = (embeddings['h'].centre - embeddings['t'].centre).norm(
            p=1, dim=-1)
        pos_diff = torch.mean(
            diff[torch.where(embeddings['label'] == 1)]).view(1)
        neg_diff = torch.mean(
            diff[torch.where(embeddings['label'] == 0)]).view(1)

        if torch.isnan(pos_diff):
            pos_diff = torch.Tensor([0.])

        if torch.isnan(neg_diff):
            neg_diff = torch.Tensor([margin])

        return p, pos_diff, neg_diff

    def get_loss(self, scores: torch.Tensor,
                 label: torch.Tensor) -> torch.Tensor:
        log_p = scores[0]
        log1mp = log1mexp(log_p)
        logits = torch.stack([log1mp, log_p], dim=-1)
        centre_loss = self.loss_f_centre(scores[2], scores[1],
                                         torch.Tensor([1]))
        self.centre_loss_metric(centre_loss.item())
        loss = self.loss_f(logits,
                           label) + self.regularization_weight * centre_loss

        return loss

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
            'mrr': self.mrr.get_metric(reset),
            'centre_loss': self.centre_loss_metric.get_metric(reset)
        }

        return metrics


@Model.register('BCE-bayesian-box-model')
class BCEBayesianBoxModel(BCEDeltaPenaltyModel):
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
                 gumbel_beta: float = 1.) -> None:
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
            neg_samples_in_dataset_reader=neg_samples_in_dataset_reader)

        self.gumbel_beta = gumbel_beta


    def _get_triple_score(self, head: BoxTensor, tail: BoxTensor,
                          relation: BoxTensor) -> torch.Tensor:
        """ Gets score using conditionals.

        :: note: We do not need to worry about the dimentions of the boxes. If
                it can sensibly broadcast it will.
            """
        if self.is_eval():
            if len(head.data.shape) > len(tail.data.shape):
                tail.data = torch.cat(head.data.shape[-3]*[tail.data])
            elif len(head.data.shape) < len(tail.data.shape):
                head.data = torch.cat(tail.data.shape[-3]*[head.data])

        head_tail_box_vol = head.intersection_log_soft_volume(
            tail, temp=self.softbox_temp, gumbel_beta=self.gumbel_beta, bayesian=True)
        # score = tail_head_relation_box_vol - tail_relation_box.log_soft_volume(
        #    temp=self.softbox_temp)
        score = head_tail_box_vol - tail.log_soft_volume(
            temp=self.softbox_temp)

        return score


@Model.register('BCE-bayesian-classification-box-model')
class BCEBayesianClassificationBoxModel(BCEBoxClassificationModel):
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
                 gumbel_beta: float = 1.) -> None:
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
            neg_samples_in_dataset_reader=neg_samples_in_dataset_reader)

        self.gumbel_beta = gumbel_beta


    def _get_triple_score(self, head: BoxTensor, tail: BoxTensor,
                          relation: BoxTensor) -> torch.Tensor:
        """ Gets score using conditionals.

        :: note: We do not need to worry about the dimentions of the boxes. If
                it can sensibly broadcast it will.
            """
        if self.is_eval():
            if len(head.data.shape) > len(tail.data.shape):
                tail.data = torch.cat(head.data.shape[-3]*[tail.data])
            elif len(head.data.shape) < len(tail.data.shape):
                head.data = torch.cat(tail.data.shape[-3]*[head.data])

        head_tail_box_vol = head.intersection_log_soft_volume(
            tail, temp=self.softbox_temp, gumbel_beta=self.gumbel_beta, bayesian=True)
        # score = tail_head_relation_box_vol - tail_relation_box.log_soft_volume(
        #    temp=self.softbox_temp)
        score = head_tail_box_vol - tail.log_soft_volume(
            temp=self.softbox_temp)
        if len(np.where(score>0)[0]):
            breakpoint()
        return score

    def get_regularization_penalty(self) -> Union[float, torch.Tensor]:

        if self.is_eval():
            return 0.0

        if self.regularization_weight > 0:
            vols = torch.exp(self.h.get_volumes(temp=self.softbox_temp))
            min_vol = 0.01
            # don't penalize if box has very less vol
            # deviation = (vols - 0.5)**2
            # large_mask = (vols > (0.01)**self.embedding_dim)
            small_mask = (vols < min_vol)
            vols = vols[small_mask]
            diff = min_vol - vols
            # penalty = self.regularization_weight * torch.sum(vols)
            penalty = self.regularization_weight * torch.sum(diff)
            # track the reg loss
            self.regularization_loss(penalty.item())

            return penalty
        else:
            return 0.0

@Model.register('BCE-sampled-classification-box-model')
class BCESampledClassificationBoxModel(BCEBoxClassificationModel):
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
                 n_samples: int = 1,
                 sigma_init: float = 1.0
                 ) -> None:
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
            neg_samples_in_dataset_reader=neg_samples_in_dataset_reader)

        self.n_samples = n_samples
        self.sigma_init = sigma_init
        self.sigma_cov_embeddings(num_entities, embedding_dim)
        
    def sigma_cov_embeddings(self, num_embeddings: int, embedding_dim: int,):
        self.sigma = nn.Embedding(num_embeddings=num_embeddings,
                                      embedding_dim=embedding_dim,
                                      sparse=False
                                     )
        with torch.no_grad():
            self.sigma.weight.data = self.sigma_init * torch.ones_like(
                                                    self.sigma.weight.data)

    def reparam_trick(self, box, embedding_keys):
        dev = embedding_keys.device
        dist = MultivariateNormal(torch.zeros(box.data.shape[0], box.data.shape[-1]).to(dev),
                            torch.eye(box.data.shape[-1]).to(dev))

        samples = dist.sample(torch.Size([self.n_samples]))
        sample = torch.mean(samples, axis=0)
        L = torch.sqrt(self.sigma(embedding_keys)**2) #Cholesky decomposition

        shift = (L * sample).view([box.data.shape[0], 1, box.data.shape[-1]])
        shift = shift.repeat(1, box.data.shape[-2], 1) #same amount of shift in z & Z

        box.data = box.data + shift #shift z
        return box

    def get_box_embeddings_training(  # type:ignore
        self,
        h: torch.Tensor,
        r: torch.Tensor,
        t: torch.Tensor,  # type:ignore
        label: torch.Tensor,
        **kwargs) -> Dict[str, BoxTensor]:  # type: ignore
        
        head = self.reparam_trick(self.h(h), h)
        tail = self.reparam_trick(self.t(t), t)
    
        return {
            'h': head,
            't': tail,
            'r': self.r(r),
            'label': label,
        }

    def get_sigma(self) -> torch.Tensor:
        with torch.no_grad():
            all_index = torch.arange(0, self.num_entities,
                            dtype=torch.long, device=self.sigma.weight.device)
            return torch.norm(self.sigma(all_index), p='fro', dim=1)


    def get_histograms_to_log(self) -> Dict[str, torch.Tensor]:

        return {
            "Cholesky_histogram": self.get_sigma(),
            "relation_volume_histogram": self.get_r_vol(),
            "head_entity_volume_historgram": self.get_h_vol(),
            "tail_entity_volume_historgram": self.get_t_vol()
        }

    def get_regularization_penalty(self) -> Union[float, torch.Tensor]:

        if self.is_eval():
            return 0.0

        if self.regularization_weight > 0:
            vols = torch.exp(self.h.get_volumes(temp=self.softbox_temp))
            # don't penalize if box has very less vol
            mask = (vols > (0.01)**self.embedding_dim)
            vols = vols[mask]
            penalty = self.regularization_weight * torch.sum(vols)

            # track the reg loss
            self.regularization_loss(penalty.item())

            return penalty
        else:
            return 0.0


@Model.register('BCE-sampled-box-model')
class BCESampledBoxModel(BCEDeltaPenaltyModel):
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
                 n_samples: int = 1,
                 sigma_init: float = 1.0) -> None:
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
            neg_samples_in_dataset_reader=neg_samples_in_dataset_reader)
        self.n_samples = n_samples
        self.sigma_init = sigma_init
        self.num_entities = num_entities
        self.sigma_cov_embeddings(num_entities, embedding_dim)
        
    def sigma_cov_embeddings(self, num_embeddings: int, embedding_dim: int,):
        self.sigma = nn.Embedding(num_embeddings=num_embeddings,
                                      embedding_dim=embedding_dim,
                                      sparse=False
                                     )
        with torch.no_grad():
            self.sigma.weight.data = self.sigma_init * torch.ones_like(
                                                    self.sigma.weight.data)

    def reparam_trick(self,
                      box: BoxTensor,
                      embedding_keys: torch.Tensor) -> BoxTensor:
        dev = embedding_keys.device
        dist = MultivariateNormal(torch.zeros(box.data.shape[0], box.data.shape[-1]).to(dev),
                            torch.eye(box.data.shape[-1]).to(dev))

        samples = dist.sample(torch.Size([self.n_samples]))
        sample = torch.mean(samples, axis=0)
        L = torch.sqrt(self.sigma(embedding_keys)**2) #Cholesky decomposition

        shift = (L * sample).view([box.data.shape[0], 1, box.data.shape[-1]])
        shift = shift.repeat(1, box.data.shape[-2], 1) #same amount of shift in z & Z

        box.data = box.data + shift #shift z
        return box

    def get_box_embeddings_training(  # type:ignore
        self,
        h: torch.Tensor,
        r: torch.Tensor,
        t: torch.Tensor,  # type:ignore
        label: torch.Tensor,
        **kwargs) -> Dict[str, BoxTensor]:  # type: ignore
        
        head = self.reparam_trick(self.h(h), h)
        tail = self.reparam_trick(self.t(t), t)
    
        return {
            'h': head,
            't': tail,
            'r': self.r(r),
            'label': label,
        }

    def get_sigma(self) -> torch.Tensor:
        with torch.no_grad():
            all_index = torch.arange(0, self.num_entities,
                            dtype=torch.long, device=self.sigma.weight.device)
            return torch.norm(self.sigma(all_index), p='fro', dim=1)


    def get_histograms_to_log(self) -> Dict[str, torch.Tensor]:

        return {
            "Cholesky_histogram": self.get_sigma(),
            "relation_volume_histogram": self.get_r_vol(),
            "head_entity_volume_historgram": self.get_h_vol(),
            "tail_entity_volume_historgram": self.get_t_vol()
        }

    def get_regularization_penalty(self) -> Union[float, torch.Tensor]:

        if self.is_eval():
            return 0.0

        if self.regularization_weight > 0:
            vols = torch.exp(self.h.get_volumes(temp=self.softbox_temp))
            # don't penalize if box has very less vol
            mask = (vols > (0.01)**self.embedding_dim)
            vols = vols[mask]
            penalty = self.regularization_weight * torch.sum(vols)

            # track the reg loss
            self.regularization_loss(penalty.item())

            return penalty
        else:
            return 0.0
