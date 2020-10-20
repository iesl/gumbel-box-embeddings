from typing import Tuple, Dict, Any, Union
from .max_margin_models import MaxMarginConditionalModel
from .bce_models import BCEBoxModel, BCEBoxClassificationModel
from allennlp.models import Model
from allennlp.training.metrics import Average
import torch
from boxes.box_wrapper import BoxTensor


@Model.register('hard-box')
class HardBoxModel(MaxMarginConditionalModel):
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
            stiffness: float = 1.0,
            sigmoid: bool = False
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
            margin=margin,
            number_of_negative_samples=number_of_negative_samples,
            debug=debug,
            regularization_weight=regularization_weight,
            init_interval_center=init_interval_center,
            init_interval_delta=init_interval_delta)
        self.sigmoid = sigmoid

        if self.sigmoid:
            self.non_linearity = torch.nn.Sigmoid()
        else:
            self.non_linearity = torch.nn.Identity()
        self.stiffness = stiffness
        self.dist_penalty = torch.nn.L1Loss(reduction='none')

    def _get_triple_score(self, head: BoxTensor, tail: BoxTensor,
                          relation: BoxTensor) -> torch.Tensor:
        """ Do not use this function for this class and its children.
        Implement get_scores_directly

        """
        raise NotImplementedError

    def get_scores(self,
                   embeddings: Dict) -> Tuple[torch.Tensor, torch.Tensor]:
        p_violations = embeddings['p_h'].contains_violations(
            embeddings['p_t'], margin=self.margin)
        n_violations = embeddings['n_h'].does_not_contain_violations(
            embeddings['n_t'],
            margin=self.margin)  # if self.adversarial_negative:
        # with torch.no_grad():
        # weights = torch.nn.functional.log_softmax(
        # n_s * self.adv_neg_softmax_temp, dim=-1).detach()
        # n_s = weights + n_s

        return (p_violations, n_violations)

    def get_loss(self, scores: Tuple[torch.Tensor, torch.Tensor],
                 label: torch.Tensor) -> torch.Tensor:
        # max margin loss expects label to be float
        label = label.to(scores[0].dtype)
        loss = torch.sum(self.non_linearity(
            self.stiffness * scores[0])) + 0.1 * torch.sum(
                self.non_linearity(self.stiffness * scores[1]))

        return loss

    def get_ranks(self, embeddings: Dict[str, BoxTensor]) -> Any:
        if not self.is_eval():
            raise RuntimeError("get_ranks called during training")

        return {}

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:

        return {'avg_rank': 0}


@Model.register('hard-box-classification')
class HardBoxModelClassification(BCEBoxClassificationModel):
    def __init__(
            self,
            num_entities: int,
            num_relations: int,
            embedding_dim: int,
            box_type: str = 'DeltaBoxTensor',
            single_box: bool = False,
            softbox_temp: float = 10.,
            margin: float = 0.0,
            number_of_negative_samples: int = 0,
            debug: bool = False,
            regularization_weight: float = 0,
            init_interval_center: float = 0.25,
            init_interval_delta: float = 0.1,
            stiffness: float = 1.0,
            sigmoid: bool = False
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
        self.sigmoid = sigmoid

        if self.sigmoid:
            self.non_linearity = torch.nn.Sigmoid()
        else:
            self.non_linearity = torch.nn.Identity()
        self.stiffness = stiffness
        self.dist_penalty = torch.nn.L1Loss(reduction='none')

    def get_scores(self,
                   embeddings: Dict) -> Tuple[torch.Tensor, torch.Tensor]:

        p_violations = embeddings['h'].contains_violations(
            embeddings['t'],
            margin=self.margin)[torch.where(embeddings['label'] == 1)]
        n_violations = embeddings['h'].does_not_contain_violations(
            embeddings['t'],
            margin=self.margin)[torch.where(embeddings['label'] == 0)]

        return (p_violations, n_violations)

    def get_loss(self, scores: Tuple[torch.Tensor, torch.Tensor],
                 label: torch.Tensor) -> torch.Tensor:
        # max margin loss expects label to be float
        label = label.to(scores[0].dtype)
        loss = torch.mean(self.non_linearity(
            self.stiffness * scores[0])) + torch.mean(
                self.non_linearity(self.stiffness * scores[1]))

        # if not self.is_eval():
        #     with torch.no_grad():
        #         breakpoint()
        #         self.train_f1(torch.cat([scores[0], scores[1]]), torch.cat([torch.ones_like(scores[0]), torch.zeros_like(scores[1])]))

        return loss

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        if self.is_eval():
            if not self.istest:
                metrics = self.threshold_with_f1.get_metric(reset)
            else:
                p, r, f = self.test_f1.get_metric(reset)
                metrics = {'precision': p, 'recall': r, 'fscore': f}

        else:
            metrics = {}
            metrics[
                'regularization_loss'] = self.regularization_loss.get_metric(
                    reset)

        return metrics

    def _get_triple_score_hard_box(self,
                          head: BoxTensor, 
                          tail: BoxTensor,
                          per_dim_op='max',
                          accross_dim_op='max') -> torch.Tensor:
        """ Do not use this function for this class and its children.
        Implement get_scores_directly

        """
        return head.contains_violations(tail, per_dim_op=per_dim_op, 
            accross_dim_op=accross_dim_op, margin=self.margin)

    def get_ranks(self, embeddings: Dict[str, BoxTensor]) -> Any:
        if self.is_test():
            return self.get_test(embeddings)

        s = self._get_triple_score_hard_box(embeddings['h'], embeddings['t'],
                                   per_dim_op='max', accross_dim_op='max')
        # preds = torch.stack((p_s, n_s), dim=1)  # shape = (batch, 2)
        # self.valid_f1(preds, labels)
        labels = embeddings['label']
        # upate the metrics
        self.threshold_with_f1(s, labels)

        return {}


@Model.register('hard-box-classification-soft-neg')
class HardBoxModelClassificationSoftNegative(HardBoxModelClassification):
    
    def get_scores(self,
                   embeddings: Dict) -> Tuple[torch.Tensor, torch.Tensor]:

        p_violations = self._get_triple_score_hard_box(embeddings['h'],embeddings['t'], per_dim_op='max',
                          accross_dim_op='max') [torch.where(embeddings['label'] == 1)]
        n_violations = self._get_triple_score_hard_box(embeddings['t'], embeddings['h'], per_dim_op='min',
                          accross_dim_op='min')[torch.where(embeddings['label'] == 0)]

        return (p_violations, n_violations)


@Model.register('BCE-centre-distance-hard-box-model')
class BCECentreDistanceHardBoxModel(BCEBoxModel):
    def __init__(self,
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
                 stiffness: float = 1.0,
                 sigmoid: bool = False) -> None:
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

        self.sigmoid = sigmoid

        if self.sigmoid:
            self.non_linearity = torch.nn.Sigmoid()
        else:
            self.non_linearity = torch.nn.Identity()
        self.stiffness = stiffness
        self.dist_penalty = torch.nn.L1Loss(reduction='none')
        self.margin = margin
        self.number_of_negative_samples = number_of_negative_samples
        self.centre_loss_metric = Average()
        self.loss_f_centre: torch.nn.modules._Loss = torch.nn.MarginRankingLoss(  # type: ignore
            margin=margin, reduction='mean')

    def get_scores(self,
                   embeddings: Dict) -> Tuple[torch.Tensor, torch.Tensor]:

        p_violations = embeddings['h'].contains_violations(
            embeddings['t'],
            margin=self.margin)[torch.where(embeddings['label'] == 1)]
        n_violations = embeddings['h'].does_not_contain_violations(
            embeddings['t'],
            margin=self.margin)[torch.where(embeddings['label'] == 0)]

        diff = (embeddings['h'].centre - embeddings['t'].centre).norm(
            p=1, dim=-1)
        pos_diff = torch.mean(
            diff[torch.where(embeddings['label'] == 1)]).view(1)
        neg_diff = torch.mean(
            diff[torch.where(embeddings['label'] == 0)]).view(1)

        if torch.isnan(pos_diff):
            pos_diff = torch.Tensor([0.])

        if torch.isnan(neg_diff):
            neg_diff = torch.Tensor([self.margin])
        # if self.adversarial_negative:
        # with torch.no_grad():
        # weights = torch.nn.functional.log_softmax(
        # n_s * self.adv_neg_softmax_temp, dim=-1).detach()
        # n_s = weights + n_s

        return (p_violations, n_violations), (pos_diff, neg_diff)

    def get_loss(self, scores: Tuple[torch.Tensor, torch.Tensor],
                 label: torch.Tensor) -> torch.Tensor:
        # max margin loss expects label to be float
        label = label.to(scores[0][0].dtype)
        centre_loss = self.loss_f_centre(scores[1][1], scores[1][0],
                                         torch.Tensor([1]))
        self.centre_loss_metric(centre_loss.item())
        loss = torch.sum(self.non_linearity(
            self.stiffness * scores[0][1])) + 0.1 * torch.sum(
                self.non_linearity(self.stiffness * scores[0][1])
        ) + self.regularization_weight * centre_loss

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
