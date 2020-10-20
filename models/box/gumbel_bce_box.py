from typing import Tuple, Dict, Any, Union
from .base import BaseBoxModel
from .bce_models import BCEBoxClassificationModel, BCEDeltaPenaltyModel
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


@Model.register('BCE-gumbel-sampled-classification-model')
class BCEGumbelSampledClassificationModel(BCEBoxClassificationModel):
	
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
            neg_samples_in_dataset_reader=neg_samples_in_dataset_reader)
        self.gumbel_beta = gumbel_beta
        self.n_samples = n_samples

    def reparam_trick(self,
                      box: BoxTensor,
                      gumbel_beta: float=1.,
                      n_samples: int=10) -> torch.Tensor:
        dev = box.data.device
        m = Gumbel(torch.zeros(box.data.shape[0], box.data.shape[-1]).to(dev), torch.tensor([1.0]).to(dev))
        samples = m.sample(torch.Size([n_samples]))
        sample_fwd = torch.mean(samples, axis=0)
        samples = m.sample(torch.Size([n_samples]))
        sample_bwd = -torch.mean(samples, axis=0)

        z = sample_fwd * gumbel_beta + box.z.data
        Z = sample_bwd * gumbel_beta + box.Z.data

        return BoxTensor(torch.stack((z, Z), -2))
    
    def _get_triple_score(self, head: BoxTensor, tail: BoxTensor, relation: BoxTensor) -> torch.Tensor:
        if self.is_eval():
            if len(head.data.shape) > len(tail.data.shape):
                tail.data = torch.cat(head.data.shape[-3]*[tail.data])
            elif len(head.data.shape) < len(tail.data.shape):
                head.data = torch.cat(tail.data.shape[-3]*[head.data])

        head_sample = self.reparam_trick(head, gumbel_beta=self.gumbel_beta, n_samples=self.n_samples)
        tail_sample = self.reparam_trick(tail, gumbel_beta=self.gumbel_beta, n_samples=self.n_samples)

        intersection_sample_box = head_sample.gumbel_intersection(tail_sample, gumbel_beta=self.gumbel_beta)
        intersection_box = head.gumbel_intersection(tail, gumbel_beta=self.gumbel_beta)

        intersection_volume_fwd = intersection_sample_box._log_gumbel_volume(
            intersection_sample_box.z, intersection_box.Z)
        intersection_volume_bwd = intersection_sample_box._log_gumbel_volume(
            intersection_box.z, intersection_sample_box.Z)

        
        tail_volume_fwd = tail_sample._log_gumbel_volume(tail.z, tail_sample.Z)
        tail_volume_bwd = tail_sample._log_gumbel_volume(tail_sample.z, tail.Z)
        
        # score = (intersection_volume_fwd + intersection_volume_bwd)/2 - (
        #     tail_volume_fwd + tail_volume_bwd)/2

        intersection_score = torch.logsumexp(torch.stack((intersection_volume_fwd, 
            intersection_volume_bwd)), 0)
        tail_score = torch.logsumexp(torch.stack((tail_volume_fwd, tail_volume_bwd)), 0)

        score = intersection_score - tail_score
        if len(torch.where(score>0)[0]):
            breakpoint()
        return score

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


@Model.register('BCE-gumbel-sampled-model')
class BCEGumbelSampledModel(BCEDeltaPenaltyModel):
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
            neg_samples_in_dataset_reader=neg_samples_in_dataset_reader)
        self.gumbel_beta = gumbel_beta
        self.n_samples = n_samples

    def reparam_trick(self,
                      box: BoxTensor,
                      gumbel_beta: float=1.,
                      n_samples: int=10) -> torch.Tensor:
        dev = box.data.device
        m = Gumbel(torch.zeros(box.data.shape[0], box.data.shape[-1]).to(dev), torch.tensor([1.0]).to(dev))
        samples = m.sample(torch.Size([n_samples]))
        sample_fwd = torch.mean(samples, axis=0)
        samples = m.sample(torch.Size([n_samples]))
        sample_bwd = -torch.mean(samples, axis=0)

        z = sample_fwd * gumbel_beta + box.z.data
        Z = sample_bwd * gumbel_beta + box.Z.data

        return BoxTensor(torch.stack((z, Z), -2))
    
    def _get_triple_score(self, head: BoxTensor, tail: BoxTensor, relation: BoxTensor) -> torch.Tensor:
        if self.is_eval():
            if len(head.data.shape) > len(tail.data.shape):
                tail.data = torch.cat(head.data.shape[-3]*[tail.data])
            elif len(head.data.shape) < len(tail.data.shape):
                head.data = torch.cat(tail.data.shape[-3]*[head.data])

        head_sample = self.reparam_trick(head, gumbel_beta=self.gumbel_beta, n_samples=self.n_samples)
        tail_sample = self.reparam_trick(tail, gumbel_beta=self.gumbel_beta, n_samples=self.n_samples)

        intersection_sample_box = head_sample.gumbel_intersection(tail_sample, gumbel_beta=self.gumbel_beta)
        intersection_box = head.gumbel_intersection(tail, gumbel_beta=self.gumbel_beta)

        intersection_volume_fwd = intersection_sample_box._log_gumbel_volume(
            intersection_sample_box.z, intersection_box.Z)
        intersection_volume_bwd = intersection_sample_box._log_gumbel_volume(
            intersection_box.z, intersection_sample_box.Z)

        
        tail_volume_fwd = tail_sample._log_gumbel_volume(tail.z, tail_sample.Z)
        tail_volume_bwd = tail_sample._log_gumbel_volume(tail_sample.z, tail.Z)
        
        # score = (intersection_volume_fwd + intersection_volume_bwd)/2 - (
        #     tail_volume_fwd + tail_volume_bwd)/2
        # mean of the score should be logsumexp(fwd, bwd) - log2
        intersection_score = torch.logsumexp(torch.stack((intersection_volume_fwd, 
            intersection_volume_bwd)), 0)
        tail_score = torch.logsumexp(torch.stack((tail_volume_fwd, tail_volume_bwd)), 0)
        score = intersection_score - tail_score

        if len(torch.where(score>0)[0]):
            breakpoint()
        return score


@Model.register('BCE-gumbel-sampled-movielens-model')
class GumbelSampledMovieLensModel(BCEGumbelSampledClassificationModel):
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
        self.kl_div = Average()


    def get_loss(self, scores: torch.Tensor,
                 label: torch.Tensor) -> torch.Tensor:
        log_p = scores
        log1mp = log1mexp(log_p)

        return torch.mean(-label*log_p - (1-label)*log1mp)

    def get_loss_validation(self, scores: torch.Tensor,
                 label: torch.Tensor) -> torch.Tensor:
        log_p = scores
        log1mp = log1mexp(log_p)

        kld = torch.mean(label * (torch.log(label) - log_p) + (1 - label) * (
                                                       torch.log(1-label) - log1mp))

        return kld

    def get_ranks(self, embeddings: Dict[str, BoxTensor]) -> Any:
        with torch.no_grad():
            s = self._get_triple_score(embeddings['h'], embeddings['t'],
                                   embeddings['r'])
            labels = embeddings['label']
            loss = self.get_loss_validation(s, labels)
            self.kl_div(loss.item())
        return {'kl_div': loss.item()}

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {
            'Kl_div': self.kl_div.get_metric(reset)
            }


@Model.register('BCE-bessel-exact-movielens-model')
class BesselExactMovieLensModel(GumbelSampledMovieLensModel):
    def _get_triple_score(self,
                          head: BoxTensor,
                          tail: BoxTensor,
                          relation: BoxTensor) -> torch.Tensor:
        if self.is_eval():
            if len(head.data.shape) > len(tail.data.shape):
                tail.data = torch.cat(head.data.shape[-3]*[tail.data])
            elif len(head.data.shape) < len(tail.data.shape):
                head.data = torch.cat(tail.data.shape[-3]*[head.data])

        intersection_box = head.gumbel_intersection(tail, gumbel_beta=self.gumbel_beta)
        
        intersection_vol = intersection_box._log_bessel_volume(intersection_box.z,
            intersection_box.Z, gumbel_beta=self.gumbel_beta)
        tail_vol = tail._log_bessel_volume(tail.z, tail.Z, gumbel_beta=self.gumbel_beta)

        score = intersection_vol - tail_vol
        return score


@Model.register('BCE-bessel-approx-movielens-model')
class BesselApproxMovieLensModel(GumbelSampledMovieLensModel):
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
        self.gumbel_beta = gumbel_beta
        self.softbox_temp = softbox_temp
        # self.softbox_temp = min(1/self.gumbel_beta, 7)

    def _get_triple_score(self,
                          head: BoxTensor,
                          tail: BoxTensor,
                          relation: BoxTensor) -> torch.Tensor:
        if self.is_eval():
            if len(head.data.shape) > len(tail.data.shape):
                tail.data = torch.cat(head.data.shape[-3]*[tail.data])
            elif len(head.data.shape) < len(tail.data.shape):
                head.data = torch.cat(tail.data.shape[-3]*[head.data])

        intersection_box = head.gumbel_intersection(tail, gumbel_beta=self.gumbel_beta)
        intersection_vol = intersection_box._log_soft_volume_adjusted(intersection_box.z,
            intersection_box.Z, temp=self.softbox_temp, gumbel_beta=self.gumbel_beta)
        tail_vol = tail._log_soft_volume_adjusted(tail.z, tail.Z,
            temp=self.softbox_temp, gumbel_beta=self.gumbel_beta)

        score = intersection_vol - tail_vol

        return score


@Model.register('BCE-bessel-approx-classification-model')
class BCEBesselApproxClassificationModel(BCEBoxClassificationModel):
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
            neg_samples_in_dataset_reader=neg_samples_in_dataset_reader)
        self.gumbel_beta = gumbel_beta
        self.softbox_temp = softbox_temp
        # self.softbox_temp = min(1/self.gumbel_beta, 7)

    def _get_triple_score(self,
                          head: BoxTensor,
                          tail: BoxTensor,
                          relation: BoxTensor) -> torch.Tensor:
        if self.is_eval():
            if len(head.data.shape) > len(tail.data.shape):
                tail.data = torch.cat(head.data.shape[-3]*[tail.data])
            elif len(head.data.shape) < len(tail.data.shape):
                head.data = torch.cat(tail.data.shape[-3]*[head.data])

        intersection_box = head.gumbel_intersection(tail, gumbel_beta=self.gumbel_beta)
        intersection_vol = intersection_box._log_soft_volume_adjusted(intersection_box.z,
            intersection_box.Z, temp=self.softbox_temp, gumbel_beta=self.gumbel_beta)
        tail_vol = tail._log_soft_volume_adjusted(tail.z, tail.Z,
            temp=self.softbox_temp, gumbel_beta=self.gumbel_beta)

        score = intersection_vol - tail_vol

        return score


@Model.register('BCE-bessel-approx-model')
class BCEBesselApproxModel(BCEDeltaPenaltyModel):
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
            neg_samples_in_dataset_reader=neg_samples_in_dataset_reader)
        self.gumbel_beta = gumbel_beta
        self.softbox_temp = softbox_temp
        # self.softbox_temp = min(1/self.gumbel_beta, 7)

    def _get_triple_score(self,
                          head: BoxTensor,
                          tail: BoxTensor,
                          relation: BoxTensor) -> torch.Tensor:
        if self.is_eval():
            if len(head.data.shape) > len(tail.data.shape):
                tail.data = torch.cat(head.data.shape[-3]*[tail.data])
            elif len(head.data.shape) < len(tail.data.shape):
                head.data = torch.cat(tail.data.shape[-3]*[head.data])

        intersection_box = head.gumbel_intersection(tail, gumbel_beta=self.gumbel_beta)
        intersection_vol = intersection_box._log_soft_volume_adjusted(intersection_box.z,
            intersection_box.Z, temp=self.softbox_temp, gumbel_beta=self.gumbel_beta)
        tail_vol = tail._log_soft_volume_adjusted(tail.z, tail.Z,
            temp=self.softbox_temp, gumbel_beta=self.gumbel_beta)

        score = intersection_vol - tail_vol

        return score


@Model.register('BCE-bessel-exact-model')
class BCEBesselExactModel(BCEBesselApproxModel):
    def _get_triple_score(self,
                          head: BoxTensor,
                          tail: BoxTensor,
                          relation: BoxTensor) -> torch.Tensor:
        if self.is_eval():
            if len(head.data.shape) > len(tail.data.shape):
                tail.data = torch.cat(head.data.shape[-3]*[tail.data])
            elif len(head.data.shape) < len(tail.data.shape):
                head.data = torch.cat(tail.data.shape[-3]*[head.data])

        intersection_box = head.gumbel_intersection(tail, gumbel_beta=self.gumbel_beta)
        
        intersection_vol = intersection_box._log_bessel_volume(intersection_box.z,
            intersection_box.Z, gumbel_beta=self.gumbel_beta)
        tail_vol = tail._log_bessel_volume(tail.z, tail.Z, gumbel_beta=self.gumbel_beta)

        score = intersection_vol - tail_vol
        return score


@Model.register('BCE-bessel-exact-classification-model')
class BCEBesselExactClassificationModel(BCEBesselApproxClassificationModel):
    def _get_triple_score(self,
                          head: BoxTensor,
                          tail: BoxTensor,
                          relation: BoxTensor) -> torch.Tensor:
        if self.is_eval():
            if len(head.data.shape) > len(tail.data.shape):
                tail.data = torch.cat(head.data.shape[-3]*[tail.data])
            elif len(head.data.shape) < len(tail.data.shape):
                head.data = torch.cat(tail.data.shape[-3]*[head.data])

        intersection_box = head.gumbel_intersection(tail, gumbel_beta=self.gumbel_beta)
        
        intersection_vol = intersection_box._log_bessel_volume(intersection_box.z,
            intersection_box.Z, gumbel_beta=self.gumbel_beta)
        tail_vol = tail._log_bessel_volume(tail.z, tail.Z, gumbel_beta=self.gumbel_beta)

        score = intersection_vol - tail_vol
        return score