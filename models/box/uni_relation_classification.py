from typing import Tuple, Dict, Any
from .base import BaseBoxModel
from typing import Optional
from allennlp.models import Model
from allennlp.training.metrics import Average, F1Measure
from boxes.box_wrapper import SigmoidBoxTensor, BoxTensor
from boxes.modules import BoxEmbedding
from allennlp.modules.token_embedders import Embedding
import torch
import numpy as np
from ..metrics import HitsAt10


@Model.register('uni-relation-model-classification')
class UniRelationalModelClassifiaction(BaseBoxModel):
    def create_embeddings_layer(self,
                                num_entities: int,
                                num_relations: int,
                                embedding_dim: int,
                                single_box: bool = False) -> None:

        self.entity_head = BoxEmbedding(
            num_embeddings=num_entities,
            box_embedding_dim=embedding_dim,
            box_type=self.box_type,
            sparse=True)
        if not single_box:
            self.entity_tail =BoxEmbedding(
                num_embeddings=num_entities,
                box_embedding_dim=embedding_dim,
                box_type=self.box_type,
                sparse=True)
        else:
            self.entity_tail = self.entity_head

    def __init__(
            self,
            num_entities: int,
            num_relations: int,
            embedding_dim: int,
            box_type: str = 'SigmoidBoxTensor',
            softbox_temp: float = 10.,
            single_box: bool = False,
            margin: float = 0.0,
            vocab: Optional[None] = None,
            debug: bool = False
            # we don't need vocab but some api relies on its presence as an argument
    ) -> None:
        super().__init__()
        self.debug = debug
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.embedding_dim = embedding_dim
        self.box_type = box_type
        self.create_embeddings_layer(num_entities, num_relations,
                                     embedding_dim, single_box)
        self.loss_f = torch.nn.MarginRankingLoss(  # type: ignore
            margin=margin, reduction='mean')
        self.softbox_temp = softbox_temp
        self.margin = margin
        self.f1 = F1Measure(1)

    def get_box_embeddings_training(  # type:ignore
            self,
            p_h: torch.Tensor,
            p_r: torch.Tensor, 
            p_t: torch.Tensor, # type:ignore
            n_h: torch.Tensor,
            n_r: torch.Tensor,
            n_t: torch.Tensor,
            **kwargs) -> Dict[str, BoxTensor]:  # type: ignore

        if self.debug:
            with torch.no_grad():
                if sum([(lambda x: np.any((x >= self.num_entities).numpy()))(x)
                        for x in [p_h, p_t, n_h, n_t]]) > 0:
                    breakpoint()

                if sum(
                    [(lambda x: np.any((x >= self.num_relations).numpy()))(x)
                     for x in [p_r, n_r]]) > 0:
                    breakpoint()

        return {
            'p_h': self.entity_head(p_h),
            'p_t': self.entity_tail(p_t),
            'n_h': self.entity_head(n_h),
            'n_t': self.entity_tail(n_t),
        }

    def get_box_embeddings_val(self, 
                               h: torch.Tensor,
                               t: torch.Tensor,
                               r: torch.Tensor,
                               label: torch.Tensor) -> Dict[str, torch.tensor]:

        if not self.is_eval():
            raise RuntimeError("get_box_embeddings_val called during training")
        with torch.no_grad():
            embs = {
                'head': self.entity_head(h),
                'tail': self.entity_tail(t),
                'label': label
            }
        return embs

    def get_ranks(self, embeddings: Dict[str, torch.tensor]) -> Any:

        if not self.is_eval():
            raise RuntimeError("get_ranks called during training")
        with torch.no_grad():
            b = embeddings
            t_given_h_pos, t_given_h_neg = b['tail'].log_conditional_prob(b['head'], temp=self.softbox_temp)
            pred = torch.stack([t_given_h_neg, t_given_h_pos],-1)

            self.f1(pred, b['label'])

            return {
                'pred': pred
            }

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:

        if not self.training:
            return {
               'f1': self.f1.get_metric(reset)[-1]
            }
        else:
            return {}

    def get_scores(self,
                   embeddings: Dict) -> Tuple[torch.Tensor, torch.Tensor]:

        p_s = embeddings['p_h'].intersection_log_soft_volume(embeddings['p_t'], temp=self.softbox_temp)
        n_s = embeddings['n_h'].intersection_log_soft_volume(embeddings['n_t'], temp=self.softbox_temp)

        return (p_s, n_s)

    def get_loss(self, scores: Tuple[torch.Tensor, torch.Tensor],
                 label: torch.Tensor) -> torch.Tensor:
        # max margin loss expects label to be float
        label = label.to(scores[0].dtype)

        return self.loss_f(*scores, label)

@Model.register('uni-relation-model-classification-conditional')
class UniRelationalModelConditionalMaxMargin(UniRelationalModelClassifiaction):
    def __init__(self,
                num_entities: int,
                num_relations: int,
                embedding_dim: int,
                box_type: str = 'SigmoidBoxTensor',
                softbox_temp: float = 10.,
                single_box: bool = False,
                margin: float = 0.0,
                vocab: Optional[None] = None,
                debug: bool = False
                ) -> None:
        super().__init__(
                num_entities,
                num_relations,
                embedding_dim,
                box_type=box_type,
                softbox_temp=softbox_temp,
                single_box=single_box,
                margin=margin,
                vocab=None,
                debug=False
            )

    def get_scores(self,
                   embeddings: Dict) -> Tuple[torch.Tensor, torch.Tensor]:
        p_s_intersection = embeddings['p_h'].intersection_log_soft_volume(embeddings['p_t'], temp=self.softbox_temp)
        n_s_intersection = embeddings['n_h'].intersection_log_soft_volume(embeddings['n_t'], temp=self.softbox_temp)
        p_s = p_s_intersection - embeddings['p_h'].log_soft_volume(temp=self.softbox_temp)
        n_s = n_s_intersection - embeddings['n_h'].log_soft_volume(temp=self.softbox_temp)
        return (p_s, n_s)

@Model.register('uni-relation-bce-model-classification')
class UniRelationalModelBce(UniRelationalModelClassifiaction):
    def __init__(self,
                num_entities: int,
                num_relations: int,
                embedding_dim: int,
                box_type: str = 'SigmoidBoxTensor',
                softbox_temp: float = 10.,
                single_box: bool = False,
                margin: float = 0.0,
                vocab: Optional[None] = None,
                debug: bool = False
                ) -> None:
        super().__init__(
                num_entities,
                num_relations,
                embedding_dim,
                box_type=box_type,
                softbox_temp=softbox_temp,
                single_box=single_box,
                margin=margin,
                vocab=None,
                debug=False
            )

    def get_box_embeddings_training(  # type:ignore
            self,
            h: torch.Tensor,
            t: torch.Tensor,  # type:ignore
            r: torch.Tensor,
            label: torch.Tensor,
            **kwargs) -> Dict[str, BoxTensor]:  # type: ignore

        return {
            'h': self.entity_head(h),
            't': self.entity_tail(t),
            'label': label,
        }

    def loss_f(self, pos_score, neg_score:Tuple[torch.Tensor, torch.Tensor],
               label:torch.Tensor) -> torch.Tensor:
        loss =   -(1 - label) * neg_score - label * pos_score
        return torch.mean(loss, 0)

    def get_scores(self,
                   embeddings: Dict) -> Tuple[torch.Tensor, torch.Tensor]:
        t_given_h_pos, t_given_h_neg = embeddings['t'].log_conditional_prob(
                                embeddings['h'], temp=self.softbox_temp)
        return (t_given_h_pos, t_given_h_neg)

@Model.register('uni-relation-bceexp-model-classification')
class UniRelationalModelBceExp(UniRelationalModelBce):
    def __init__(self,
                num_entities: int,
                num_relations: int,
                embedding_dim: int,
                box_type: str = 'SigmoidBoxTensor',
                softbox_temp: float = 10.,
                single_box: bool = False,
                margin: float = 0.0,
                vocab: Optional[None] = None,
                debug: bool = False
                ) -> None:
        super().__init__(
                num_entities,
                num_relations,
                embedding_dim,
                box_type=box_type,
                softbox_temp=softbox_temp,
                single_box=single_box,
                margin=margin,
                vocab=None,
                debug=False
            )

        self.loss_f_bce = torch.nn.BCELoss(reduction='mean')


    def get_scores(self,
                   embeddings: Dict) -> Tuple[torch.Tensor, torch.Tensor]:

        intersection = embeddings['h'].intersection_log_soft_volume(
            embeddings['t'], temp=self.softbox_temp)
        score = torch.exp(intersection
            - embeddings['h'].log_soft_volume(temp=self.softbox_temp))
        return score


    def get_loss(self, scores: torch.Tensor,
                 label: torch.Tensor) -> torch.Tensor:
        # max margin loss expects label to be float
        label = label.to(scores.dtype)
        return self.loss_f_bce(scores, label)


@Model.register('uni-relation-bcenll-model-classification')
class UniRelationalModelBceNll(UniRelationalModelBce):
    def __init__(self,
                num_entities: int,
                num_relations: int,
                embedding_dim: int,
                box_type: str = 'SigmoidBoxTensor',
                softbox_temp: float = 10.,
                single_box: bool = False,
                margin: float = 0.0,
                vocab: Optional[None] = None,
                debug: bool = False
                ) -> None:
        super().__init__(
                num_entities,
                num_relations,
                embedding_dim,
                box_type=box_type,
                softbox_temp=softbox_temp,
                single_box=single_box,
                margin=margin,
                vocab=None,
                debug=False
            )

        self.loss_f_nll = torch.nn.NLLLoss(reduction='mean')


    def get_scores(self,
                   embeddings: Dict) -> Tuple[torch.Tensor, torch.Tensor]:

        t_given_h_pos, t_given_h_neg = embeddings['t'].log_conditional_prob(
                                embeddings['h'], temp=self.softbox_temp)
        return (t_given_h_neg, t_given_h_pos)


    def get_loss(self, scores: Tuple[torch.Tensor, torch.Tensor],
                 label: torch.Tensor) -> torch.Tensor:
        # max margin loss expects label to be float
        # label = label.to(scores[0].dtype)
        scores = torch.stack(scores, -1)
        
        return self.loss_f_nll(scores, label)
