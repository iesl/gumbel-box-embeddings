from typing import Tuple, Dict, Any
from .base import BaseBoxModel
from typing import Optional
from allennlp.models import Model
from allennlp.training.metrics import Average
from boxes.box_wrapper import SigmoidBoxTensor, BoxTensor
from boxes.modules import BoxEmbedding
from allennlp.modules.token_embedders import Embedding
import torch
import numpy as np
from ..metrics import HitsAt10


@Model.register('uni-relation-model-head-tail')
class UniRelationModel(BaseBoxModel):
    def create_embeddings_layer(self, num_entities: int, num_relations: int,
                                embedding_dim: int) -> None:
        self.entity_head = BoxEmbedding(
            num_embeddings=num_entities,
            box_embedding_dim=embedding_dim,
            box_type=self.box_type,
            sparse=True)
        self.entity_tail =BoxEmbedding(
            num_embeddings=num_entities,
            box_embedding_dim=embedding_dim,
            box_type=self.box_type,
            sparse=True)

    def __init__(
            self,
            num_entities: int,
            num_relations: int,
            embedding_dim: int,
            box_type: str = 'SigmoidBoxTensor',
            softbox_temp: float = 10.,
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
                                     embedding_dim)
        self.loss_f = torch.nn.MarginRankingLoss(  # type: ignore
            margin=margin, reduction='mean')
        self.softbox_temp = softbox_temp
        self.margin = margin
        # used only during eval
        self.precesion_parent = Average()
        self.recall_parent = Average()
        self.precesion_child = Average()
        self.recall_child =Average()

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
                               node: torch.Tensor,
                               gt_parent: torch.Tensor,
                               gt_child: torch.Tensor) -> Dict[str, torch.tensor]:

        if not self.is_eval():
            raise RuntimeError("get_box_embeddings_val called during training")
        with torch.no_grad():
            embs = {
                'node_head': self.entity_head(node),
                'node_tail': self.entity_tail(node),
                'gt_child': gt_child,
                'gt_parent': gt_parent
            }
        return embs

    def get_ranks(self, embeddings: Dict[str, torch.tensor]) -> Any:

        if not self.is_eval():
            raise RuntimeError("get_ranks called during training")
        with torch.no_grad():
            b = embeddings
            intersection_prob_head = b['node_head'].intersection_log_soft_volume(
                self.entity_tail(torch.arange(self.num_entities).unsqueeze(0)), temp=self.softbox_temp)

            intersection_prob_tail = b['node_tail'].intersection_log_soft_volume(
                self.entity_head(torch.arange(self.num_entities).unsqueeze(0)), temp=self.softbox_temp)  

            score_parent = (intersection_prob_tail - self.entity_tail(
                torch.arange(self.num_entities)).log_soft_volume(temp=self.softbox_temp)).reshape(-1)
            score_child = (intersection_prob_head - b['node_head'].log_soft_volume(temp=self.softbox_temp)).reshape(-1)

            parent_set = torch.zeros(self.num_entities)
            child_set = torch.zeros(self.num_entities)
            
            parent = (torch.where(
                score_parent > np.log(0.5), score_parent, parent_set)).nonzero().reshape(-1)
            child = (torch.where(
                score_child > np.log(0.5), score_child, child_set)).nonzero().reshape(-1)
            # import pdb; pdb.set_trace()
            parent = set(np.asarray(parent))
            child = set(np.asarray(child))
            gt_parent = set(np.asarray(b['gt_parent']).reshape(-1))
            gt_child = set(np.asarray(b['gt_child']).reshape(-1))
            
            try:
                p_p = len(parent.intersection(gt_parent))/len(gt_parent)
                r_p = len(parent.intersection(gt_parent))/len(parent)
                p_c = len(child.intersection(gt_child))/len(gt_child)
                r_c = len(child.intersection(gt_child))/len(child)
            except:
                p_p = 0.0
                r_p = 0.0
                p_c = 0.0
                r_c = 0.0

            self.precesion_parent(p_p)
            self.recall_parent(r_p)
            self.precesion_child(p_c)
            self.recall_child(r_c)

            return {
                'p_p': p_p,
                'r_p': r_p,
                'p_c': p_c,
                'r_c': r_c
            }

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:

        return {
            'precesion_parent': self.precesion_parent.get_metric(reset),
            'recall_parent': self.recall_parent.get_metric(reset),
            'precesion_child': self.precesion_child.get_metric(reset),
            'recall_child': self.recall_child.get_metric(reset)
        }

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

@Model.register('uni-relation-model-head-tail-conditional-training')
class UniRelationalModelConditionalMaxMargin(UniRelationModel):
    def __init__(self,
                num_entities: int,
                num_relations: int,
                embedding_dim: int,
                box_type: str = 'SigmoidBoxTensor',
                softbox_temp: float = 10.,
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
