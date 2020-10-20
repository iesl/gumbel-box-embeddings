from typing import Callable, Dict, Union, Optional, Tuple, List
from allennlp.models import Model
from allennlp.modules.token_embedders.embedding import Embedding
from typing import Any
import torch
from boxes.box_wrapper import SigmoidBoxTensor, BoxTensor
from boxes.modules import BoxEmbedding
from allennlp.training.metrics import Average
from ..metrics import HitsAt10, HitsAt1, HitsAt3


class TensorBoardLoggable(object):
    def get_scalars_to_log(self) -> Dict[str, torch.Tensor]:
        raise NotImplementedError

    def get_histograms_to_log(self) -> Dict[str, torch.Tensor]:
        raise NotImplementedError


class BaseModel(Model, TensorBoardLoggable):
    head_names = ['h', 'p_h', 'n_h', 'tr_h', 'hr_e']
    tail_names = ['t', 'p_t', 'n_t', 'tr_e', 'hr_t']
    relation_names = ['r', 'hr_r', 'tr_r', 'p_r', 'n_r']

    def create_entity_embedding_layer(
            self, num_entities: int, embedding_dim: int, *args: Any,
            **kwargs: Any) -> Union[BoxEmbedding, Embedding]:
        raise NotImplementedError

    def create_relations_embedding_layer(
            self, num_entities: int, embedding_dim: int, *args: Any,
            **kwargs: Any) -> Union[BoxEmbedding, Embedding]:
        raise NotImplementedError

    def create_embeddings_layer(self, num_entities: int, num_relations: int,
                                embedding_dim: int, entity_kwargs: Dict,
                                relations_kwargs: Dict) -> None:
        raise NotImplementedError

    def create_transform_layers(self, *args: Any, **kwargs: Any) -> None:
        pass


class BaseBoxModel(BaseModel):
    head_names = ['h', 'p_h', 'n_h', 'tr_h', 'hr_e']
    tail_names = ['t', 'p_t', 'n_t', 'tr_e', 'hr_t']
    relation_names = ['r', 'hr_r', 'tr_r', 'p_r', 'n_r']

    def create_entity_embedding_layer(self, num_entities, embedding_dim,
                                      box_type, sparse, init_interval_center,
                                      init_interval_delta) -> BoxEmbedding:

        return BoxEmbedding(
            num_embeddings=num_entities,
            box_embedding_dim=embedding_dim,
            box_type=self.box_type,
            sparse=False,
            init_interval_center=init_interval_center,
            init_interval_delta=init_interval_delta)

    def create_relations_embedding_layer(
            self, num_entities, embedding_dim, box_type, sparse,
            init_interval_center, init_interval_delta) -> BoxEmbedding:

        return BoxEmbedding(
            num_embeddings=num_entities,
            box_embedding_dim=embedding_dim,
            box_type=self.box_type,
            sparse=False,
            init_interval_center=init_interval_center,
            init_interval_delta=init_interval_delta)

    def create_embeddings_layer(self, num_entities: int, num_relations: int,
                                embedding_dim: int, single_box: bool,
                                entities_init_interval_center: float,
                                entities_init_interval_delta: float,
                                relations_init_interval_center: float,
                                relations_init_interval_delta: float) -> None:
        self.h = self.create_entity_embedding_layer(
            num_entities, embedding_dim, self.box_type, False,
            entities_init_interval_center, entities_init_interval_delta)

        if not single_box:
            self.t = self.create_entity_embedding_layer(
                num_entities, embedding_dim, self.box_type, False,
                entities_init_interval_center, entities_init_interval_delta)
        else:
            self.t = self.h

        self.r = self.create_relations_embedding_layer(
            num_relations, embedding_dim, self.box_type, False,
            relations_init_interval_center, relations_init_interval_delta)
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

    def __init__(
            self,
            num_entities: int,
            num_relations: int,
            embedding_dim: int,
            box_type: str = "SigmoidBoxTensor",
            single_box: bool = False,
            softbox_temp: float = 10.,
            number_of_negative_samples: int = 0,
            debug: bool = False,
            regularization_weight: float = 0,
            init_interval_center: float = 0.25,
            init_interval_delta: float = 0.1,
    ) -> None:
        super().__init__(vocab=None)
        self.debug = debug
        self.number_of_negative_samples = number_of_negative_samples
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.embedding_dim = embedding_dim
        self.box_type = box_type
        self.softbox_temp = softbox_temp
        self.init_interval_delta = init_interval_delta
        self.init_interval_center = init_interval_center
        self.single_box = single_box
        self.create_embeddings_layer(num_entities, num_relations,
                                     embedding_dim, single_box,
                                     init_interval_center, init_interval_delta,
                                     init_interval_center, init_interval_delta)

        self.regularization_weight = regularization_weight
        self.regularization_loss = Average()
        self.head_replacement_rank_avg = Average()
        self.tail_replacement_rank_avg = Average()
        self.avg_rank = Average()
        self.hitsat10 = HitsAt10()
        self.head_hitsat1 = HitsAt1()
        self.tail_hitsat1 = HitsAt1()
        self.head_hitsat3 = HitsAt3()
        self.tail_hitsat3 = HitsAt3()
        self.head_replacement_mrr = Average()
        self.tail_replacement_mrr = Average()
        self.mrr = Average()
        self.int_volume_train = Average()
        self.int_volume_dev = Average()

    def is_eval(self) -> bool:
        return not self.training

    def get_appropriate_embedding(self, name: str, idx_tensor: torch.Tensor
                                  ) -> Union[torch.Tensor, BoxTensor]:

        if name in self.appropriate_emb:
            return self.appropriate_emb[name](idx_tensor)  # type:ignore
        else:
            raise ValueError(
                "{} is not in self.appropriate_emb. ".format(name))

    def get_box_embeddings_val(  # type: ignore
            self, **kwargs
    ) -> Dict[str, Union[BoxTensor, torch.Tensor]]:  # type: ignore

        return {
            name: self.get_appropriate_embedding(name, idx_tensor)

            for name, idx_tensor in kwargs.items()
        }

    def get_box_embeddings_training(  # type: ignore
            self, **kwargs
    ) -> Dict[str, Union[BoxTensor, torch.Tensor]]:  # type: ignore

        return {
            name: self.get_appropriate_embedding(name, idx_tensor)

            for name, idx_tensor in kwargs.items()
        }

    def get_expected_head(
            self, kwargs: Dict[str, torch.Tensor]) -> Tuple[str, torch.Tensor]:
        head = None
        name = None

        for expected_name in self.head_names:
            if expected_name in kwargs:
                if head is not None:
                    raise ValueError(
                        "Multiple head entities in single Instance")
                head = kwargs[expected_name]
                name = expected_name

        if (head is None) or name is None:
            raise ValueError

        return name, head

    def get_expected_tail(
            self, kwargs: Dict[str, torch.Tensor]) -> Tuple[str, torch.Tensor]:
        tail = None
        name = None

        for expected_name in self.tail_names:
            if expected_name in kwargs:
                if tail is not None:
                    raise ValueError(
                        "Multiple tail entities in single Instance")
                tail = kwargs[expected_name]
                name = expected_name

        if (tail is None) or name is None:
            raise ValueError

        return name, tail

    def get_expected_relation(
            self, kwargs: Dict[str, torch.Tensor]) -> Tuple[str, torch.Tensor]:
        rel = None
        name = None

        for expected_name in self.relation_names:
            if expected_name in kwargs:
                if rel is not None:
                    raise ValueError(
                        "Multiple relation entities in single Instance")
                rel = kwargs[expected_name]
                name = expected_name

        if (rel is None) or name is None:
            raise ValueError

        return name, rel

    def get_random_entities(self, size_: int, device: torch.device,
                            dtype: torch.dtype) -> torch.Tensor:
        # create the tensors for negatives
        with torch.no_grad():
            negatives = torch.randint(
                low=0,
                high=self.num_entities,
                size=(size_, ),
                dtype=dtype,
                device=device)

        return negatives

    def fill_random_entities_(self, inp: torch.Tensor) -> torch.Tensor:
        return inp.random_(0, self.num_entities)

    def repeat(self, t: torch.Tensor, times: int) -> torch.Tensor:
        return t.repeat(times)

    def batch_with_negative_samples(self, **kwargs) -> Dict[str, torch.Tensor]:
        raise NotImplementedError

    def forward(self, **kwargs):  # type: ignore

        if self.is_eval():
            embeddings = self.get_box_embeddings_val(**kwargs)
            ranks = self.get_ranks(embeddings)

            if 'label' in kwargs:
                #s = self.get_scores(embeddings)
                #ranks['loss'] = self.get_loss(s, kwargs['label'])
                ranks['loss'] = None
            else:
                ranks['loss'] = None

            return ranks
        # Do negative sampling if required

        if self.number_of_negative_samples > 0:
            kwargs = self.batch_with_negative_samples(**kwargs)

        with torch.autograd.set_detect_anomaly(self.debug):
            label = kwargs.get('label', None)
            embeddings = self.get_box_embeddings_training(**kwargs)
            scores = self.get_scores(embeddings)

            if label is not None:
                loss = self.get_loss(scores, label)
            else:
                loss = None

        return {'loss': loss, 'scores': scores}

    def get_regularization_penalty(self) -> Union[float, torch.Tensor]:
        if self.is_eval():
            return 0.0

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

            relation_penalty = self.r.get_bounding_box().log_soft_volume(
                temp=self.softbox_temp)

            if (relation_penalty < 0).all():
                relation_penalty = relation_penalty * 0

            reg_loss = (self.regularization_weight *
                        (entity_penalty + relation_penalty))
            # track the reg loss
            self.regularization_loss(reg_loss.item())

            return reg_loss
        else:
            return 0.0

    def get_scalars_to_log(self) -> Dict[str, torch.Tensor]:
        return {}

    def get_r_vol(self) -> torch.Tensor:
        with torch.no_grad():
            v = self.r.get_volumes(temp=self.softbox_temp)

        return v

    def get_h_vol(self) -> torch.Tensor:
        with torch.no_grad():
            v = self.h.get_volumes(temp=self.softbox_temp)

        return v

    def get_t_vol(self) -> torch.Tensor:
        with torch.no_grad():
            v = self.t.get_volumes(temp=self.softbox_temp)

        return v

    def get_histograms_to_log(self) -> Dict[str, torch.Tensor]:

        return {
            "relation_volume_histogram": self.get_r_vol(),
            "head_entity_volume_historgram": self.get_h_vol(),
            "tail_entity_volume_historgram": self.get_t_vol()
        }

    def get_scores(self, embeddings: Any) -> Any:
        """Given box embeddings returns the score for positive and
        negative samples"""
        raise NotImplementedError

    def get_ranks(self, embeddings: Any) -> Dict:
        raise NotImplementedError

    def get_loss(self, scores: Any, label: Any) -> Any:  # type: ignore
        raise NotImplementedError
