from typing import Tuple, Dict, Any, Union

import torch
from allennlp.models import Model
from torch import nn

from boxes.box_wrapper import SigmoidBoxTensor, BoxTensor, DeltaBoxTensor
from .kbc_models import BCEAffineTransformGumbelKbcModel


box_types = {
    'SigmoidBoxTensor': SigmoidBoxTensor,
    'DeltaBoxTensor': DeltaBoxTensor,
    'BoxTensor': BoxTensor
}
@Model.register('BCE-translation-gumbel-kbc-model')
class BCETranslationGumbelKbcModel(BCEAffineTransformGumbelKbcModel):
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
        self.get_relation_embeddings(num_relations, embedding_dim)

    def get_relation_embeddings(self, num_embeddings: int, embedding_dim: int):
        
        self.relation_delta_bias = nn.Embedding(num_embeddings=num_embeddings,
                                                embedding_dim=embedding_dim,
                                                sparse= False)
        
        self.relation_min_bias = nn.Embedding(num_embeddings=num_embeddings,
                                              embedding_dim=embedding_dim,
                                              sparse= False)

        nn.init.xavier_uniform_(self.relation_delta_bias.weight.data)
        nn.init.xavier_uniform_(self.relation_min_bias.weight.data)
    
    def get_relation_transform(self, box: BoxTensor, relation: torch.Tensor):
        transformed_box = box_types[self.box_type](box.data.clone())
        bias_delta = self.relation_delta_bias(relation)
        bias_min = self.relation_min_bias(relation)
        if len(box.data.shape) == 3:
           transformed_box.data[:,0,:] = transformed_box.data[:,0,:].clone() + bias_min
           transformed_box.data[:,1,:] = nn.functional.softplus(transformed_box.data[:,1,:].clone()  + bias_delta)
        else:
           transformed_box.data[:,:,0,:] = transformed_box.data[:,:,0,:].clone()  + bias_min
           transformed_box.data[:,:,1,:] = nn.functional.softplus(transformed_box.data[:,:,1,:].clone() + bias_delta)
        return transformed_box

