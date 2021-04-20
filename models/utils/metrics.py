import torch


def single_rank(scores: torch.Tensor) -> torch:
    higher_elements = torch.sum(scores[0:-1] > scores[-1])
    ties = torch.sum(scores[0:-1] == scores[-1])
    return (higher_elements + ties/2 + 1.0).item()
