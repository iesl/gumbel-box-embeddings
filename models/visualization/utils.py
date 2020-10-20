from dataclasses import dataclass
from typing import Optional, Mapping, Tuple, List, Dict, Any
from itertools import cycle
import boxes.modules as bm
import torch
from copy import deepcopy
import plotly.graph_objs as go

euler_gamma = 0.57721566490153286060
beta = 2.0

# These are Seaborn's paired colors, for reference:
sns_paired_colors = [
    "#a6cee3",
    "#1f78b4",
    "#b2df8a",
    "#33a02c",
    "#fb9a99",
    "#e31a1c",
    "#fdbf6f",
    "#ff7f00",
    "#cab2d6",
    "#6a3d9a",
    "#ffff99",
    "#b15928",
]

#
plotly_colors = [
    '#1f77b4',  # muted blue
    '#ff7f0e',  # safety orange
    '#2ca02c',  # cooked asparagus green
    '#d62728',  # brick red
    '#9467bd',  # muted purple
    '#8c564b',  # chestnut brown
    '#e377c2',  # raspberry yogurt pink
    '#7f7f7f',  # middle gray
    '#bcbd22',  # curry yellow-green
    '#17becf',  # blue-teal
    '#17b4cf',
    '#17b41f',
    '#11b4cf',
    # blue-teal
]


def colors():
    return cycle(sns_paired_colors + plotly_colors)


@dataclass
class BoxPosition:
    x0: Optional[float] = None
    x1: Optional[float] = None
    y0: Optional[float] = None
    y1: Optional[float] = None

    @property
    def z(self):
        return torch.tensor([self.x0, self.y0])

    @property
    def Z(self):
        return torch.tensor([self.x1, self.y1])


@dataclass
class Node:
    """Contains the all the information pertaining to a
    particular node in the graph, including its box"""
    name: Optional[str] = None
    idx: Optional[int] = None
    box: Optional[BoxPosition] = None
    color: Optional[str] = None

    def copy_and_update(self, name=None, idx=None, box=None, color=None):
        """Create a copy and update it """
        copy = deepcopy(self)
        copy.name = name
        copy.idx = idx
        copy.box = box
        copy.color = color

        return copy

    def go_box_args(self):
        box = self.box

        return dict(
            x=[box.x0, box.x0, box.x1, box.x1, box.x0],
            y=[box.y0, box.y1, box.y1, box.y0, box.y0],
            mode="lines",
            fill="toself",
            name=self.name,
            marker={"color": self.color},
            showlegend=True)

    def go_box(self):

        # actual box boundary

        return go.Scatter(**self.go_box_args())


def get_box_from_module(box_embedding_module: bm.TBoxTensor,
                        idx: Optional[int] = None) -> List:
    """ Assumes the boxes to be 2D only.
        Returns ((x0, y0), (x1, y1)) or batched version of this
    """

    if idx is None:
        with torch.no_grad():
            all_boxes = box_embedding_module.all_boxes()

            boxes = all_boxes.z.detach().tolist(), all_boxes.Z.detach().tolist(
            )
    else:
        with torch.no_grad():
            t = box_embedding_module(torch.tensor(idx, dtype=torch.long))

            boxes = t.z.detach().tolist(), t.Z.detach().tolist()

    return boxes


def get_box_for_idx(box_embedding: bm.TBoxTensor, idx: int
                    ) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    (x0, y0), (x1, y1) = get_box_from_module(box_embedding, idx)
    return x0 , x1, y0, y1


def get_nodes(name2idx: Mapping[str, int],
              box_embedding: Optional[bm.TBoxTensor] = None) -> List[Dict]:
    nodes = []
    for color, (name, idx) in zip(colors(), name2idx.items()):
        box_pos = BoxPosition(
            *get_box_for_idx(box_embedding, idx)) if box_embedding else None
        node = Node(name=name, idx=idx, box=box_pos, color=color)
        nodes.append(node)

    return nodes

def get_box_for_idx_adjusted(box_embedding: bm.TBoxTensor, idx: int
                    ) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    (x0, y0), (x1, y1) = get_box_from_module(box_embedding, idx)
    return x0 +2 , x1 + 2 , y0 - 2 , y1 - 2

def get_nodes_adjusted(name2idx: Mapping[str, int],
              box_embedding: Optional[bm.TBoxTensor] = None) -> List[Dict]:
    nodes = []
    for color, (name, idx) in zip(colors(), name2idx.items()):
        box_pos = BoxPosition(
            *get_box_for_idx_adjusted(box_embedding, idx)) if box_embedding else None
        node = Node(name=name, idx=idx, box=box_pos, color=color)
        nodes.append(node)

    return nodes


def get_box_tensor(boxtensor_type, box_position: BoxPosition) -> torch.Tensor:
    return boxtensor_type.from_zZ(box_position.z, box_position.Z)


axis_defaults = dict(
    range=[0, 0.1],
    showgrid=False,
    zeroline=False,
    showline=False,
    ticks="",
    showticklabels=False,
)

layout_defaults = dict(
    xaxis=deepcopy(axis_defaults),
    yaxis=dict(scaleanchor="x",
               **deepcopy(axis_defaults)),  # this makes it square
    autosize=True,
    width=600,
    height=600,
    #     margin=go.layout.Margin(
    #         l=50,
    #         r=50,
    #         b=50,
    #         t=50,
    #         pad=2
    #     ),
)


@dataclass
class NodeHistory:
    history: List[Tuple[Any, Node]] = None

    @classmethod
    def create(cls, name2idx: Mapping[str, int],
               models: List = None) -> "NodeHistory":
        history = []

        for step, model in enumerate(models):
            nodes = get_nodes(name2idx, model.h)
            history.append((step, nodes))

        return cls(history)

    def frames(self) -> List[Dict]:
        nodes_history = self.history

        if nodes_history is None:
            raise RuntimeError
        frames = [{
            "data": [node.go_box() for node in nodes],
            "name": str(i)
        } for i, nodes in nodes_history]

        return frames

    def slider_steps(self):
        if self.history is None:
            raise RuntimeError

        return [{
            "args": [
                [str(step)],
                {
                    "frame": {
                        "duration": 300,
                        "redraw": False
                    },
                    "mode": "immediate",
                    "transition": {
                        "duration": 300,
                        "easing": "linear"
                    },
                },
            ],
            "label":
            str(step),
            "method":
            "animate",
        } for step, node in self.history]

    def sliders(self):
        return [
            dict(
                active=0,
                yanchor="top",
                xanchor="left",
                currentvalue={
                    "font": {
                        "size": 20
                    },
                    "prefix": "Step:",
                    "visible": True,
                    "xanchor": "right"
                },
                transition={
                    "duration": 300,
                    "easing": "linear"
                },
                pad={
                    "b": 10,
                    "t": 50
                },
                len=0.9,
                x=0.1,
                y=0,
                steps=self.slider_steps())
        ]

    def buttons(self):
        return [
            {
                "args": [
                    None,
                    {
                        "frame": {
                            "duration": 500,
                            "redraw": False
                        },
                        "fromcurrent": True,
                        "transition": {
                            "duration": 300,
                            "easing": "linear"
                        },
                    },
                ],
                "label":
                "Play",
                "method":
                "animate",
            },
            {
                "args": [
                    (None),
                    {
                        "frame": {
                            "duration": 0,
                            "redraw": False
                        },
                        "mode": "immediate",
                        "transition": {
                            "duration": 0
                        },
                    },
                ],
                "label":
                "Pause",
                "method":
                "animate",
            },
        ]

    def update_menus(self):
        return [{
            "buttons": self.buttons(),
            "direction": "left",
            "pad": {
                "r": 10,
                "t": 87
            },
            "showactive": False,
            "type": "buttons",
            "x": 0.1,
            "xanchor": "right",
            "y": 0,
            "yanchor": "top",
        }]

    def layout(self):
        la = deepcopy(layout_defaults)
        la["sliders"] = self.sliders()
        la["updatemenus"] = self.update_menus()

        return la

    def fig_dict(self):
        frames = self.frames()
        layout = self.layout()

        return dict(data=frames[0]["data"], layout=layout, frames=frames)
