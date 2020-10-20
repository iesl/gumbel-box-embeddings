import plotly.offline as py
import plotly.graph_objs as go
import itertools
from .boxutils import *

py.init_notebook_mode(connected=True)

boundary = go.Scatter(
    x=[0, 0, 1, 1, 0],
    y=[0, 1, 1, 0, 0],
    mode="lines",
    showlegend=False,
    line=go.scatter.Line(color="#111", width=1),
    hoverinfo="skip",
)

axis_defaults = dict(
    range=[-0.1, 1.1],
    showgrid=False,
    zeroline=False,
    showline=False,
    ticks="",
    showticklabels=False,
)

layout_defaults = dict(
    xaxis=axis_defaults,
    yaxis=dict(scaleanchor="x", **axis_defaults),  # this makes it square
    autosize=False,
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

# These are Plotly's defaults, but I need more control of them.
color_list = [
    "#1f77b4",  # muted blue
    "#ff7f0e",  # safety orange
    "#2ca02c",  # cooked asparagus green
    "#d62728",  # brick red
    "#9467bd",  # muted purple
    "#8c564b",  # chestnut brown
    "#e377c2",  # raspberry yogurt pink
    "#7f7f7f",  # middle gray
    "#bcbd22",  # curry yellow-green
    "#17becf",  # blue-teal
]

# These are Seaborn's paired colors, for reference:
sns_paired = [
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

# Overriding for a specific plot
# Uncomment this and the colors will be chosen from this list
# TODO: Make this an argument of plotting functions? A property of the model?
# color_list = [
#     color_list[4],
#     sns_paired[0],
#     sns_paired[1],
#     sns_paired[2],
#     sns_paired[3],
#     sns_paired[4],
#     sns_paired[5]
# ]


def create_boxes(boxes, labels=None):
    assert (
        boxes.shape[1] == 2
    ), "Can only draw boxes in dimension 2, current dimension is {}.".format(
        boxes.shape[1]
    )
    if labels is None:
        labels = range(boxes.shape[0])
    boxes_data = []
    colors = itertools.cycle(color_list)
    for box, label in zip(boxes, labels):
        color = colors.__next__()
        z = box[0]
        Z = box[1]
        boxes_data += [
            go.Scatter(
                x=[z[0], z[0], Z[0], Z[0], z[0]],
                y=[z[1], Z[1], Z[1], z[1], z[1]],
                mode="lines",
                fill="toself",
                name=label,
                marker={"color": color},
            ),
            go.Scatter(
                x=[z[0]],
                y=[z[1]],
                mode="markers",
                marker={"color": color},
                name="z",
                showlegend=False,
            ),
            go.Scatter(
                x=[Z[0]],
                y=[Z[1]],
                mode="markers",
                marker={"color": color},
                name="Z",
                showlegend=False,
            ),
        ]
    return boxes_data


# This next version uses the shapes library for plotly, but I found animations to be slow with this so I'm not going to use it.
# def create_boxes(boxes, colors="deep"):
#     '''
#     colors should be a list of colors, eg.
#             ["#9b59b6", "#3498db", "#95a5a6", "#e74c3c", "#34495e", "#2ecc71"]
#         or a name of a seaborn color palette, eg. "hls"
#     '''
#     assert boxes.shape[1] == 2, "Can only draw boxes in dimension 2, current dimension is {}.".format(boxes.shape[1])
#     palette = sns.color_palette(colors,len(boxes))
#     shapes = []
#     for box, color in zip(boxes, palette):
#         shapes.append({
#             'type': 'rect',
#             'x0': box[0,0],
#             'y0': box[1,0],
#             'x1': box[0,1],
#             'y1': box[1,1],
#             'line': {
#                 'color': 'rgba({},{},{},1)'.format(*color),
#                 'width': 2,
#             },
#             'fillcolor': 'rgba({},{},{},0.5)'.format(*color),
#         })
#     return shapes


def draw_boxes(boxes, labels=None):
    py.iplot(
        {"data": [boundary] + create_boxes(boxes, labels), "layout": layout_defaults}
    )


def animate_boxes(boxes_hist, hist=None, labels=None):
    if hist is None:
        hist = range(len(boxes_hist))
    if labels is None:
        frames = [
            {"data": [boundary] + create_boxes(b), "name": str(i)}
            for i, b in zip(hist, boxes_hist)
        ]
    elif len(labels) < len(boxes_hist):
        # labels are to be used for all of the boxes
        frames = [
            {"data": [boundary] + create_boxes(b, labels), "name": str(i)}
            for i, b in zip(hist, boxes_hist)
        ]
    else:
        frames = [
            {"data": [boundary] + create_boxes(b, l), "name": str(i)}
            for i, b, l in zip(hist, boxes_hist, labels)
        ]

    steps = [
        {
            "args": [
                (str(i)),
                {
                    "frame": {"duration": 50, "redraw": False},
                    "mode": "immediate",
                    "transition": {"duration": 50, "easing": "linear"},
                },
            ],
            "label": str(i),
            "method": "animate",
        }
        for i in hist
    ]

    fig = dict(data=frames[0]["data"], layout=layout_defaults, frames=frames)
    fig["layout"]["sliders"] = [
        dict(
            active=0,
            yanchor="top",
            xanchor="left",
            transition={"duration": 50, "easing": "linear"},
            pad={"b": 10, "t": 50},
            len=0.9,
            x=0.1,
            y=0,
            steps=steps,
        )
    ]
    fig["layout"]["updatemenus"] = [
        {
            "buttons": [
                {
                    "args": [
                        None,
                        {
                            "frame": {"duration": 50, "redraw": False},
                            "fromcurrent": True,
                            "transition": {"duration": 50, "easing": "linear"},
                        },
                    ],
                    "label": "Play",
                    "method": "animate",
                },
                {
                    "args": [
                        (None),
                        {
                            "frame": {"duration": 0, "redraw": False},
                            "mode": "immediate",
                            "transition": {"duration": 0},
                        },
                    ],
                    "label": "Pause",
                    "method": "animate",
                },
            ],
            "direction": "left",
            "pad": {"r": 10, "t": 87},
            "showactive": False,
            "type": "buttons",
            "x": 0.1,
            "xanchor": "right",
            "y": 0,
            "yanchor": "top",
        }
    ]

    py.iplot(fig)


def labels(boxes, probs=None, names=None):
    if type(boxes) == list or len(boxes.shape) == 4:
        # we have a list of lists of boxes, such as boxes_hist
        return [labels(bxs, probs, names) for bxs in boxes]
    elif probs is None and names is None:
        return [f"{v:.2f}" for v in volume(boxes)]
    elif probs is not None and names is None:
        return [f"{v:.2f} -> {p:.2f}" for v, p in zip(volume(boxes), probs)]
    elif probs is None and names is not None:
        return [f"[{v:.2f}]  {n}" for n, v in zip(names, volume(boxes))]
    else:
        return [
            f"[{v:.2f} -> {p:.2f}]  {n}" for n, v, p in zip(names, volume(boxes), probs)
        ]

