import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib import animation, rc
import seaborn as sns

# this makes it so the javascript widget is used in Jupyter
# to display animations
rc('animation', html='jshtml')

def init_figure():
    fig, ax = plt.subplots(figsize=(10,10))
    ax.set_aspect("equal", "box")
    ax.axis([-0.1, 1.1, -0.1, 1.1])
    plt.close() # this removes the spurious plot
    return fig, ax


def add_boxes(boxes_and_labels, ax, colors="deep"):
    """
    colors should be a list of colors, eg.
            ["#9b59b6", "#3498db", "#95a5a6", "#e74c3c", "#34495e", "#2ecc71"]
        or a name of a seaborn color palette, eg. "hls"
    """
    # I had to make this a tuple due to FuncAnimation in animate_boxes
    boxes, labels = boxes_and_labels
    assert (
        boxes.shape[1] == 2
    ), "Can only draw boxes in dimension 2, current dimension is {}.".format(
        boxes.shape[1]
    )
    if labels is None:
        labels = range(boxes.shape[0])
    palette = sns.color_palette(colors, len(boxes))
    for box, label, color in zip(boxes, labels, palette):
        min_corner = box[0, :]
        max_corner = box[1, :]
        width, height = max_corner - min_corner
        r = Rectangle(box[0, :], width, height, alpha=0.5, color=color)
        ax.add_patch(r)
    # ax.legend(bbox_to_anchor=(1.2, 1), loc=9)


def add_weighted_boxes(boxes_and_weights, ax, colors="deep"):
    ax.clear()
    boxes, weights, labels = boxes_and_weights
    l = Line2D([weights, weights], [0,1])
    ax.add_line(l)
    for b in boxes:
        add_boxes((b, labels), ax, colors)


def draw_weighted_boxes(boxes, weight, labels=None, colors="deep"):
    fig, ax = init_figure()
    add_weighted_boxes((boxes, weight, labels), ax, colors)
    return fig



def animate_boxes(boxes_hist, weight_hist, labels=None, colors="deep", interval=200):
    send_boxes = [(b,w,None) for b, w in zip(boxes_hist, weight_hist)]
    fig, ax = init_figure()
    ani = animation.FuncAnimation(
        fig,
        add_weighted_boxes,
        send_boxes,
        fargs=(ax, colors),
        interval=interval,
        repeat=False,
    )
    return ani