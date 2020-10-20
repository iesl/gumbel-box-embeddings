import matplotlib.pyplot as plt
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
    ax.clear()
    palette = sns.color_palette(colors, len(boxes))
    for box, label, color in zip(boxes, labels, palette):
        min_corner = box[0, :]
        max_corner = box[1, :]
        width, height = max_corner - min_corner
        r = Rectangle(box[0, :], width, height, alpha=0.5, color=color, label=label)
        ax.add_patch(r)
    ax.legend(bbox_to_anchor=(1.2, 1), loc=9)


def draw_boxes(boxes, labels=None, colors="deep"):
    fig, ax = init_figure()
    add_boxes((boxes, labels), ax, colors)
    return fig



def animate_boxes(boxes_hist, labels=None, colors="deep", interval=200):
    if labels is None:
        send_boxes = [(b,None) for b in boxes_hist]
    elif len(labels) == len(boxes_hist):
        send_boxes = [(b,l) for b,l in zip(boxes_hist, labels)]
    else:
        send_boxes = [(b,labels) for b in boxes_hist]
    fig, ax = init_figure()
    ani = animation.FuncAnimation(
        fig,
        add_boxes,
        send_boxes,
        fargs=(ax, colors),
        interval=interval,
        repeat=False,
    )
    return ani