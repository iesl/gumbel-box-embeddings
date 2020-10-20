import pytest
from .modules import *
from collections import OrderedDict

@pytest.fixture(
    params = [Boxes, DeltaBoxes]
)


def box_3_4_5(request):
    Boxes = request.param
    return Boxes(3,4,5)


def test_Boxes_forward_no_args(box_3_4_5):
    assert box_3_4_5().shape == (3,4,2,5)


def test_Boxes_forward_list(box_3_4_5):
    unitbox_out = box_3_4_5([1,3])
    assert unitbox_out.shape == (3,2,2,5)


def test_UnitBoxes_to_SigmoidBoxes():
    unit_boxes = Boxes(2, 1982, 50)
    unit_boxes_orig = unit_boxes.boxes.clone()
    sigmoid_boxes = SigmoidBoxes(2, 1982, 50)
    sigmoid_boxes._from_UnitBoxes(unit_boxes)
    del unit_boxes
    sigmoid_boxes_out = sigmoid_boxes()
    assert torch.allclose(unit_boxes_orig, sigmoid_boxes_out)


def test_UnitBoxes_zero_init_min_vol():
    unit_boxes = Boxes(100, 10000, 50, torch.finfo(torch.float32).tiny)
    assert (torch.min(clamp_volume(unit_boxes())) > 0)


def test_UnitBoxes_init_in_unit_cube():
    unit_boxes = Boxes(100, 10000, 50)
    boxes = unit_boxes()
    assert (boxes.clamp(0,1) == boxes).all()
