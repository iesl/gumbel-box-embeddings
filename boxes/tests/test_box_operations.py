import pytest
from hypothesis import given, example
from hypothesis import strategies as st
from boxes.box_operations import *
from boxes.modules import *
import torch
import copy

test_cases = dict()

# Boxes(1,2,1)
A = torch.tensor([[0.1], [0.4]])
B = torch.tensor([[0.3], [0.8]])
unitboxes = Boxes(1, 2, 1)
unitboxes.boxes[0,0] = A
unitboxes.boxes[0,1] = B
small_boxes = torch.tensor([[1,0]], dtype=torch.uint8)
test_case = dict(
    A = A,
    B = B,
    unitboxes = unitboxes,
    boxparam = unitboxes,
    boxes = unitboxes.boxes,
    ids = torch.tensor([[0,1]]),
    int_boxes =  torch.tensor([[0.3], [0.4]]),
    min_vol = 0.4,
    vol_func = clamp_volume,
    correct_smallest_containing_box = torch.tensor([[[0.1],[0.8]]]),
    small_boxes = small_boxes,
    boxes_ind = small_boxes,
    volumes = torch.tensor([[0.3, 0.5]]),
    disjoint_boxes = torch.tensor([[0]], dtype=torch.uint8),
    overlapping_boxes = torch.tensor([[1]], dtype=torch.uint8),
    containing_boxes = torch.tensor([[0]], dtype=torch.uint8),
)
test_cases["Boxes(1,2,1)"] = test_case


# Boxes(1,2,2)
A = torch.tensor([[0.1, 0.3], [0.4, 0.7]])
B = torch.tensor([[0.3, 0.2], [1.0, 0.9]])
unitboxes = Boxes(1, 2, 2)
unitboxes.boxes[0,0] = A
unitboxes.boxes[0,1] = B
small_boxes = torch.tensor([[0,0]], dtype=torch.uint8)
test_case = dict(
    A = A,
    B = B,
    unitboxes = unitboxes,
    boxparam = unitboxes,
    boxes = unitboxes.boxes,
    ids = torch.tensor([[0,1]]),
    int_boxes = torch.tensor([[[0.3, 0.3], [0.4, 0.7]]]),
    min_vol = 0.1,
    vol_func = clamp_volume,
    correct_smallest_containing_box = torch.tensor([[[0.1, 0.2], [1.0, 0.9]]]),
    small_boxes = small_boxes,
    boxes_ind = small_boxes,
    volumes = torch.tensor([[0.12, 0.49]]),
    disjoint_boxes = torch.tensor([[0]], dtype=torch.uint8),
    overlapping_boxes = torch.tensor([[1]], dtype=torch.uint8),
    containing_boxes = torch.tensor([[0]], dtype=torch.uint8),
)
test_cases["Boxes(1,2,2)"] = test_case


# Boxes(1,2,2) disjoint in one dimension
A = torch.tensor([[0.1, 0.3], [0.2, 0.7]])
B = torch.tensor([[0.3, 0.2], [1.0, 0.9]])
unitboxes = Boxes(1, 2, 2)
unitboxes.boxes[0,0] = A
unitboxes.boxes[0,1] = B
small_boxes = torch.tensor([[1,0]], dtype=torch.uint8)
test_case = dict(
    A = A,
    B = B,
    unitboxes = unitboxes,
    boxparam = unitboxes,
    boxes = unitboxes.boxes,
    ids = torch.tensor([[0,1]]),
    int_boxes = torch.tensor([[0.3, 0.3], [0.2, 0.7]]),
    min_vol = 0.1,
    vol_func = clamp_volume,
    correct_smallest_containing_box = torch.tensor([[[0.1, 0.2], [1.0, 0.9]]]),
    small_boxes = small_boxes,
    boxes_ind = small_boxes,
    volumes = torch.tensor([[0.04, 0.49]]),
    disjoint_boxes = torch.tensor([[1]], dtype=torch.uint8),
    overlapping_boxes = torch.tensor([[0]], dtype=torch.uint8),
    containing_boxes = torch.tensor([[0]], dtype=torch.uint8),
)
test_cases["Boxes(1,2,2) disjoint"] = test_case


# Boxes(1,2,2) contained
A = torch.tensor([[0.1, 0.3], [0.2, 0.7]])
B = torch.tensor([[0.1, 0.2], [1.0, 0.9]])
unitboxes = Boxes(1, 2, 2)
unitboxes.boxes[0,0] = A
unitboxes.boxes[0,1] = B
small_boxes = torch.tensor([[1,0]], dtype=torch.uint8)
test_case = dict(
    A = A,
    B = B,
    unitboxes = unitboxes,
    boxparam = unitboxes,
    boxes = unitboxes.boxes,
    ids = torch.tensor([[0,1]]),
    int_boxes = torch.tensor([[0.1, 0.3], [0.2, 0.7]]),
    min_vol = 0.1,
    vol_func = clamp_volume,
    correct_smallest_containing_box = torch.tensor([[[0.1, 0.2], [1.0, 0.9]]]),
    small_boxes = small_boxes,
    boxes_ind = small_boxes,
    volumes = torch.tensor([[0.04, 0.63]]),
    disjoint_boxes = torch.tensor([[0]], dtype=torch.uint8),
    overlapping_boxes = torch.tensor([[1]], dtype=torch.uint8),
    containing_boxes = torch.tensor([[1]], dtype=torch.uint8),
)
test_cases["Boxes(1,2,2) contained"] = test_case


# Boxes(1,3,2) Intersection Possibilities
unitboxes = Boxes(1, 4, 1)
unitboxes.boxes.data = torch.tensor([
    [[0.1], [0.4]], #0
    [[0.5], [0.8]], #1
    [[0.9], [1.0]], #2
    [[0.2], [0.6]], #3
    [[0.3], [0.4]], #4
    [[0.6], [0.7]], #5
    [[0.7], [0.9]], #6
    ])[None]
test_case = dict(
    unitboxes = unitboxes,
    boxparam = unitboxes,
    ids = torch.tensor([[0,1], [2,6], [1,2], [0,3], [1,3], [1,6], [4,0], [5,1], [4,3]]),
    disjoint_boxes = torch.tensor([[1,1,1,0,0,0,0,0,0]], dtype=torch.uint8),
    containing_boxes = torch.tensor([[0,0,0,0,0,0,1,1,1]], dtype=torch.uint8),
    probs = torch.tensor([0,0.5,1,0,0.2,1,0,0.7,1]),
    needing_push = torch.tensor([[0,0,0,0,0,0,1,1,0]], dtype=torch.uint8),
    needing_pull = torch.tensor([[0,1,1,0,0,0,0,0,0]], dtype=torch.uint8),
)
test_cases["Boxes(1,3,2) Intersection Possibilities"] = test_case

def skip_unless_parameters_exist(**kwargs):
    missing_vars = []
    for name, value in kwargs.items():
        if value is None:
            missing_vars.append(name)
    if len(missing_vars) > 0:
        reason = "Not enough info to complete this test case.\n"
        reason += "To check this case, set the following variables:\n"
        reason += ", ".join(missing_vars)
        pytest.skip(reason)
        return False
    else:
        return True


def params_from(test_cases, *var_names):
    params = list()
    for (name, t) in test_cases.items():
        if set(var_names).issubset(t.keys()):
            params.append(pytest.param(*(t[v] for v in var_names), id=name))
    args = ", ".join(var_names)
    return args, params


@pytest.mark.parametrize(*params_from(test_cases, "boxparam", "A", "B"))
def test_single_ids(boxparam, A, B):
    assert (A == boxparam(torch.tensor([0]))).all()
    assert (B == boxparam(torch.tensor([1]))).all()


@pytest.mark.parametrize(*params_from(test_cases, "boxparam", "A", "B"))
def test_pair_ids(boxparam, A, B):
    out = boxparam(torch.tensor([[0,1]]))
    out_A = out[:,:,0]
    out_B = out[:,:,1]
    assert (A == out_A).all()
    assert (B == out_B).all()


@pytest.mark.parametrize(*params_from(test_cases, "boxparam", "vol_func", "volumes"))
def test_volume(boxparam, vol_func, volumes):
    out = vol_func(boxparam())
    assert (torch.isclose(out, volumes)).all()


@pytest.mark.parametrize(*params_from(test_cases, "boxparam", "volumes"))
def test_log_clamp_volume(boxparam, volumes):
    out = log_clamp_volume(boxparam())
    assert (torch.isclose(out, torch.log(volumes))).all()


@pytest.mark.parametrize(*params_from(test_cases, "boxes", "correct_smallest_containing_box"))
def test_smallest_containing_box(boxes, correct_smallest_containing_box):
    out = smallest_containing_box(boxes)
    assert (correct_smallest_containing_box == out).all()


@pytest.mark.parametrize(*params_from(test_cases, "boxparam", "int_boxes", "ids"))
def test_intersection_boxes(boxparam, int_boxes, ids):
    out_boxes = intersection(boxparam(ids[:,0]), boxparam(ids[:,1]))
    assert (int_boxes == out_boxes).all()


@pytest.mark.parametrize(*params_from(test_cases, "boxparam", "vol_func", "min_vol", "small_boxes"))
def test_detect_small_boxes(boxparam, vol_func, min_vol, small_boxes):
    out = detect_small_boxes(boxparam(), vol_func, min_vol)
    assert (out == small_boxes).all()


@pytest.mark.parametrize(*params_from(test_cases, "boxparam", "boxes_ind", "min_vol", "vol_func"))
def test_replace_Z_by_cube(boxparam, boxes_ind, min_vol, vol_func):
    out = replace_Z_by_cube(boxparam(), boxes_ind, min_vol)
    new_boxes = torch.stack((boxparam()[:,:,0][boxes_ind], out), dim=2)
    assert torch.isclose(vol_func(new_boxes), torch.tensor(min_vol)).all()


@pytest.mark.parametrize(*params_from(test_cases, "boxparam", "boxes_ind", "min_vol", "vol_func"))
def test_replace_Z_by_cube_(boxparam, boxes_ind, min_vol, vol_func):
    boxparam = copy.deepcopy(boxparam) # This is *not* the best way to do this, should use a fixture
    replace_Z_by_cube_(boxparam.boxes, boxes_ind, min_vol)
    assert (vol_func(boxparam()) >= min_vol - 1e-8).all()
    assert detect_small_boxes(boxparam(), vol_func, min_vol - 1e-8).sum() == 0


@pytest.mark.parametrize(*params_from(test_cases, "boxparam", "ids", "disjoint_boxes"))
def test_disjoint_boxes_mask(boxparam, ids, disjoint_boxes):
    out = disjoint_boxes_mask(boxparam(ids[:,0]), boxparam(ids[:,1]))
    assert (out == disjoint_boxes).all()
    assert out.shape == disjoint_boxes.shape


@pytest.mark.parametrize(*params_from(test_cases, "boxparam", "ids", "overlapping_boxes"))
def test_overlapping_boxes_mask(boxparam, ids, overlapping_boxes):
    out = overlapping_boxes_mask(boxparam(ids[:,0]), boxparam(ids[:,1]))
    assert (out == overlapping_boxes).all()
    assert out.shape == overlapping_boxes.shape


@pytest.mark.parametrize(*params_from(test_cases, "boxparam", "ids", "containing_boxes"))
def test_containing_boxes_mask(boxparam, ids, containing_boxes):
    out = containing_boxes_mask(boxparam(ids[:,0]), boxparam(ids[:,1]))
    assert (out == containing_boxes).all()
    assert out.shape == containing_boxes.shape


@pytest.mark.parametrize(*params_from(test_cases, "boxparam", "ids", "probs", "needing_push"))
def needing_push_mask(boxparam, ids, probs, needing_push):
    out = needing_push_mask(boxparam(ids[:,0]), boxparam(ids[:,1]), probs)
    assert (out == needing_push).all()
    assert out.shape == needing_push.shape


@pytest.mark.parametrize(*params_from(test_cases, "boxparam", "ids", "probs", "needing_pull"))
def test_needing_pull_mask(boxparam, ids, probs, needing_pull):
    out = needing_pull_mask(boxparam(ids[:,0]), boxparam(ids[:,1]), probs)
    assert (out == needing_pull).all()
    assert out.shape == needing_pull.shape


@given(num_models=st.integers(1,10), num_boxes=st.integers(1,1000), dim=st.integers(1,100), min_vol=st.floats(1e-6,1e-1))
def test_unitboxes_replace_Z_by_cube(num_models, num_boxes, dim, min_vol):
    unitboxes = Boxes(num_models, num_boxes, dim)
    small_boxes = detect_small_boxes(unitboxes.boxes, clamp_volume, min_vol)
    out = replace_Z_by_cube(unitboxes.boxes, small_boxes, min_vol)
    new_boxes = torch.stack((unitboxes()[:,:,0][small_boxes], out), dim=2)
    assert (clamp_volume(new_boxes) >= min_vol-1e-6).all()


@given(num_models=st.integers(1,10), num_boxes=st.integers(1,1000), dim=st.integers(1,100), min_vol=st.floats(1e-6,1e-1))
def test_unitboxes_replace_Z_by_cube_inplace(num_models, num_boxes, dim, min_vol):
    unitboxes = Boxes(num_models, num_boxes, dim)
    small_boxes = detect_small_boxes(unitboxes.boxes, clamp_volume, min_vol)
    replace_Z_by_cube_(unitboxes.boxes, small_boxes, min_vol)
    assert (detect_small_boxes(unitboxes.boxes, clamp_volume, min_vol - 1e-6) == 0).all()
