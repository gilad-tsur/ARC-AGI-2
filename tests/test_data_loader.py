import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest
pytest.importorskip("torch")

from data_loader import ARCDataset


def test_dataset_item_shapes():
    ds = ARCDataset('.', split='training', mode='train')
    inp, out, in_mask, out_mask, task_id = ds[0]
    assert inp.shape == (30, 30)
    assert out.shape == (30, 30)
    assert in_mask.shape == (30, 30)
    assert out_mask.shape == (30, 30)
    assert isinstance(task_id, str)
