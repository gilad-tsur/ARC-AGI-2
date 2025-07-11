import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest
torch = pytest.importorskip("torch")
from utils import pad_grid, unpad_grid


def test_pad_and_unpad_round_trip():
    grid = [[1, 2, 3], [4, 5, 6]]
    padded, mask = pad_grid(grid, size=5)
    assert padded.shape == (5, 5)
    assert mask.sum() == 6

    unpadded = unpad_grid(padded, mask)
    assert torch.equal(unpadded, torch.tensor(grid))
