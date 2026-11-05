import pytest
import torch

from data_loader.data_loaders import NoisyPointInputs, NeSPReSODataset, check_input_noise

N = 4000


def _base():
    x = torch.zeros(N, 9)
    x[:, 6:] = torch.tensor([36.0, 27.0, 0.07])
    return NeSPReSODataset(x, torch.zeros(N, 32))


def _reads(ds):
    return torch.stack([ds[i][0] for i in range(len(ds))])


def test_noise_lands_on_sat_columns_with_requested_sigma():
    ds = NoisyPointInputs(_base(), {"sss": 0.0, "sst": 0.29, "ssh": 0.035}, seed=1)
    x = _reads(ds)
    assert torch.equal(x[:, :7], _reads(_base())[:, :7])
    assert x[:, 7].std().item() == pytest.approx(0.29, rel=0.05)
    assert x[:, 8].std().item() == pytest.approx(0.035, rel=0.05)
    assert x[:, 8].mean().item() == pytest.approx(0.07, abs=0.003)


def test_fresh_draw_each_read_and_base_untouched():
    base = _base()
    ds = NoisyPointInputs(base, {"sss": 0.0, "sst": 0.29, "ssh": 0.035}, seed=1)
    first, second = ds[0][0], ds[0][0]
    assert not torch.equal(first, second)
    assert base[0][0][8].item() == pytest.approx(0.07)


def test_seed_reproduces_sequence():
    a = _reads(NoisyPointInputs(_base(), {"sss": 0.1, "sst": 0.1, "ssh": 0.1}, seed=7))
    b = _reads(NoisyPointInputs(_base(), {"sss": 0.1, "sst": 0.1, "ssh": 0.1}, seed=7))
    assert torch.equal(a, b)


def test_zero_sigma_is_identity():
    ds = NoisyPointInputs(_base(), {"sss": 0.0, "sst": 0.0, "ssh": 0.0}, seed=3)
    assert torch.equal(_reads(ds), _reads(_base()))


@pytest.mark.parametrize("cache,use_err,pair", [
    ({"spatial_pad": 1}, False, False),
    ({"temporal_pad": 2}, False, False),
    ({"sat_patch_shape": [3, 3, 3, 3]}, False, False),
    ({}, True, False),
    ({}, False, True),
])
def test_refuses_non_point_layouts(cache, use_err, pair):
    with pytest.raises(ValueError):
        check_input_noise({"ssh": 0.035, "seed": 1}, cache, use_err, pair)


def test_spec_needs_seed_and_known_keys():
    with pytest.raises(ValueError):
        check_input_noise({"ssh": 0.035}, {}, False, False)
    with pytest.raises(ValueError):
        check_input_noise({"adt": 0.035, "seed": 1}, {}, False, False)
    assert check_input_noise({"ssh": 0.035, "seed": 1}, {}, False, False) == {"sss": 0.0, "sst": 0.0, "ssh": 0.035}
