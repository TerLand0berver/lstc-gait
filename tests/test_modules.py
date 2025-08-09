import torch
from lstc import LocalSpatioTemporalConv, LocalSpatioTemporalPooling, AsymmetricSpatioTemporalBlock, LSTCBackbone


def test_lstc_forward_shapes():
    x = torch.randn(2, 4, 8, 32, 22)
    m = LocalSpatioTemporalConv(4, 6, kernel_t=3, kernel_h=5, kernel_w=3, num_stripes=4)
    y = m(x)
    assert y.shape[0] == 2 and y.shape[1] == 6
    # temporal and spatial dims should be valid (>=1)
    assert y.shape[2] >= 1 and y.shape[3] >= 1 and y.shape[4] >= 1


def test_lstp_topk_and_soft():
    x = torch.randn(2, 8, 10, 16, 12)
    hard = LocalSpatioTemporalPooling(num_stripes=4, topk=2, soft=False)
    soft = LocalSpatioTemporalPooling(num_stripes=4, topk=2, soft=True, temperature=0.1)
    yh = hard(x)
    ys = soft(x)
    assert yh.shape == (2, 8 * 4)
    assert ys.shape == (2, 8 * 4)


def test_asymmetric_block_branches_toggle():
    x = torch.randn(1, 8, 6, 16, 12)
    # all branches
    b_all = AsymmetricSpatioTemporalBlock(8, 12, num_stripes=4, use_temporal=True, use_spatial=True, use_joint=True)
    y_all = b_all(x)
    # temporal only
    b_t = AsymmetricSpatioTemporalBlock(8, 12, num_stripes=4, use_temporal=True, use_spatial=False, use_joint=False)
    y_t = b_t(x)
    # joint only (dynamic)
    b_j = AsymmetricSpatioTemporalBlock(8, 12, num_stripes=4, use_temporal=False, use_spatial=False, use_joint=True, joint_type="dynamic")
    y_j = b_j(x)
    assert y_all.shape[1] == 12 and y_t.shape[1] == 12 and y_j.shape[1] == 12


def test_backbone_end_to_end():
    model = LSTCBackbone(in_channels=1, base_channels=8, num_stripes=4, embedding_dim=32)
    x = torch.randn(2, 1, 12, 32, 22)
    out = model(x)
    assert "feat_map" in out and "embedding" in out
    assert out["embedding"].shape == (2, 32)
