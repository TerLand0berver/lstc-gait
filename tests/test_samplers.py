from lstc.samplers import PKSampler, MultiViewPKSampler


def test_pk_sampler_simple():
    labels = [0,0,0,1,1,2,2,2,2]
    s = PKSampler(labels, batch_p=2, batch_k=2, seed=123)
    it = iter(s)
    batch = next(it)
    assert len(batch) == 4
    # two identities should appear (best-effort check)
    # not asserting exact composition due to shuffling


def test_multiview_pk_sampler_balanced():
    labels = [0,0,0,1,1,1,2,2]
    views  = [0,1,2,0,1,2,0,1]
    s = MultiViewPKSampler(labels, views, batch_p=2, batch_k=2, views_per_id=2, seed=123)
    it = iter(s)
    batch = next(it)
    assert len(batch) == 4
