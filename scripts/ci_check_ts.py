import torch
from lstc import LSTCBackbone


def main() -> None:
    ckpt = torch.load("runs/lstc_real/best.pt", map_location="cpu")
    embed_dim = ckpt.get("args", {}).get("embedding_dim", 256)
    num_stripes = ckpt.get("args", {}).get("num_stripes", 8)
    base_channels = ckpt.get("args", {}).get("base_channels", 16)

    model = LSTCBackbone(
        in_channels=1,
        base_channels=base_channels,
        num_stripes=num_stripes,
        embedding_dim=embed_dim,
    )
    model.load_state_dict(ckpt["model"])  # type: ignore[index]
    model.eval()

    ts = torch.jit.load("runs/export/model.ts")
    ts.eval()

    x = torch.randn(1, 1, 8, 64, 44)
    with torch.no_grad():
        e_ref = model(x)["embedding"]
        e_ts = ts(x)

    def l2n(t: torch.Tensor) -> torch.Tensor:
        return t / (t.norm(dim=1, keepdim=True) + 1e-12)

    e_ref_n = l2n(e_ref)
    e_ts_n = l2n(e_ts)
    cos = torch.sum(e_ref_n * e_ts_n, dim=1).mean().item()
    print({"cosine": cos})
    assert cos > 0.999, f"Cosine similarity too low: {cos}"


if __name__ == "__main__":
    main()
