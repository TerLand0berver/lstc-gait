import torch
from lstc import LSTCBackbone


def main():
    # N, C, T, H, W
    x = torch.randn(2, 1, 16, 64, 44)
    model = LSTCBackbone(in_channels=1, base_channels=16, num_stripes=8, embedding_dim=128)
    out = model(x)
    print("feat_map:", tuple(out["feat_map"].shape))
    print("embedding:", tuple(out["embedding"].shape))


if __name__ == "__main__":
    main()
