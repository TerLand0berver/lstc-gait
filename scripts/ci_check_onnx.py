import numpy as np
import onnxruntime as ort
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

    x = torch.randn(1, 1, 8, 64, 44)
    with torch.no_grad():
        e_ref = model(x)["embedding"].numpy()
    e_ref = e_ref / (np.linalg.norm(e_ref, axis=1, keepdims=True) + 1e-12)

    sess = ort.InferenceSession("runs/export/model.onnx", providers=["CPUExecutionProvider"])  # type: ignore[no-untyped-call]
    e_onnx = sess.run(["embedding"], {sess.get_inputs()[0].name: x.numpy()})[0]
    e_onnx = e_onnx / (np.linalg.norm(e_onnx, axis=1, keepdims=True) + 1e-12)

    cos = float((e_ref * e_onnx).sum(axis=1).mean())
    print({"onnx_cosine": cos})
    assert cos > 0.995, f"ONNX cosine too low: {cos}"


if __name__ == "__main__":
    main()
