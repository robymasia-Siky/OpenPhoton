"""
OpenPhoton Model Exporter (MIT)
Esporta modelli PyTorch open-source (es. HTDemucs 4-Stem) nel formato ONNX
atteso da OpenPhoton: input [1, 2, 192000] -> output [1, 4, 2, 192000].

Uso:
  pip install torch demucs onnx onnxscript
  python tools/export_onnx_models.py --out openphoton_htdemucs.onnx
"""
import argparse
import torch
import torch.nn as nn

class DemucsOnnxWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        return self.model(wav)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="openphoton_htdemucs.onnx")
    args = parser.parse_args()

    from demucs.pretrained import get_model
    bag = get_model("htdemucs")
    core = bag.models[0].eval().cpu()

    wrapper = DemucsOnnxWrapper(core).eval()
    dummy = torch.zeros(1, 2, 48000 * 4, dtype=torch.float32)

    print("Esportazione HTDemucs in ONNX in corso...")
    torch.onnx.export(
        wrapper,
        dummy,
        args.out,
        input_names=["audio"],
        output_names=["stems"],
        opset_version=17,
        do_constant_folding=True,
    )
    print(f"Fatto! Carica '{args.out}' dal pulsante 'Load .onnx...' in OpenPhoton.")

if __name__ == "__main__":
    main()
