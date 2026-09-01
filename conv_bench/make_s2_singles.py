#!/usr/bin/env python3
"""Build ONE model per stride-2 conv so image mode can be timed per conv.

The head models hold two convs each. In BUFFER mode MNN tags every conv kernel with its shape
(ConvBuf2D-ori-b1ci18hi96wi128co16...), so per_conv_us() can split the pair. In IMAGE mode both
convs are simply named "Convolution0" -- no shape, no index -- so the pair cannot be split by
name, and positional de-interleaving breaks the moment one conv takes the Winograd path (3
dispatches instead of 1). One conv per model removes the ambiguity entirely: conv_all_us() then
returns that conv's time in either mode.

Each conv is emitted twice: PReLU-fused (what deploys) and unfused (CPU ground truth -- the CPU
backend ignores leakyReluSlope, so a fused model on CPU returns the un-activated answer and any
cosine against it is a false mismatch).
"""
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "conv_bench"))
from block_fixture import build_onnx
from bench import convert

OUT = REPO / "conv_bench" / "conv_probe_bundle" / "models"
TMP = REPO / "conv_bench" / "_s2_tmp"; TMP.mkdir(exist_ok=True)

# the 8 stride-2 convs of the reduced shape set, with the kernel that won section 6
S2 = [
    ("s2_1_8_192x256",   dict(cin=1,  cout=8,  H=192, W=256, stride=2, pad=1, prelu=0), "conv_2d_c4h1w1"),
    ("s2_8_16_96x128",   dict(cin=8,  cout=16, H=96,  W=128, stride=2, pad=1, prelu=0), "conv_2d_c4h1w1"),
    ("s2_18_16_96x128",  dict(cin=18, cout=16, H=96,  W=128, stride=2, pad=1, prelu=1), "conv_2d_c8h2w1"),
    ("s2_16_32_48x64",   dict(cin=16, cout=32, H=48,  W=64,  stride=2, pad=1, prelu=1), "conv_2d_c8h1w1"),
    ("s2_34_32_48x64",   dict(cin=34, cout=32, H=48,  W=64,  stride=2, pad=1, prelu=1), "conv_2d_c8h2w1"),
    ("s2_32_48_24x32",   dict(cin=32, cout=48, H=24,  W=32,  stride=2, pad=1, prelu=1), "conv_2d_c4h1w1"),
    ("s2_64_64_24x32",   dict(cin=64, cout=64, H=24,  W=32,  stride=2, pad=1, prelu=1), "conv_2d_c8h1w1"),
    ("s2_64_96_12x16",   dict(cin=64, cout=96, H=12,  W=16,  stride=2, pad=1, prelu=1), "conv_2d_c8h1w1"),
]

manifest = []
for name, spec, best in S2:
    onnx_p = TMP / f"{name}.onnx"
    build_onnx(str(onnx_p), [spec])
    convert(str(onnx_p), str(OUT / f"{name}.mnn"), fp16=False, fuse_prelu=True)
    convert(str(onnx_p), str(OUT / f"{name}_unfused.mnn"), fp16=False, fuse_prelu=False)
    manifest.append(dict(key=name,
                         label=f"{spec['cin']}->{spec['cout']}@{spec['H']}x{spec['W']} s2",
                         model=f"{name}.mnn", unfused=f"{name}_unfused.mnn",
                         shape=[1, spec["cin"], spec["H"], spec["W"]],
                         prelu=spec["prelu"], best=best))
    print(f"   {name}: {manifest[-1]['label']}  best={best}  prelu={spec['prelu']}")

import json
(REPO / "conv_bench" / "s2_manifest.json").write_text(json.dumps(manifest, indent=2))
print(f"\n{len(manifest)} single-conv models written to {OUT}")
