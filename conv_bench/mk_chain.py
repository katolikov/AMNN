# build a self-chain of N identical C->C convs (optionally +prelu) for sustained-load timing
import numpy as np, onnx
from onnx import helper, TensorProto, numpy_helper
def make_chain(path, N, C, H, W, depth=6, act="none", k=3, stride=1):
    inits=[]; nodes=[]; prev="input"
    for j in range(depth):
        w=numpy_helper.from_array((np.random.randn(C,C,k,k).astype(np.float32)*0.05),f"w{j}")
        b=numpy_helper.from_array((np.random.randn(C).astype(np.float32)*0.01),f"b{j}")
        inits+=[w,b]
        co=f"c{j}"; nodes.append(helper.make_node("Conv",[prev,f"w{j}",f"b{j}"],[co],kernel_shape=[k,k],pads=[k//2]*4,strides=[stride,stride]))
        if act=="prelu":
            s=numpy_helper.from_array((0.05+0.4*np.arange(C)/max(C-1,1)).astype(np.float32),f"s{j}"); inits.append(s)
            out=f"p{j}" if j<depth-1 else "output"
            nodes.append(helper.make_node("PRelu",[co,f"s{j}"],[out])); prev=out
        else:
            prev=co if j<depth-1 else "output"
            if j==depth-1: nodes[-1].output[0]="output"
    x=helper.make_tensor_value_info("input",TensorProto.FLOAT,[N,C,H,W])
    y=helper.make_tensor_value_info("output",TensorProto.FLOAT,[N,C,H,W])
    m=helper.make_model(helper.make_graph(nodes,"ch",[x],[y],inits),opset_imports=[helper.make_opsetid("",13)]); m.ir_version=9
    onnx.save(m,path)
