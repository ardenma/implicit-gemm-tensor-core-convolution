import math
import time
import argparse
from dataclasses import dataclass

import torch
import numpy as np
from torch import memory_format, nn
from torch.autograd import Function
from torch.utils.cpp_extension import load
from tqdm import tqdm

@dataclass
class TestConfig:
    num_epochs: int
    k: int
    c: int
    n: int
    h: int
    w:int

def geo_mean(iterable):
    a = np.array(iterable)
    return a.prod()**(1.0/len(a))

def get_conv_version(version: int):
    # Parse version
    if version == 1:
        return ConvTCFunctionV1
    elif version == 2:
        return ConvTCFunctionV2
    elif version == 3:
        return ConvTCFunctionV3
    elif version == -1:
        return ConvTCFunctionUpperBound
    else:
        raise Exception(f"Unknown version {version}")

sources = [
    "src/conv.cpp",
    "src/conv_3x3_same_padding.cu",
    "src/conv_3x3_same_padding_tc_v1.cu",
    "src/conv_3x3_same_padding_tc_v2.cu",
    "src/conv_3x3_same_padding_tc_v3.cu",
    "src/conv_3x3_same_padding_tc_upper_bound.cu",
]

include = []

pytorch_extension = load(
    name="pytorch_extension",
    sources=sources,
    verbose=True,
    extra_include_paths=include
)

torch.manual_seed(42)

class ConvFunction(Function):
    @staticmethod
    def forward(ctx, input, weights, bias):
        outputs = pytorch_extension.conv_forward(input, weights, bias)
        ctx.save_for_backward([input, weights, bias])
        return outputs[0]

    @staticmethod
    def backward(ctx, grad):
        input, weights, bias = ctx.saved_variables
        d_input = torch.rand(input.size)
        d_weights = torch.rand(weights.size)
        d_bias = torch.rand(bias.size)
        return d_input, d_weights, d_bias

class ConvTCFunctionV1(Function):
    @staticmethod
    def forward(ctx, input, weights):
        outputs = pytorch_extension.conv_tc_forward_v1(input, weights)
        ctx.save_for_backward([input, weights])
        return outputs[0]

    @staticmethod
    def backward(ctx, grad):
        input, weights, bias = ctx.saved_variables
        d_input = torch.rand(input.size)
        d_weights = torch.rand(weights.size)
        d_bias = torch.rand(bias.size)
        return d_input, d_weights, d_bias

class ConvTCFunctionV2(Function):
    @staticmethod
    def forward(ctx, input, weights):
        outputs = pytorch_extension.conv_tc_forward_v2(input, weights)
        ctx.save_for_backward([input, weights])
        return outputs[0]

    @staticmethod
    def backward(ctx, grad):
        input, weights, bias = ctx.saved_variables
        d_input = torch.rand(input.size)
        d_weights = torch.rand(weights.size)
        d_bias = torch.rand(bias.size)
        return d_input, d_weights, d_bias

class ConvTCFunctionV3(Function):
    @staticmethod
    def forward(ctx, input, weights):
        outputs = pytorch_extension.conv_tc_forward_v3(input, weights)
        ctx.save_for_backward([input, weights])
        return outputs[0]

    @staticmethod
    def backward(ctx, grad):
        input, weights, bias = ctx.saved_variables
        d_input = torch.rand(input.size)
        d_weights = torch.rand(weights.size)
        d_bias = torch.rand(bias.size)
        return d_input, d_weights, d_bias

class ConvTCFunctionUpperBound(Function):
    @staticmethod
    def forward(ctx, input, weights):
        outputs = pytorch_extension.conv_tc_forward_upper_bound(input, weights)
        ctx.save_for_backward([input, weights])
        return outputs[0]

    @staticmethod
    def backward(ctx, grad):
        input, weights, bias = ctx.saved_variables
        d_input = torch.rand(input.size)
        d_weights = torch.rand(weights.size)
        d_bias = torch.rand(bias.size)
        return d_input, d_weights, d_bias

def nchw_to_nhwc(x):
    return x.permute(0,2,3,1).contiguous()

def nhwc_to_nchw(x):
    return x.permute(0,3,1,2).contiguous()

def test_channel_conversion():
    x = torch.rand(10, 64, 128, 128, dtype=torch.float16)
    x_prime = nchw_to_nhwc(x)
    x_dprime = nhwc_to_nchw(x_prime)
    assert torch.allclose(x, x_dprime)

def test_conv_acc():
    assert torch.cuda.is_available()
    cuda_device = torch.device("cuda")  # device object representing GPU

    # Create ref conv2d implementation and move to gpu
    conv_ref = nn.Sequential(
        nn.Conv2d(3, 1, 3, stride=1, padding=1, bias=True, padding_mode="zeros")
    )
    conv_ref.to(dtype=torch.float16, device=cuda_device)

    # Grab the weights and bias
    weights = conv_ref.state_dict()['0.weight']
    bias = conv_ref.state_dict()['0.bias']
    x = torch.rand(1, 3, 64, 64, dtype=torch.float16, device=cuda_device)
    
    # Forward pass reference
    y_ref = nchw_to_nhwc(conv_ref(x))

    # Forward pass custom
    x = nchw_to_nhwc(x)
    y = ConvFunction.apply(x, weights, bias)

    assert torch.allclose(y, y_ref, atol=1e-3, rtol=1e-2)

def test_conv_perf(config: TestConfig):
    assert torch.cuda.is_available()
    cuda_device = torch.device("cuda")  # device object representing GPU

    # Create ref conv2d implementation and move to gpu
    conv_ref = nn.Sequential(
        nn.Conv2d(config.c, config.k, 3, stride=1, padding=1, bias=True, padding_mode="zeros")
    )
    conv_ref.to(dtype=torch.float16, device=cuda_device)

    # Grab the weights and bias
    weights = conv_ref.state_dict()['0.weight']
    bias = conv_ref.state_dict()['0.bias']
    x = torch.rand(config.n, config.c, config.h, config.w, dtype=torch.float16, device=cuda_device)
    
    # Forward pass reference
    torch.cuda.synchronize()
    start = time.time()
    for _ in tqdm(range(config.num_epochs)):
        y_ref = nchw_to_nhwc(conv_ref(x))
    torch.cuda.synchronize()
    end = time.time()
    ref_time = end - start

    # Forward pass custom
    x = nchw_to_nhwc(x)
    torch.cuda.synchronize()
    start = time.time()
    for _ in tqdm(range(config.num_epochs)):
        y = ConvFunction.apply(x, weights, bias)
    torch.cuda.synchronize()
    end = time.time()
    custom_time = end - start

    print('Pytorch Convolution Forward: {:.3f} s | Custom Convolution Forward {:.3f} s'.format(ref_time, custom_time))


def test_conv_tc_acc(version: int):
    assert torch.cuda.is_available()
    cuda_device = torch.device("cuda")  # device object representing GPU

    # Parse version
    ConvTCFunction = get_conv_version(version)

    # Create ref conv2d implementation and move to gpu
    conv_ref = nn.Sequential(
        nn.Conv2d(16, 16, 3, stride=1, padding=1, bias=False, padding_mode="zeros")
    )
    conv_ref.to(dtype=torch.float16, device=cuda_device)

    # Grab the weights and bias
    weights = conv_ref.state_dict()['0.weight']
    x = torch.rand(1, 16, 32, 1, dtype=torch.float16, device=cuda_device)

    # Forward pass reference
    y_ref = nchw_to_nhwc(conv_ref(x))

    # Forward pass custom
    x = nchw_to_nhwc(x)
    y = ConvTCFunction.apply(x, weights)

    assert torch.allclose(y, y_ref, atol=1e-3, rtol=1e-2)

def warmup_conv_tc_perf(config: TestConfig, version: int):
    cuda_device = torch.device("cuda")  # device object representing GPU

    # Parse version
    ConvTCFunction = get_conv_version(version)

    # Create ref conv2d implementation and move to gpu
    conv_ref = nn.Sequential(
        nn.Conv2d(config.c, config.k, 3, stride=1, padding=1, bias=False, padding_mode="zeros")
    )
    conv_ref.to(dtype=torch.float16, device=cuda_device)

    # Grab the weights and bias
    weights = conv_ref.state_dict()['0.weight']
    x = torch.rand(config.n, config.c, config.h, config.w, dtype=torch.float16, device=cuda_device)

    # Warmup
    for _ in range(config.num_epochs):
        y_ref = conv_ref(x)
    for _ in range(config.num_epochs):
        y = ConvTCFunction.apply(x, weights)

def test_conv_tc_perf(config: TestConfig, version: int):
    assert torch.cuda.is_available()
    cuda_device = torch.device("cuda")  # device object representing GPU

    # Parse version
    ConvTCFunction = get_conv_version(version)

    # Create ref conv2d implementation and move to gpu
    conv_ref = nn.Sequential(
        nn.Conv2d(config.c, config.k, 3, stride=1, padding=1, bias=False, padding_mode="zeros")
    )
    conv_ref.to(dtype=torch.float16, device=cuda_device)

    # Grab the weights and bias
    weights = conv_ref.state_dict()['0.weight']
    x = torch.rand(config.n, config.c, config.h, config.w, dtype=torch.float16, device=cuda_device)

    # Forward pass reference
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(config.num_epochs):
        y_ref = conv_ref(x)
    torch.cuda.synchronize()
    end = time.time()
    ref_time = end - start

    # Forward pass custom
    x = nchw_to_nhwc(x)
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(config.num_epochs):
        y = ConvTCFunction.apply(x, weights)
    torch.cuda.synchronize()
    end = time.time()
    custom_time = end - start

    print(f'Problem Size: (n={config.n:4d}, c={config.c:3d}, h={config.h:3d}, w={config.w:3d}, k={config.k:3d}) | PyTorch Conv Fprop: {ref_time:.3f} s | Custom Conv Fprop {custom_time:.3f} s | Relative speedup: {ref_time/custom_time:.3f}x')
    return ref_time / custom_time

def main(args):
    torch.cuda.empty_cache()
    conv_version = 3

    if args.perf:
        num_epochs = 100

        # (1024, 1, 64, 64)
        input1_configs = [
            TestConfig(num_epochs=num_epochs, k=64, c=64, n=1024, h=16, w=16),
            TestConfig(num_epochs=num_epochs, k=128, c=64, n=1024, h=16, w=16),
            TestConfig(num_epochs=num_epochs, k=128, c=128, n=1024, h=8, w=8),
            TestConfig(num_epochs=num_epochs, k=256, c=128, n=1024, h=8, w=8),
            TestConfig(num_epochs=num_epochs, k=256, c=256, n=1024, h=4, w=4),
            TestConfig(num_epochs=num_epochs, k=512, c=256, n=1024, h=4, w=4),
            TestConfig(num_epochs=num_epochs, k=512, c=512, n=1024, h=2, w=2),
        ]

        # (128, 1, 128, 128)
        input2_configs = [
            TestConfig(num_epochs=num_epochs, k=64, c=64, n=128, h=32, w=32),
            TestConfig(num_epochs=num_epochs, k=128, c=64, n=128, h=32, w=32),
            TestConfig(num_epochs=num_epochs, k=128, c=128, n=128, h=16, w=16),
            TestConfig(num_epochs=num_epochs, k=256, c=128, n=128, h=16, w=16),
            TestConfig(num_epochs=num_epochs, k=256, c=256, n=128, h=8, w=8),
            TestConfig(num_epochs=num_epochs, k=512, c=256, n=128, h=8, w=8),
            TestConfig(num_epochs=num_epochs, k=512, c=512, n=128, h=4, w=4),            
        ]

        small_configs = [
            TestConfig(num_epochs=num_epochs, k=16, c=16, n=100, h=32, w=32),
            TestConfig(num_epochs=num_epochs, k=16, c=16, n=100, h=16, w=16),
            TestConfig(num_epochs=num_epochs, k=16, c=16, n=100, h=8, w=8),
            TestConfig(num_epochs=num_epochs, k=16, c=16, n=100, h=4, w=4),
            TestConfig(num_epochs=num_epochs, k=16, c=16, n=10, h=32, w=32),
            TestConfig(num_epochs=num_epochs, k=16, c=16, n=10, h=16, w=16),
            TestConfig(num_epochs=num_epochs, k=16, c=16, n=10, h=8, w=8),
            TestConfig(num_epochs=num_epochs, k=16, c=16, n=10, h=4, w=4),
            TestConfig(num_epochs=num_epochs, k=16, c=16, n=1, h=32, w=32),
            TestConfig(num_epochs=num_epochs, k=16, c=16, n=1, h=16, w=16),
            TestConfig(num_epochs=num_epochs, k=16, c=16, n=1, h=8, w=8),
            TestConfig(num_epochs=num_epochs, k=16, c=16, n=1, h=4, w=4),
        ]

        # Warmup first
        print("Warming up....")
        warmup_conv_tc_perf(input1_configs[0], version=conv_version);
        print("Done warming up!")
        print("******************************************************** Input 1 Test Configs *********************************************************")
        input1_results = [test_conv_tc_perf(config, version=conv_version) for config in input1_configs]
        print("***************************************************************************************************************************************")

        print("******************************************************** Input 2 Test Configs *********************************************************")
        input2_results = [test_conv_tc_perf(config, version=conv_version) for config in input2_configs]
            
        print("***************************************************************************************************************************************")

        print("************************************************************ Small Configs ************************************************************")
        small_results = [test_conv_tc_perf(config, version=conv_version) for config in small_configs]
        print("***************************************************************************************************************************************")

        print(f"Geomean relative speedup on input1 configs: {geo_mean(input1_results)}")
        print(f"Geomean relative speedup on input2 configs: {geo_mean(input2_results)}")
        print(f"Geomean relative speedup on small configs: {geo_mean(small_results)}")

    elif args.acc:
        test_conv_acc()
        test_conv_tc_acc()
    else:
        print(f"Unrecognized argument combination, expected --perf or --acc, but got None of them. Exiting...")

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--perf", action="store_true")
    parser.add_argument("--acc", action="store_true")
    args = parser.parse_args()
    main(args)