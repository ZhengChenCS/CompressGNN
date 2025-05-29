import os
import glob
from setuptools import find_packages, setup
import pybind11

import torch
from torch.utils.cpp_extension import CppExtension

torch_ver = [int(x) for x in torch.__version__.split(".")[:2]]
assert torch_ver >= [1, 3], "Requires PyTorch >= 1.3"

def get_extensions():
    this_dir = os.path.dirname(os.path.abspath(__file__))
    extensions_dir = os.path.join(this_dir, "csrc")
    
    # Collect all source files, excluding header files
    sources = glob.glob(os.path.join(extensions_dir, "*.cpp")) + \
              glob.glob(os.path.join(extensions_dir, "**", "*.cc"))

    # Use CppExtension for CPU or CUDAExtension for GPU
    extension = CppExtension  # Change to CUDAExtension if CUDA is needed

    include_dirs = [extensions_dir, pybind11.get_include()] + torch.utils.cpp_extension.include_paths()

    ext_modules = [
        extension(
            "compressgnn_sample",
            sources,
            include_dirs=include_dirs,
            language="c++",
            extra_compile_args={"cxx": ["-std=c++17", "-fopenmp", "-O3", "-w"]},
            extra_link_args=['-lgomp']
        )
    ]
    return ext_modules

setup(
    name="compressgnn_sample",
    version="1.0",
    python_requires=">=3.6",
    description="CompressGNN sample libraries",
    ext_modules=get_extensions(),
    install_requires=["pytest", "pybind11"],
    cmdclass={"build_ext": torch.utils.cpp_extension.BuildExtension}
)