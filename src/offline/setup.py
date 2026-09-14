import glob
import os
from pathlib import Path
import subprocess
import sysconfig

import pybind11
from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext

HERE = Path(__file__).resolve().parent
VENDOR = HERE / 'vendor' / 'CompressGraph'
CORE = VENDOR / 'core' / 'batch'
if not (CORE / 'compress.hpp').is_file():
    raise RuntimeError('Missing CompressGraph dependency. Run git submodule update --init src/offline/vendor/CompressGraph from the repository root.')

sources = glob.glob(str(HERE / 'csrc' / '*.cpp')) + glob.glob(str(HERE / 'csrc' / '**' / '*.cc')) + glob.glob(str(HERE / 'csrc' / '**' / '*.cpp'))
sources += [str(HERE / 'bindings' / 'batch_cpu.cpp'), str(CORE / 'cpu.cpp'), str(CORE / 'common.cpp')]
extensions = [Extension(
    'compressgnn_offline', sources,
    include_dirs=[str(HERE / 'csrc'), str(VENDOR), pybind11.get_include()],
    language='c++', extra_compile_args=['-std=c++17', '-fopenmp', '-O3'],
    extra_link_args=['-lgomp'])]

BUILD_CUDA = os.environ.get('COMPRESSGNN_BUILD_CUDA', '0')
if BUILD_CUDA not in ('0', '1'):
    raise RuntimeError('COMPRESSGNN_BUILD_CUDA must be 0 or 1')
if BUILD_CUDA == '1':
    extensions.append(Extension('compressgnn_batch_cuda', sources=[]))


class BuildExtensions(build_ext):
    def build_extension(self, ext):
        if ext.name != 'compressgnn_batch_cuda':
            return super().build_extension(ext)
        cuda = Path(os.environ.get('COMPRESSGNN_CUDA_HOME', os.environ.get('CUDA_HOME', '/usr/local/cuda')))
        nvcc = cuda / 'bin' / 'nvcc'
        if not nvcc.is_file():
            raise RuntimeError('CUDA compiler missing; set COMPRESSGNN_CUDA_HOME or build CPU-only')
        arch = os.environ.get('COMPRESSGNN_CUDA_ARCH', '75')
        if not arch.isdigit():
            raise RuntimeError('COMPRESSGNN_CUDA_ARCH must be a numeric compute capability, e.g. 86')
        output = Path(self.get_ext_fullpath(ext.name)).resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        command = [str(nvcc), '-O3', '-std=c++17', '--shared', '-Xcompiler', '-fPIC',
                   '-gencode', 'arch=compute_{0},code=sm_{0}'.format(arch),
                   '-gencode', 'arch=compute_{0},code=compute_{0}'.format(arch),
                   '-I'+str(VENDOR), '-I'+pybind11.get_include(), '-I'+sysconfig.get_path('include'),
                   str(CORE/'cuda.cu'), str(CORE/'common.cpp'), str(HERE/'bindings'/'batch_cuda.cpp'),
                   '-Xlinker', '-rpath', '-Xlinker', str(cuda/'lib64'), '-o', str(output)]
        subprocess.run(command, check=True)


setup(name='compressgnn_offline', version='1.1', python_requires='>=3.6',
      description='CompressGNN offline libraries with CompressGraph batch compression',
      ext_modules=extensions, cmdclass={'build_ext': BuildExtensions},
      install_requires=['pytest', 'pybind11'])
