from setuptools import setup
import torch.utils.cpp_extension as torch_cpp_ext
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os
import pathlib
setup_dir = os.path.dirname(os.path.realpath(__file__))
HERE = pathlib.Path(__file__).absolute().parent

def remove_unwanted_pytorch_nvcc_flags():
    REMOVE_NVCC_FLAGS = [
        '-D__CUDA_NO_HALF_OPERATORS__',
        '-D__CUDA_NO_HALF_CONVERSIONS__',
        '-D__CUDA_NO_BFLOAT16_CONVERSIONS__',
        '-D__CUDA_NO_HALF2_OPERATORS__',
    ]
    for flag in REMOVE_NVCC_FLAGS:
        try:
            torch_cpp_ext.COMMON_NVCC_FLAGS.remove(flag)
        except ValueError:
            pass

def get_cuda_arch_flags():
    import subprocess, shutil

    candidates = ["75", "80", "86", "90"]
    nvcc = shutil.which("nvcc")
    if nvcc is not None:
        try:
            supported = set(subprocess.check_output([nvcc, "--list-gpu-arch"], text=True).split())
            candidates = [arch for arch in candidates if f"compute_{arch}" in supported]
        except Exception:
            pass
    flags = []
    for arch in candidates:
        flags.extend(["-gencode", f"arch=compute_{arch},code=sm_{arch}"])
    return flags
    
def get_marlin_arch_flags():
    import subprocess, shutil

    candidates = ["80", "86", "89", "90"]
    nvcc = shutil.which("nvcc")
    if nvcc is not None:
        try:
            supported = set(subprocess.check_output([nvcc, "--list-gpu-arch"], text=True).split())
            candidates = [arch for arch in candidates if f"compute_{arch}" in supported]
        except Exception:
            pass
    flags = []
    for arch in candidates:
        flags.extend(["-gencode", f"arch=compute_{arch},code=sm_{arch}"])
    return flags


def third_party_cmake(extra_pip_flags=None):
    import subprocess, sys, shutil
    
    cmake = shutil.which('cmake')
    if cmake is None:
            raise RuntimeError('Cannot find CMake executable.')

    retcode = subprocess.call([
        cmake, 
        "-DCMAKE_CUDA_ARCHITECTURES=75;80;86",
        HERE
    ])
    if retcode != 0:
        sys.stderr.write("Error: CMake configuration failed.\n")
        sys.exit(1)

    # install fast hadamard transform
    hadamard_dir = os.path.join(HERE, 'third-party/fast-hadamard-transform')
    try:
        import fast_hadamard_transform  # noqa: F401
        return
    except ImportError:
        pass
    pip = shutil.which('pip')
    
    # Build pip command with base flags. Use the current interpreter so the
    # fast-hadamard-transform build can see the same torch installation.
    pip_cmd = [sys.executable, '-m', 'pip', 'install', '-e', hadamard_dir, '--no-build-isolation', '--no-deps']
    
    # Add extra flags if provided
    if extra_pip_flags:
        pip_cmd.extend(extra_pip_flags)
    
    retcode = subprocess.call(pip_cmd)
    if retcode != 0:
        sys.stderr.write("Error: fast-hadamard-transform installation failed.\n")
        sys.exit(1)

def get_build_args():
    """Get pip build arguments from BUILD_ARGS environment variable"""
    build_args = os.environ.get('BUILD_ARGS', '')
    if build_args:
        return build_args.split()
    return []

def get_kernels():
    extra_kernels = os.environ.get('BUILD_KERNELS', '')
    default_kernels = [
        'deploy/kernels/bindings.cpp',
        'deploy/kernels/gemm.cu',
        'deploy/kernels/quant.cu',
        'deploy/kernels/flashinfer.cu',
    ]
    if extra_kernels:
        return extra_kernels.split() + default_kernels
    else:
        return default_kernels


def get_marlin_kernels():
    return [
        'third-party/marlin/marlin/marlin_cuda.cpp',
        'third-party/marlin/marlin/marlin_cuda_kernel.cu',
    ]


def get_extensions():
    extensions = [
        CUDAExtension(
            name='deploy._CUDA',
            sources=get_kernels(),
            include_dirs=get_include_dirs(),
            extra_compile_args={
                'cxx': [],
                'nvcc': get_cuda_arch_flags(),
            }
        )
    ]
    marlin_sources = get_marlin_kernels()
    if all(os.path.exists(os.path.join(setup_dir, source)) for source in marlin_sources):
        extensions.append(
            CUDAExtension(
                name='deploy._MARLIN',
                sources=marlin_sources,
                extra_compile_args={
                    'cxx': [],
                    'nvcc': get_marlin_arch_flags(),
                }
            )
        )
    else:
        print('Warning: third-party/marlin is unavailable; building without Marlin W4A16 support.')
    return extensions


def get_include_dirs():
    include_dirs = [
        os.path.join(setup_dir, 'deploy/kernels/include'),
        os.path.join(setup_dir, 'third-party/cutlass/include'),
        os.path.join(setup_dir, 'third-party/cutlass/tools/util/include'),
    ]
    cuda_home = torch_cpp_ext.CUDA_HOME or os.environ.get('CUDA_HOME') or '/usr/local/cuda'
    cccl_include = os.path.join(cuda_home, 'include', 'cccl')
    if os.path.isdir(cccl_include):
        include_dirs.append(cccl_include)
    return include_dirs

if __name__ == '__main__':
    # Get build args from environment variable
    extra_pip_flags = get_build_args()
    
    # Call third_party_cmake with extra flags
    third_party_cmake(extra_pip_flags if extra_pip_flags else None)
    
    remove_unwanted_pytorch_nvcc_flags()
    setup(
        name='flatquant',
        packages=['flatquant', 'deploy', 'flatquant_w4a4'],
        ext_modules=get_extensions(),
        cmdclass={
            'build_ext': BuildExtension
        }
    )
