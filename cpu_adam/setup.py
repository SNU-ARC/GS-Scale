from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CppExtension

setup(
    name='cpu_adam',
    ext_modules=[
        CppExtension(
        name='cpu_adam', 
        sources=['src/cpu_adam_api.cpp', 'src/cpu_adam.cpp'],
        extra_compile_args=['-O3', '-g', '-fopenmp', '-pthread', '-std=c++17'],
        extra_link_args=['-fopenmp'])
    ],
    cmdclass={'build_ext': BuildExtension},
    zip_safe=False,
)

