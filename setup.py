from mlx import extension
from setuptools import setup

setup(
    ext_modules=[
        extension.CMakeExtension("ftmetal.kernel._C", sourcedir="python/ftmetal/kernel/csrc")
    ],
    cmdclass={"build_ext": extension.CMakeBuild},
)
