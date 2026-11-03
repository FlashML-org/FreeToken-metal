from mlx import extension
from setuptools import setup

setup(
    ext_modules=[
        extension.CMakeExtension("ftmetal.kernel._C", sourcedir="python/ftmetal/kernel"),
        extension.CMakeExtension(
            "ftmetal.command_graph._C", sourcedir="python/ftmetal/command_graph/csrc"
        ),
    ],
    cmdclass={"build_ext": extension.CMakeBuild},
)
