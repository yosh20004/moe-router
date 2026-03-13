from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


class BuildExtensionInBuildDir(BuildExtension):
    def finalize_options(self):
        super().finalize_options()
        self.inplace = False


setup(
    name="moe_router",
    version="0.1.0",
    packages=["moe_router"],
    ext_modules=[
        CUDAExtension(
            name="moe_router._C",
            sources=[
                "moe_router/csrc/router.cpp",
                "moe_router/csrc/fused_topk_with_score_function.cu",
                "moe_router/csrc/fused_score_for_moe_aux_loss.cu",
                "moe_router/csrc/fused_moe_aux_loss.cu",
            ],
            extra_compile_args={
                "cxx": ["-O3", "-std=c++17"],
                "nvcc": ["-O3", "-std=c++17", "--use_fast_math"],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtensionInBuildDir},
)
