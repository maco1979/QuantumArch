"""QuantumArch — 受量子力学启发的神经网络架构范式"""

from setuptools import find_packages, setup

# 读取 README.md 作为长描述
with open("README.md", encoding="utf-8") as f:
    long_description = f.read()

# 显式列出所有包，避免 setuptools 多顶层包歧义
packages = [
    "design",
    "experiments",
    "engineering",
    "paper_system",
    "quantum_core",
    "verification_suite",
    "optimization_system",
]

# 确保 __init__.py 存在
for pkg in packages:
    import os
    init_path = os.path.join(pkg, "__init__.py")
    if not os.path.exists(init_path):
        with open(init_path, "w", encoding="utf-8") as f:
            f.write(f'"""{pkg} package"""\n')

setup(
    name="QuantumArch",
    version="0.1.0",
    author="maco1979",
    description="Quantum-Inspired Neural Architecture with Complex-Valued Dynamics",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/maco1979/QuantumArch",
    packages=packages,
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "pyyaml>=6.0",
        "prometheus-client>=0.17.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.0.0",
            "pyright>=1.1.300",
            "isort>=5.12.0",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
