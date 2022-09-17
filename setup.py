from setuptools import find_packages, setup

setup(
    name="geos",
    version="0.1.0",
    description="2D geometric primitives, spatial-grid rectangle merging, boolean CSG, and vectorized NumPy SDFs",
    author="Kartikesh Mishra",
    license="MIT",
    packages=find_packages(include=["geos", "geos.*"]),
    install_requires=[
        "numpy>=1.21.0",
    ],
    python_requires=">=3.8",
)
