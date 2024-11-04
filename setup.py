from setuptools import setup

setup(
    name="tracers",
    version="0.1dev",
    author="Max Jacobi",
    packages=["tracers"],
    include_package_data=True,
    install_requires=[
        "numpy",
        "scipy",
        "h5py",
        "matplotlib",
    ],
    python_requires='>=3.10',
    description="Tracer integration in post-processing for hydro codes",
)
