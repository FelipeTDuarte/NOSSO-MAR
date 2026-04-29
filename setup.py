from setuptools import setup, find_packages

setup(
    name="nossomar",
    version="0.1.0",
    description="NOSSO-MAR: Neural Operator Scalable System for Ocean Modeling",
    packages=find_packages("src"),
    package_dir={"": "src"},
    install_requires=[
        "numpy",
        "scipy",
        "xarray",
        "pandas",
        "torch",
        "netCDF4",
    ],
    include_package_data=True,
)
