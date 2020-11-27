from setuptools import setup, find_packages

setup(
    name='darwin',
    version='0.1.0',
    packages=find_packages(),
    package_data={
        '': ['*.pyx'],
    }
)

