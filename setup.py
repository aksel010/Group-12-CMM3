from setuptools import setup, find_packages

setup(
    name='btms_simulation',
    version='1.0.0',
    description='Battery Thermal Management System simulation',
    author='Group 12',
    packages=find_packages(),  # Discovers all packages under project root
    install_requires=[
        'numpy>=1.21.0',
        'scipy>=1.7.0',
        'matplotlib>=3.5.0',
        'pandas>=1.3.0',
    ],
    python_requires='>=3.8',
)
