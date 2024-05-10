from setuptools import find_packages,setup

setup(
    name='ComputerVision',
    version='0.0.1',
    author='marcel wolfram',
    author_email='marcel.wolfram@iaeste.pl',
    install_requires=[
        "pathlib",
        "pytest-shutil",
        "pillow",
        "matplotlib",
        "tqdm",
        "pytest-timeit",
        "torchmetrics",
        "mlxtend",
        "scikit-learn",
        "torchvision",
        "torch",
        "torchinfo"],
    packages=find_packages()
)