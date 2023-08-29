from pkg_resources import parse_requirements
from setuptools import find_packages, setup

with open("requirements.txt") as f:
    requirements = [str(req) for req in parse_requirements(f.read())]

setup(
    name="cpi_prediction",
    version="1.1.0",
    package_dir={"": "src"},
    packages=find_packages("src"),
    package_data={"cpi_prediction": ["data/*.db"]},
    install_requires=requirements,
)
