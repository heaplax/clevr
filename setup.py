from pathlib import Path

from setuptools import find_packages, setup

if __name__ == "__main__":

    REQUIREMENTS = [
    ]

    setup(
        name="clevr",
        packages=find_packages(),
        include_package_data=True,
        version="0.0.1",
        license="Apache 2.0",
        description="CLEVR dataset test framework",
        keywords=["machine learning"],
        install_requires=REQUIREMENTS,
    )
