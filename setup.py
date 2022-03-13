from setuptools import setup, find_packages

import os

def get_install_requirements(path):
    content = open(os.path.join(os.path.dirname(__file__), path)).read()
    return [req for req in content.split("\n") if req != "" and not req.startswith("#")]

def setup_package():
    setup(
        setup_requires=["setuptools_scm"],
        name="cyclic-boosting",
        author="Blue Yonder GmbH",
        install_requires=get_install_requirements("requirements.txt"),
        packages=find_packages(),
        classifiers=["Programming Language :: Python"],
        use_scm_version=True
    )

if __name__ == "__main__":
    setup_package()
