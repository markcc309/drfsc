import setuptools
long_description = 'A Python package implementing a distributed randomised feature selection algorithm.'

with open('requirements.txt') as f:
    install_requires = f.read().splitlines()
    
with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="drfsc",
    version="0.9.0",
    author="Mark Chiu Chong",
    author_email="mark.chiuchong@uq.net.au",
    description="Python package for the DRFSC algorithm",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/markcc309/drfsc",
    package_dir={'drfsc': 'src'},
    packages=['drfsc'],
    package_data={'': ['data/wdbc.data']},
    classifiers=(
        "Topic :: Scientific/Engineering",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Development Status :: 4 - Beta",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ),
    include_package_data=True,
    install_requires=install_requires,
    license='MIT',
)
