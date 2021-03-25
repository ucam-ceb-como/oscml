from setuptools import setup, find_packages

setup(
    name='oscml',
    version='1.0.0',
    author='Andreas Eibeck, Daniel Nurkowski, Angiras Menon, Jiaru Bai',
    license='MIT',
    description='Organic solar cells PCE prediction models',
    long_description=open('README.md').read(),
    packages=find_packages(exclude=("tests")),
    python_requires='>=3.7, <4',
    entry_points={
        'console_scripts': [
             'oscml_run=oscml.hpo.train:start'
        ],
    },
    include_package_data=True
)