from setuptools import setup

setup(
    name='chemlog',
    version='1.0.0',
    packages=['chemlog'],
    install_requires=[
        'fastobo',
        'networkx',
        'pandas',
        'rdkit',
        'requests',
        'tqdm',
        'click',
        'gavel'
    ],
    author='Simon Flügel',
    author_email='simon.fluegel@uos.de',
    description='Peptide classifier for ChEBI / PubChem',
    license='MIT',

)