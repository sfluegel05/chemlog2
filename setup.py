from setuptools import setup

setup(
    name='chemlog2',
    version='0.1.0',
    packages=['chemlog2'],
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