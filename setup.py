from setuptools import setup

from pathlib import Path
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name='chemlog',
    long_description=long_description,
    long_description_content_type='text/markdown',
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
    license='GNU General Public License v3.0',

)