from setuptools import setup,find_packages
from typing import List
FILEPATH='requirements.txt'
HYPHEN_E_DOT='-e .'

def get_requirements(filepath:str)-> List[str]:
    requirements=[]
    try:
        with open(filepath) as file:
            req=file.readlines()
            requirements=[r.replace('\n','') for r in req]
            if HYPHEN_E_DOT in requirements:
                requirements.remove(HYPHEN_E_DOT)
            return requirements
    except Exception as e:
        print('Cannot Open or Find the File', e)

setup(
    name='Network Security Project',
    version='0.0.1',
    description='Network Security Project',
    author='Sambhav Surthi',
    author_email='sambhavsurthi08@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements(filepath=FILEPATH)
)