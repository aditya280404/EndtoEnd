from setuptools import find_packages,setup
from typing import List
hypen='-e .'
def get_require(file_path:str)->List[str]:
    requirements=[]
    with open(file_path) as file_obj:
        requirements=file_obj.readlines()
        requirements=[req.replace('\n','')for req in requirements]
        if hypen in requirements:
            requirements.remove(hypen)
    return requirements

setup(
    name='EndtoEnd',
    version=0.1,
    author='Aditya',
    packages=find_packages(),
    install_requires=get_require('requirements.txt')
    
    
)