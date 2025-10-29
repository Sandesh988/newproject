from setuptools import find_packages,setup
from typing import List
hypen_e_dot='-e .'
def get_requirement(file_path:str)->List[str]:
    reqirement=[]
    with open(file_path) as file:
        reqirement=file.readlines()
        reqirement=[req.replace("\n","")for req in reqirement]

        if hypen_e_dot in reqirement:
            reqirement.remove(hypen_e_dot)
    return reqirement


setup(
name='mlproject',
version='0.0.1',
author='sandesh k',
author_email='sandeshsandeshk27@gmail.com',
packages=find_packages(),
install_requires=get_requirement('requirements.txt'),
)