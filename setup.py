from setuptools import setup, find_packages
import os
from glob import glob

package_name = 'ranger_llm_ui'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['tests']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
    ],
    install_requires=[
        'setuptools',
        'gradio>=4.0.0',
        'langchain>=0.1.0',
        'langchain-openai>=0.0.5',
        'langchain-community>=0.0.10',
        'pydantic>=2.0.0',
    ],
    zip_safe=True,
    maintainer='Anh Nguyen',
    maintainer_email='anh0001@example.com',
    description='LLM-driven natural language operator UI for the Ranger robot',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'ui_node = ranger_llm_ui.ui_node:main',
        ],
    },
)
