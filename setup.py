from setuptools import setup, find_packages
import os
import sys
from glob import glob

package_name = 'ranger_llm_ui'

# Add ros-technician-cli submodule to the Python path for imports
# This allows us to import from the 'rosa' package within the submodule
submodule_src = os.path.join(os.path.dirname(__file__), 'ros-technician-cli', 'src')
if os.path.exists(submodule_src) and submodule_src not in sys.path:
    sys.path.insert(0, submodule_src)

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['tests']),
    # Include the submodule's src directory as a package directory
    package_dir={
        '': '.',
        'rosa': 'ros-technician-cli/src/rosa',
    },
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
        # LangChain dependencies aligned with ros-technician-cli (ROSA)
        'langchain~=0.3.23',
        'langchain-openai~=0.3.14',
        'langchain-community~=0.3.21',
        'langchain-core~=0.3.52',
        'langchain-ollama~=0.3.2',
        'pydantic>=2.0.0',
        'python-dotenv>=1.0.1',
        'PyYAML>=6.0.1',
        'rich',  # Used by ROSA for output formatting
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
