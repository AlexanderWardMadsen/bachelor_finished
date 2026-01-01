from glob import glob
import os
from setuptools import find_packages, setup

package_name = 'social_force_layer'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),

        (os.path.join("share", package_name, "launch"), glob("launch/*.py")),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='lucas',
    maintainer_email='lujoe21@student.sdu.dk',
    description='TODO: Package description',
    license='Apache-2.0',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'social_force_predictor = social_force_layer.social_force_predictor:main',
            'social_force_model = social_force_layer.social_force_model:main',
        ],
    },
)
