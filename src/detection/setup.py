from glob import glob
import os
from setuptools import find_packages, setup

package_name = 'detection'

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
    maintainer='alex',
    maintainer_email='alexmadsen80@gmail.com',
    description='TODO: Package description',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
        'human_node = detection.human_node:main',
        'pose_app_node = detection.pose_app_node:main',
        'pose_app_node_v_0_2 = detection.pose_app_node_v_0_2:main',
        'pose_app = detection.pose_app:main',
        ],
    },
)
