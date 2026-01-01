from setuptools import find_packages, setup

package_name = 'costmap_2d'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='lucas',
    maintainer_email='lujoe21@student.sdu.dk',
    description='TODO: Package description',
    license='Todo',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
    'console_scripts': [
        'costmap_2d = costmap_2d.costmap_2d:main',
        'costmap_2d_2 = costmap_2d.costmap_2d_2:main',
        'visualizer = costmap_2d.costmap_visualizer:main',
    ],
}
)
