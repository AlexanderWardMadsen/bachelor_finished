from setuptools import find_packages, setup

package_name = 'social_force_predictor'

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
    license='Apache-2.0',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'visualize_predicted = social_force_predictor.visualize_predicted:main',
            'sfp_ekf = social_force_predictor.sfp_ekf:main',
            'new_filter = social_force_predictor.new_filter:main',
        ],
    },
)
