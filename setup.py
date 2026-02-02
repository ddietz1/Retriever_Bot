from setuptools import find_packages, setup
import glob
import os

package_name = 'bluerov_control'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'),
         glob.glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Derek',
    maintainer_email='Derekdietz25@gmail.com',
    description='ROS bridge package',
    license='Apache 2.0',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'bridge_node = bluerov_control.bridge_node:main',
            'camera_node = bluerov_control.camera_node:main',
            'object_detect = bluerov_control.object_detection:main',
            'control_node = bluerov_control.control:main',
        ],
    },
)
