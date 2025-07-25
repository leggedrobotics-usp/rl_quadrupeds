import os
from setuptools import find_packages, setup
from glob import glob

package_name = 'go1_slam'
urdf = [
    f for f in glob(
        f'{package_name}/resources/go1/*',
        recursive=True
    ) if os.path.isfile(f)
]

meshes = [
    f for f in glob(
        f'{package_name}/resources/go1/meshes/*',
        recursive=True
    ) if os.path.isfile(f)
]

#ros2_ws/install/go1_slam/share/go1_slam/rviz2/rviz2.rviz

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob('launch/*')),
        ('share/' + package_name + f'/{package_name}/resources/go1', urdf),
        ('share/' + package_name + f'/{package_name}/resources/go1/meshes', meshes),
        ('share/' + package_name + '/rviz2', glob('rviz2/*.rviz')),
        ('share/' + package_name + '/config', glob('config/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='root',
    maintainer_email='root@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            f'my_node = {package_name}.my_node:main'
        ],
    },
)
