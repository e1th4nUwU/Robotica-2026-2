from setuptools import find_packages, setup

package_name = 'robot_description'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/rviz', ['rviz/rviz_proyecto.rviz']),
        ('share/' + package_name + '/urdf', ['urdf/robot_rrr_proyecto.urdf']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='eithan',
    maintainer_email='eithantrevino@gmail.com',
    description='Descripcion URDF y configuracion RViz del robot RRR en plano XY',
    license='TODO: License declaration',
    extras_require={
        'test': ['pytest'],
    },
    entry_points={
        'console_scripts': [],
    },
)
