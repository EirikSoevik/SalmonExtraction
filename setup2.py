# import pip
# import logging
# import pkg_resources
try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup


with open('requirements.txt') as f:
    install_requires = f.read().strip().split('\n')

setup(
    name='mask_rcnn',
    version='2.1',
    install_requires=install_requires,
    # dependency_links=[
    #     'https://github.com/frappe/python-pdfkit.git#egg=pdfkit'
    # ],
    # cmdclass = \
    # {
    #     'clean': CleanCommand
    # }
)


# setup(
#     name='mask-rcnn',
#     version='2.1',
#     url='https://github.com/matterport/Mask_RCNN',
#     author='Matterport',
#     author_email='waleed.abdulla@gmail.com',
#     license='MIT',
#     description='Mask R-CNN for object detection and instance segmentation',
#     packages=["mrcnn"],
#     install_requires=install_reqs,
#     include_package_data=True,
#     python_requires='>=3.4',
#     long_description="""This is an implementation of Mask R-CNN on Python 3, Keras, and TensorFlow.
# The model generates bounding boxes and segmentation masks for each instance of an object in the image.
# It's based on Feature Pyramid Network (FPN) and a ResNet101 backbone.""",
#     classifiers=[
#         "Development Status :: 5 - Production/Stable",
#         "Environment :: Console",
#         "Intended Audience :: Developers",
#         "Intended Audience :: Information Technology",
#         "Intended Audience :: Education",
#         "Intended Audience :: Science/Research",
#         "License :: OSI Approved :: MIT License",
#         "Natural Language :: English",
#         "Operating System :: OS Independent",
#         "Topic :: Scientific/Engineering :: Artificial Intelligence",
#         "Topic :: Scientific/Engineering :: Image Recognition",
#         "Topic :: Scientific/Engineering :: Visualization",
#         "Topic :: Scientific/Engineering :: Image Segmentation",
#         'Programming Language :: Python :: 3.4',
#         'Programming Language :: Python :: 3.5',
#         'Programming Language :: Python :: 3.6',
#     ],
#     keywords="image instance segmentation object detection mask rcnn r-cnn tensorflow keras",
# )
