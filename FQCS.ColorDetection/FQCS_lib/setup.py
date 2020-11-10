import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="FQCS",
    version="1.1.3",
    author="FQCS Team",
    author_email="trannamtrung1st@gmail.com",
    description="This is for FQCS Capstone Project",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/trannamtrung1st/FQCS-Research",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        'numpy', 'imutils', 'opencv-python', 'scipy', 'tensorflow',
        'tensorflow-addons'
    ],
    python_requires='>=3.5',
)
