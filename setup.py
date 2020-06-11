import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="modelfarm", # Replace with your own username
    version="0.1.0",
    author="Seunghye Wilson",
    author_email="Seunghye.Wilson@agilebeat.com",
    description="Package containing models to read OSM",
    long_description="Package containing models to read OSM",
    long_description_content_type="text",
    url="https://github.com/pypa/sampleproject",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)