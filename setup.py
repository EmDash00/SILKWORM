import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="silkworm",
    version="0.0.2.1",
    author="SIL",
    author_email="drasonchow@gmail.com",
    description="General python utilities made by SIL at IUB that we have" \
    " found to be useful.",
    long_description="""SILKWORM is the SIL Integrated Libraries and Knowledgebase
    for Work, Ongoing Research, and Managment by faculty at SIL. We have elected
    to make this set of very general utilities open source.""",
    long_description_content_type="text/markdown",
    url="https://github.com/EmDash00/SILKWORM",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Eclipse Public License 2.0 (EPL-2.0)",
        "Operating System :: OS Independent",
    ],
)

