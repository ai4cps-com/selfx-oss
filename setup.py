import io
import os
from setuptools import setup, find_packages

main_ns = {}
exec(open("selfx/version.py", encoding="utf-8").read(), main_ns) # pylint: disable=exec-used, consider-using-with

def read_req_file(req_type):
    with open(os.path.join("requirements", f"{req_type}.txt"), encoding="utf-8") as fp:
        requires = (line.strip() for line in fp)
        return [req for req in requires if req and not req.startswith("#")]


setup(
    name="selfx",
    version=main_ns["__version__"],
    author="Nemanja Hranisavljevic",
    author_email="nemanja@ai4cps.com",
    packages=find_packages(exclude=["tests*"]),
    include_package_data=True,
    license="MIT",
    description=(
        "A Python framework for building apps for AI for CPS."
        "Developed by AI4CPS."
    ),
    long_description=io.open("README.md", encoding="utf-8").read(), # pylint: disable=consider-using-with
    long_description_content_type="text/markdown",
    install_requires=read_req_file("install"),
    python_requires=">=3.8",
    extras_require={
        "dev": read_req_file("dev"),
    },
    entry_points={
        "console_scripts": [],
        "pytest11": [],
    },
    url="https://ai4cps.com/selfx",
    project_urls={
        "Documentation": "https://selfx.ai4cps.com",
        "Source": "https://github.com/ai4cps/selfx-oss",
        "Issue Tracker": "https://github.com/ai4cps/selfx-oss/issues",
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Framework :: Dash",
        "Framework :: Flask",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Manufacturing",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python",
        "Topic :: Database :: Front-Ends",
        "Topic :: Scientific/Engineering :: Visualization",
        "Topic :: Software Development :: Libraries :: Application Frameworks",
    ],
    data_files=[],
)