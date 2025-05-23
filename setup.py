from setuptools import setup, find_packages

setup(
    name="disaster-response-vr",
    version="0.1.0",
    description="AI-Driven VR/AR Geospatial Analytics for Disaster Response",
    author="Your Name",
    author_email="your.email@example.com",
    packages=find_packages(),
    install_requires=[
        line.strip() for line in open("requirements.txt").readlines()
        if not line.startswith("#") and line.strip()
    ],
    python_requires=">=3.8",
) 