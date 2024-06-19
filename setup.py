import re
from pathlib import Path

from setuptools import find_packages, setup


def get_version(file_path: Path):
    with open(file_path, "r") as file:
        file_content = file.read()

    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", file_content, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")


with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()


setup(
    name="stad",
    version=get_version(Path.cwd() / "stad" / "__init__.py"),
    packages=find_packages(),
    install_requires=[
        "jsonschema",
        "numpy",
        "pandas",
        "ffmpeg-python",
        "pyannote.audio",
        "torch",
        "transformers",
        "accelerate",
    ],
    author="Arthur Findelair",
    author_email="arthfind@gmail.com",
    description="Combines Whisper and PyAnnote for Automatic Speaker Diarized Speech Recognition",
    license="MIT",
    url="https://github.com/ArthurFDLR/speech-transcription-and-diarization",
    long_description=long_description,
    long_description_content_type="text/markdown",
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Libraries",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Operating System :: OS Independent",
    ],
)
