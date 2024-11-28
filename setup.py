from setuptools import setup, find_packages
try:
    from trustrag.version import __version__
except ImportError:
    __version__ = "unknown version"

# 读取 requirements.txt 文件中的内容
with open('requirements.txt') as f:
    required = f.read().splitlines()

setup(
    name="trustrag",
    version=__version__,
    author="trustrag-community",
    packages=['trustrag'],
    package_dir={'trustrag': 'trustrag'},
    package_data={'trustrag': ['*.*', 'applications/*', 'modules/document/*']},
    install_requires=required,
    author_email="yanqiang@ict.ac.cn",
    description="RAG Framework within Reliable input,Trusted output",
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url="https://github.com/gomate-community/GoMate",
    python_requires='>=3.9',
)