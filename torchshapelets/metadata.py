import io
import os
import re


project = 'torchshapelets'
author = "Patrick Kidger"
copyright = "2020, {}".format(author)
author_email = "contact@kidger.site"
url = "https://github.com/patrick-kidger/generalised_shapelets"
license = "MIT"
python_requires = ">=3.5, <4"
keywords = "shapelets"
classifiers = ["Development Status :: 3 - Alpha",
               "Intended Audience :: Developers",
               "Intended Audience :: Science/Research",
               "License :: OSI Approved :: MIT License",
               "Natural Language :: English",
               "Operating System :: MacOS :: MacOS X",
               "Operating System :: Microsoft :: Windows",
               "Operating System :: Unix",
               "Programming Language :: Python :: 3",
               "Programming Language :: Python :: 3.5",
               "Programming Language :: Python :: 3.6",
               "Programming Language :: Python :: 3.7",
               "Programming Language :: Python :: Implementation :: CPython",
               "Topic :: Scientific/Engineering :: Artificial Intelligence",
               "Topic :: Scientific/Engineering :: Information Analysis",
               "Topic :: Scientific/Engineering :: Mathematics"]

description = 'Generalised learnt shapelets with pseudometric discrepancies and interpretable regularisation.'

here = os.path.realpath(os.path.dirname(__file__))

# for simplicity we actually store the version in the __version__ attribute in the source
with io.open(os.path.join(here, 'src', project, '__init__.py')) as f:
    meta_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", f.read(), re.M)
    if meta_match:
        version = meta_match.group(1)
    else:
        raise RuntimeError("Unable to find __version__ string.")

with io.open(os.path.join(here, 'README.md'), 'r', encoding='utf-8') as f:
    readme = f.read()
