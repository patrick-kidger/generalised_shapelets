import setuptools
import sys
import torch.utils.cpp_extension as cpp

import metadata


extra_compile_args = []

# fvisibility flag because of https://pybind11.readthedocs.io/en/stable/faq.html#someclass-declared-with-greater-visibility-than-the-type-of-its-field-someclass-member-wattributes
if not sys.platform.startswith('win'):  # linux or mac
    extra_compile_args.append('-fvisibility=hidden')

if sys.platform.startswith('win'):  # windows
    extra_compile_args.append('/openmp')
else:  # linux or mac
    extra_compile_args.append('-fopenmp')

ext_modules = [cpp.CppExtension(name='_impl',
                                sources=['src/discrepancies.cpp',
                                         'src/pytorchbind.cpp',
                                         'src/shapelet_transform.cpp'],
                                depends=['src/discrepancies.hpp',
                                         'src/shapelet_transform.hpp'],
                                extra_compile_args=extra_compile_args)]


setuptools.setup(name=metadata.project,
                 version=metadata.version,
                 author=metadata.author,
                 author_email=metadata.author_email,
                 maintainer=metadata.author,
                 maintainer_email=metadata.author_email,
                 description=metadata.description,
                 long_description=metadata.readme,
                 url=metadata.url,
                 license=metadata.license,
                 keywords=metadata.keywords,
                 classifiers=metadata.classifiers,
                 zip_safe=False,
                 python_requires=metadata.python_requires,
                 packages=[metadata.project],
                 ext_package=metadata.project,
                 package_dir={'': 'src'},
                 ext_modules=ext_modules,
                 cmdclass={'build_ext': cpp.BuildExtension})
