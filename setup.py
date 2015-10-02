from setuptools import setup
import os

def include_files(some_dir):
    files = [_ for _ in os.listdir(some_dir) if os.path.isfile(_)]
    return (some_dir, [os.path.join(some_dir, _) for _ in files])

setup(name='matmodlab',
      version='3.0.5',  # last updated on 151002 by Scot Swan
      description='Material model development laboratory',
      long_description=('The material model laboratory (*matmodlab*) is an '
                        'object oriented model driver. The majority of the '
                        'source code is written in Python and requires no '
                        'additional building.'), classifiers=[  # Classifier list:  https://pypi.python.org/pypi?:action=list_classifiers
                   "Development Status :: 4 - Beta",
                   "Environment :: Console",
                   "Intended Audience :: Science/Research",
                   "License :: OSI Approved :: MIT License",
                   "Natural Language :: English",
                   "Operating System :: OS Independent",
                   "Programming Language :: Fortran",
                   "Programming Language :: Python :: 2.7",
                   "Topic :: Scientific/Engineering",
                   "Topic :: Scientific/Engineering :: Mathematics",
                   "Topic :: Scientific/Engineering :: Physics",
                   "Topic :: Scientific/Engineering :: Visualization",
                  ],
      url='https://github.com/tjfulle/matmodlab',
      author='Timothy Fuller, Scot Swan',
      author_email='timothy.fuller@utah.edu,scot.swan@gmail.com',
      license='MIT',
      packages=[
                'matmodlab',
                'matmodlab.fitting',
                'matmodlab.lib',
                'matmodlab.materials',
                'matmodlab.mmd',
                'matmodlab.utils',
                'matmodlab.utils.fortran',
                'matmodlab.utils.numerix',
               ],
      package_dir={
                   'matmodlab':'matmodlab',
                  },
      package_data={
                    'matmodlab':[
                                 'materials/src/*',
                                 'utils/fortran/*',
                                 'ipynb/*.py',
                                 'ipynb/profile_default/db/.file',
                                 'ipynb/profile_default/startup/*.ipy',
                                 'ipynb/profile_default/static/custom/*',
                                 'ipynb/custom/*',
                                 'examples/*',
                                 'tutorial/*',
                                 'tests/*',
                                ],
                   },
      data_files=[
                  include_files('matmodlab/materials/src'),
                  include_files('matmodlab/tests'),
                  include_files('matmodlab/examples'),
                  include_files('matmodlab/tutorial'),
                  include_files('matmodlab/ipynb'),
                  include_files('matmodlab/utils/fortran'),
                 ],
      scripts=[
               'matmodlab/bin/fdiff',
               'matmodlab/bin/fdump',
               'matmodlab/bin/mmd',
               'matmodlab/bin/mml',
               'matmodlab/bin/mml-osx',
              ],
      install_requires=['numpy', 'scipy'],
      zip_safe=False)
