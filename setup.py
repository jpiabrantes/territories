from setuptools import setup, Extension
import numpy

territories_module = Extension(
    'binding',
    sources=['src/binding.c'],    
    include_dirs=['.', 'src', numpy.get_include(), '/opt/homebrew/include'],
    extra_compile_args=['-O3'],
    library_dirs=['/opt/homebrew/lib'],
    libraries=['raylib', 'm', 'pthread', 'dl'],
)

setup(
    name='territories',
    version='0.1',
    ext_modules=[territories_module],
    py_modules=[],
)
