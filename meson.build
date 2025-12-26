project('ladyns', 'c', 'fortran',
  version: '2.0.0',
  default_options: ['b_ndebug=if-release']
)

# Important: this project builds an extension module, so the package must live in platlib.
py = import('python').find_installation(pure: false)
py_dep = py.dependency()
cc = meson.get_compiler('c')

# ----------------------------
# Install pure Python sources
# ----------------------------
# Do not mark these as pure: true because the same package also contains an extension module.

py.install_sources(
  files(
    'ladyns/__init__.py',
    'ladyns/estimate.py',
    'ladyns/inference.py'
  ),
  subdir: 'ladyns'
)

py.install_sources(
  files(
    'ladyns/optimize/__init__.py',
    'ladyns/optimize/optimize.py'
  ),
  subdir: join_paths('ladyns', 'optimize')
)

# ----------------------------
# Include dirs for NumPy + f2py
# ----------------------------
numpy_inc = run_command(
  py,
  ['-c', 'import numpy; print(numpy.get_include())'],
  check: true
).stdout().strip()

f2py_inc = run_command(
  py,
  ['-c', 'import numpy.f2py as f2py; print(f2py.get_include())'],
  check: true
).stdout().strip()

inc_dirs = include_directories(numpy_inc, f2py_inc)

# ----------------------------
# Fortran sources
# ----------------------------
# Recommended layout: ladyns/optimize/_core/*.f90
f_sources = files(
  'ladyns/optimize/_core/lasso.f90',
  'ladyns/optimize/_core/glasso.f90',
  'ladyns/optimize/_core/glasso_p.f90'
)

# If you kept original paths, use instead:
# f_sources = files(
#   'ladyns/optimize/core/lasso.f90',
#   'ladyns/optimize/core/glasso.f90',
#   'ladyns/optimize/core/glasso_p.f90'
# )

# ----------------------------
# BLAS / LAPACK linking
# ----------------------------
blas_dep = dependency('libblas', required: false)
lapack_dep = dependency('liblapack', required: false)

if not blas_dep.found()
  blas_dep = dependency('blas', required: false)
endif
if not lapack_dep.found()
  lapack_dep = dependency('lapack', required: false)
endif

deps = [py_dep]
link_args = []

if blas_dep.found() and lapack_dep.found()
  deps += [blas_dep, lapack_dep]
else
  # Fallback: link against BLAS/LAPACK from the active Python prefix (conda friendly).
  py_prefix = run_command(
    py,
    ['-c', 'import sys; print(sys.prefix)'],
    check: true
  ).stdout().strip()
  conda_libdir = join_paths(py_prefix, 'lib')

  cc.find_library('blas', dirs: [conda_libdir], required: true)
  cc.find_library('lapack', dirs: [conda_libdir], required: true)

  link_args = [
    '-L' + conda_libdir,
    '-lblas',
    '-llapack'
  ]
endif

# ----------------------------
# f2py wrapper generation
# ----------------------------
f2py_c = custom_target(
  'f2py_wrapper_core',
  input: f_sources,
  output: 'coremodule.c',
  command: [
    py, '-m', 'numpy.f2py',
    '--build-dir', '@OUTDIR@',
    '--quiet',
    '--lower',
    '--overwrite-signature',
    '-m', 'core',
    '@INPUT@'
  ],
  build_by_default: true
)

# ----------------------------
# Build and install the extension
# ----------------------------
py.extension_module(
  'core',
  sources: [
    f2py_c,
    join_paths(f2py_inc, 'fortranobject.c'),
    f_sources
  ],
  include_directories: inc_dirs,
  dependencies: deps,
  link_args: link_args,
  install: true,
  subdir: join_paths('ladyns', 'optimize')
)
