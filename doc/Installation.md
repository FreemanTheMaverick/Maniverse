# Installation

## Prerequisites

For a C++ build, you need a C++17 compiler, GNU `make`, and Eigen.
The `makefile` uses the Eigen directory configured by `EIGEN3_PATH`.

For Python usage, you need Python 3.
You can install `Maniverse` with `pip`,
or build it from source which additionally requires a C++17 compiler
and the build dependencies listed in `pyproject.toml`.
The build process automatically downloads Eigen 5.0.1.
You also need to install NumPy
if you plan to use `Maniverse` directly in your own Python code.

## Layout produced by the build

A successful C++ `make` build generates these directories at the repository root:

* `include/Maniverse/{Manifold,Optimizer,LinearSolver,Diagonalizer}/…` — public headers
* `lib/libmaniverse.a` — static archive
* `lib/libmaniverse.so` — shared library

A Python package build produces a platform-specific `Maniverse` extension in the Python build/install location instead.

## Manual C++ build

1. **Get the source** by cloning the repository or unpacking a source release.

2. **Edit the top of `makefile`** to match your system:

   | Variable | Meaning |
   |----------|---------|
   | `CXX`       | your C++17 compiler |
   | `EIGEN3_PATH`| directory that directly contains `Eigen/`, `signature_of_eigen3_matrix_library`, `unsupported/` |
   | `CXXFLAGS`  | already includes `-std=c++17 -O3 -fPIC -march=native` |

   The Makefile builds the C++ libraries. Use the pip instructions above to build the Python extension.

3. **Build**:

   ```bash
   cd Maniverse
   make -j$(nproc)
   ```

   The make-file copies all public headers to `include/Maniverse/…` and produces both `lib/libmaniverse.a` and `lib/libmaniverse.so`.

4. **Run the C++ test-suite**:

   ```bash
   make test
   ```

   Each `test/*.cpp` is compiled, executed, and its output written to `test/*.out`. A test prints `Success` or `Failed: …` once it finishes.

## Python build

1. **Install the published package** from your package index:

   ```bash
   python -m pip install Maniverse
   ```

   This installs the pre-built Python extension.
   If you plan to use `Maniverse` directly in your own Python code,
   install NumPy separately:

   ```bash
   python -m pip install numpy
   ```

2. **Build from a source checkout**.
   First, get the source by cloning the repository
   or unpacking a source release.
   Install the Python build frontend if necessary:

   ```bash
   python -m pip install build
   ```

   Build a wheel and install the generated wheel:

   ```bash
   python -m build
   python -m pip install dist/*whl
   ```

   This downloads Eigen 5.0.1 and compiles the pybind11 extension.
   The build also requires a C++17 compiler and
   the dependencies listed in `pyproject.toml`.
   If you plan to use `Maniverse` directly in your own Python code, install NumPy separately:

   ```bash
   python -m pip install numpy
   ```

3. **Verify the installation**
by importing the extension from the active Python environment:

   ```python
   import Maniverse as mv
   print(mv.__version__)        # 2.3.0
   ```

4. **Run the Python test suite**

   Run the test aggregator from the `test/` directory because the tests load their data files relative to that directory:

   ```bash
   cd test
   python test.py
   ```

   The runner executing the Python test modules requires `unittest` to be installed:

   ```bash
   python -m pip install unittest
   ```