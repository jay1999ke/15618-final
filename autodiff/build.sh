git submodule init
git submodule update
cd build
cmake .. \
-DPYTHON_INCLUDE_DIR=$(python -c "import sysconfig; print(sysconfig.get_path('include'))")  \
-DPYTHON_LIBRARY=$(python -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))")
make
export PYTHONPATH=$PWD:$PYTHONPATH
pybind11-stubgen tensorlib -o . --no-setup-py
mv tensorlib-stubs/__init__.pyi ../tensorlib/__init__.pyi
rm tensorlib-stubs -rf
cd ..