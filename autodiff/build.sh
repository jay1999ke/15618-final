git submodule init
git submodule update
python3 -m pip install pybind11-stubgen
cd build
cmake .. \
-DPYTHON_INCLUDE_DIR=$(python3 -c "import sysconfig; print(sysconfig.get_path('include'))")  \
-DPYTHON_LIBRARY=$(python3 -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))")
make
export PYTHONPATH=$PWD:$PYTHONPATH
pypath=$(python3 -m site --user-base)
if ! command -v pybind11-stubgen &> /dev/null
then
    # for ghc cluster
    alias pybind11-stubgen=$pypath/bin/pybind11-stubgen
fi
pybind11-stubgen tensorlib -o . --no-setup-py
mv tensorlib-stubs/__init__.pyi ../tensorlib/__init__.pyi
rm tensorlib-stubs -rf
cd ..