git submodule init
git submodule update
cd build
cmake .. \
-DPYTHON_INCLUDE_DIR=$(python -c "import sysconfig; print(sysconfig.get_path('include'))")  \
-DPYTHON_LIBRARY=$(python -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))")
make
export PYTHONPATH=$PWD:$PYTHONPATH
cd ..
stubgen -p tensorlib -o tensorlib/ -p tensorlib.so
mv tensorlib/tensorlib.pyi tensorlib/__init__.pyi
