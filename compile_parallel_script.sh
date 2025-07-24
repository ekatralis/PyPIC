export FC=gfortran
export F77=gfortran
export F90=gfortran

export LDFLAGS="-L/opt/homebrew/opt/libomp/lib"
export CPPFLAGS="-I/opt/homebrew/opt/libomp/include"
export FFLAGS="-fopenmp"

# f2py -m parallel_int_field_for -c parallel_interp_field_for.f90
f2py -m parallel_int_field_for_border -c parallel_interp_field_for_with_border.f90
