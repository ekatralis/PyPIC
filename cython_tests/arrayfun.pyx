import numpy as np
cimport numpy as np
cimport cython

np.import_array()

ctypedef np.intp_t INT_t
ctypedef np.npy_bool BOOL_t

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef tuple[np.ndarray, np.ndarray] where_and_wherenot(np.ndarray[BOOL_t, ndim = 1] x):
    # Use Py_ssize_t instead of int for python interfacing
    cdef Py_ssize_t array_len = x.shape[0]
    cdef Py_ssize_t i = 0, iwhere = 0, iwherenot = 0 
    cdef np.ndarray[INT_t, ndim=1] where_array = np.empty(array_len, dtype=np.intp)
    cdef np.ndarray[INT_t, ndim=1] wherenot_array = np.empty(array_len, dtype=np.intp)

    # Create array views:
    cdef BOOL_t[:] x_view = x
    cdef INT_t[:] where_view = where_array
    cdef INT_t[:] wherenot_view = wherenot_array
    

    for i in range(array_len):
       if x_view[i]:
           where_view[iwhere] = i
           iwhere += 1
       else:
           wherenot_view[iwherenot] = i
           iwherenot += 1

    # for i in range(array_len):
    #     if x[i]:
    #         where_array[iwhere] = i
    #         iwhere += 1
    #     else:
    #         wherenot_array[iwherenot] = i
    #         iwherenot += 1

    return where_array[:iwhere], wherenot_array[:iwherenot]
    # return where_array[:iwhere].copy(), wherenot_array[:iwherenot].copy() for freeing up memory usage