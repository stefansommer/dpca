/* File: gaussian.i */
%module gaussian

%{
    #define SWIG_FILE_WITH_INIT
    #include "gaussian.hpp"
%}

%include "numpy.i"

%init %{
    import_array();
%}


/* in/out declarations */
 
%apply (double* IN_ARRAY1, int DIM1) {(double* in, int inn)};
%apply (double* INPLACE_ARRAY1, int DIM1) {(double *out, int outn)};

%include "gaussian.hpp"
