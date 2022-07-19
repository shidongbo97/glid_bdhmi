#include "simplesparsematrix.h"
#include <map>
#include <ctime>
#include <limits>
#include <algorithm>

#define ABS(x) (((x)<0)?-(x):(x))
using namespace std;

template <class T>
T SimpleSparseMatrix<T>::infinityNorm(){
    T max = 0;
    for (int i = 0; i < nr; i++) {
        T rowSum = 0;

        for (int j = rowStart[i]; j < rowEnd(i); j++)
            rowSum += ABS(values[j]);

        max = (max < rowSum)?rowSum:max;
    }
    return max;
}

template <class T>
void SimpleSparseMatrix<T>::addAlpha(const vector<int> &rows, T alpha) {
    for (unsigned int i = 0; i < rows.size(); i++) {
        bool found = false;
        for (int j = rowStart[ rows[i] ]; j < rowEnd(rows[i]); j++)
            if (column[j] == rows[i]) {
                values[j] += alpha;
                found = true;
            }
    }
}

template <class T>
void SimpleSparseMatrix<T>::transpose(SimpleSparseMatrix<T> &result) {
    result.nc = nr;
    result.nr = nc;

    result.setCapacity(numNonzero);
    result.numNonzero = numNonzero;
    result.rowStart = (int*)realloc(result.rowStart,nc*sizeof(int));

    // colSize is malloc'ed -- eventually we should allocate it elsewhere
    int *colSize = (int*)malloc(nc*sizeof(int));
    memset(colSize, 0, nc*sizeof(int)); // make sure I did this right...

    for (int i = 0; i < numNonzero; i++) colSize[ column[i] ]++;

    result.rowStart[0] = 0;
    for (int i = 1; i < nc; i++)
        result.rowStart[i] = result.rowStart[i-1] + colSize[i-1];

    // now, reuse colSize to tell us now many things we've written
    memset(colSize, 0, nc*sizeof(int));
    for (int i = 0; i < nr; i++)
        for (int j = rowStart[i]; j < rowEnd(i); j++) {
            int position = colSize[ column[j] ]++;
            position += result.rowStart[ column[j] ];

            result.values[position] = values[j];
            result.column[position] = i;
        }
    free(colSize);
}

template <class T>
void SimpleSparseMatrix<T>::stack(SimpleSparseMatrix<T> **matrices, int numMatrices, int *colShifts) {

    SimpleSparseMatrix<T> &result = *this;
    int nonzeroSum = 0, rowSum = 0, maxCol = 0;
    for (int i = 0; i < numMatrices; i++) {
        nonzeroSum += matrices[i]->numNonzero;
        rowSum += matrices[i]->nr;
        maxCol = std::max(maxCol, matrices[i]->nc + colShifts[i]);
    }

    result.setCapacity(nonzeroSum);
    result.nr = rowSum;
    result.nc = maxCol;

    result.rowStart = (int*)realloc(result.rowStart, rowSum*sizeof(int));

    result.startMatrixFill();
    int rowShift = 0;
    for (int i = 0; i < numMatrices; i++) {
        for (int j = 0; j < matrices[i]->nr; j++)
            for (int k = matrices[i]->rowStart[j]; k < matrices[i]->rowEnd(j); k++)
                result.addElement(j+rowShift,matrices[i]->column[k]+colShifts[i],matrices[i]->values[k]);
        rowShift += matrices[i]->nr;
    }
}

template <class T>
SimpleSparseMatrix<T>::SimpleSparseMatrix()
    : nr(0), nc(0), numNonzero(0), values(NULL), rowStart(NULL), column(NULL) {

    int nz = 200;

    capacity = nz;
    values = (T*)malloc(nz*sizeof(T));
    rowStart = (int*)malloc(nr*sizeof(int));
    column = (int*)malloc(nz*sizeof(int));
}

template <class T>
SimpleSparseMatrix<T>::SimpleSparseMatrix(int r, int c, int nz)
    : nr(r), nc(c), numNonzero(0), capacity(nz) {

    values = (T*)malloc(nz*sizeof(T));
    rowStart = (int*)malloc(nr*sizeof(int));
    column = (int*)malloc(nz*sizeof(int));
}


template <class T>
SimpleSparseMatrix<T>::SimpleSparseMatrix(int r, int c, int nz, const int* rowstart, const int* colidxs, const T *vals)
    : nr(r), nc(c), numNonzero(nz), capacity(nz), values(NULL), rowStart(NULL), column(NULL) {

    values = (T*)malloc(nz*sizeof(T));
    rowStart = (int*)malloc(nr*sizeof(int));
    column = (int*)malloc(nz*sizeof(int));

    std::copy_n(rowstart, nr, rowStart);
    std::copy_n(colidxs, nz, column);
    std::copy_n(vals, nz, values);
}





template <class T>
void SimpleSparseMatrix<T>::multiply(SimpleSparseMatrix<T> &m, SimpleSparseMatrix<T> &result) {
    map<int, T> rowValues; // for a given column, holds nonzero values
    int curNonzero = 0;

    result.setRows(nr);
    result.setCols(m.nc);

    result.startMatrixFill();
    for (int i = 0; i < nr; i++) { // compute the i-th row
        result.rowStart[i] = curNonzero;

        rowValues.clear(); // start out with zeroes

        int end = rowEnd(i);

        for (int p = rowStart[i]; p < end; p++) { // for all elements in row i
            int k = column[p]; // this is the k-th column
            T value = values[p]; // value = element (i,k)

            int stop = m.rowEnd(k);
            for (int q = m.rowStart[k]; q < stop; q++) // look at row k in matrix m
                rowValues[ m.column[q] ] += value * m.values[q]; // add (i,k) * (k,q)
        }

        // now, copy those values into the matrix
        while (curNonzero + (int)rowValues.size() > result.capacity) result.doubleCapacity();

        // this better traverse in sorted order...
        for (typename map<int,T>::iterator it = rowValues.begin(); it != rowValues.end(); ++it) {
            //if (ABS(it->second) > numeric_limits<T>::epsilon())
                result.addElement(i,it->first,it->second);
        }

        curNonzero += rowValues.size();
    }
}

template class SimpleSparseMatrix<double>;
template class SimpleSparseMatrix<float>;
