#include <iostream>
#include "mx/dense.h"
#include "mx/dense_view.h"
#include "mx/utils/ostream.h"
#include "lu_unblocked.h"


int main(){
    // M is a rectangular matrix (i.e. more rows than columns)
    mx::Dense<double> M(6, 4, {1,2,3,4, 2,5,6,7, 3,1,4,1, 0,2,2,2, 5,0,1,3, 1,1,0,0});
    std::cout << "Original matrix M:\n" << M << std::endl;

    std::vector<mx::index_t> piv_M(3);
    mx::LUInfo info = mx::lu_unblocked(M.view(), piv_M);

    std::cout << "LU factorization info: " << info << std::endl;

    std::cout << "The pivot vector is: [ ";
    for(const auto& p : piv_M) std::cout << p << " ";
    std::cout << "]\n";

    std::cout << "The in-place LU after factorization:\n" << M << std::endl;
}