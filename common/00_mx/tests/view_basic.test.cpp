#include<iostream>
#include"mx/dense.h"
#include"mx/utils/ostream.h"

int main()
{
    mx::Dense<double> A(2,4), B(8,4);
    A.fill(1.0);
    B.fill(2.0);

    A(0,1) = 3.141;
    A(1,3) = 24.95;

    B(3,2) = 2.345;
    B(7,3) = 4.618;

    std::cout << "A = \n" << A << "\n";
    std::cout << "B = \n" << B << "\n";
    
    std::cout << "Transposed of A = \n" << A.view().transpose() << "\n";
    std::cout << "Transposed of A (copy) = \n" << A.transpose_copy() << "\n";
    std::cout << "Subview of B from element (3,2) of size [5,2] = \n" << B.view().subview(3,2,5,2) << "\n";
    std::cout << std::boolalpha 
              << "Is the subview contiguous ? " << static_cast<bool>(B.view().subview(3,2,5,2).is_contiguous()) << "\n";

    return 0;
}