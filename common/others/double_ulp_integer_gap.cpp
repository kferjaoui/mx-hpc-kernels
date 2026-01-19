#include <iostream>
#include <iomanip>

int main() {
    double x = 9007199254740992.0; // 2^53
    std::cout << std::fixed << std::setprecision(0);
    std::cout << "x-1   = " << x - 1 << "\n";
    std::cout << "x     = " << x << "\n";
    std::cout << "x+1   = " << (x + 1) << "\n"; // same as x
    std::cout << "x+2   = " << (x + 2) << "\n"; // next representable integer
}
