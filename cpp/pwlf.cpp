/*
 * Testing a generic piecewise linear function
 *
 * Contributors:
 *       fahim.sheikh@gmail.com
 */

#include <iostream>
#include <string>

#include "pwlf.h"
#define compare_with_tol(x,y) assert(std::abs(x-y) <= 1.0e-6)

int main() {
    // Following initialization attempts should fail during compilation
    // pwlf::PwLinFunc<float, 1> g({{{0.0, 1.0}}});
    // pwlf::PwLinFunc<float, 0> g({{}});
    // std::array<std::pair<float, float>, 1> someArray = {{{0.0, 0.0}}};
    // pwlf::PwLinFunc<float, 1> g(someArray);

    try {
        pwlf::PwLinFunc<float, 2> f({ { {0.0, 0.0}, {1.0, 0.5} } });
        auto y = f(0.5);
        assert(y == 0.25f);

        std::array<std::pair<float, float>, 6> upPoints = { {
            {0.0, 0.0}, {2.0, 1.0}, {4.0, 1.0}, {5.0, 3.0}, {6.0, 4.0}, {8.0, 8.0}
        } };

        pwlf::PwLinFunc<float, 6> h(upPoints);
        assert(h(1) == 0.5f);
        assert(h(0.1) == 0.05f);
        assert(h(0.5) == 0.25f);
        assert(h(3.0) == 1.0f);
        assert(h(2.5) == 1.0f);
        assert(h(4.5) == 2.0f);
        compare_with_tol(h(4.9), 2.8f);
        assert(h(5.5) == 3.5f);
        compare_with_tol(h(7.0), 6.0f);
        std::cout << "Upward segement verified" << std::endl;
        // Uncomment to check out-of-range
        // std::cout << "Main segment verified, now checking out of range" << std::endl;
        // auto outOfRange = h.ordinate(10);

        std::array<std::pair<float, float>, 5> seesaw = { {
            {0.0, 1.0}, {2.0, 2.0}, {3.0, 1.5}, {5.0, 3.0}, {6.0, 0.0}
        } };

        pwlf::PwLinFunc<float, 5> k(seesaw);
        compare_with_tol(k(1), 1.5f);
        compare_with_tol(k(2.2), 1.9f);
        compare_with_tol(k(4), 2.25f);
        compare_with_tol(k(5.5), 1.5f);

        std::cout << "Seesaw segment done" << std::endl;
        // std::cout << "Checking out of range for seesaw segment" << std::endl;
        // auto outOfRange = k.ordinate(6.1);

    } catch (std::exception& ex) {
        std::cout << "Piece-wise linear function failed: " << typeid(ex).name() << std::endl;
        return -1;
    }
    return 0;
}
