//
//  algorithm_test.cpp
//  
//
//  Created by Faheem Sheikh on 7/3/15.
//
//

#include "algorithm_test.h"
#include <iostream>

class AlgorithmTest
{
    
public:
    AlgorithmTest()
    {
        
    }
    void first_test()
    {
        
        
#if 0
        unsigned base;
        unsigned size;
        std::cout << "Base ?";
        std::cin >> base;
        std::cout << std::endl;
        std::cout << "Size ?";
        std::cin >> size;
        std::cout << std::endl;
#endif
        
        std::pair <unsigned, unsigned> input_pair(17,6);
        unsigned constexpr TOTAL_OUTPUT = 4;
        std::pair <unsigned, unsigned> output_pairs[TOTAL_OUTPUT];
        output_pairs[0] = std::pair<unsigned, unsigned>::pair(17,0);
        output_pairs[1] = std::pair<unsigned, unsigned>::pair(18,1);
        output_pairs[2] = std::pair<unsigned, unsigned>::pair(20,1);
        output_pairs[2] = std::pair<unsigned, unsigned>::pair(22,0);
        
        unsigned index_count = 0;
        unsigned success = 0;
        
        std::cout << "Base: " << input_pair.first << " Size: " << input_pair.second << std::endl;
        
        auto sfunc = [&index_count, &output_pairs, &success] (unsigned x, unsigned y) {
            
                                                              std::cout << "x: " << x << " y: " << y << std::endl;
                                                              if ((x != output_pairs[index_count].first) || (y != output_pairs[index_count].second))
                                                                success = 1ul << index_count;
                                                                index_count++;
                                                              return success;
                                                            };
        
        auto ret_val = for_aligned_regions(input_pair.first, input_pair.second, sfunc);
        
        if ((success != 0) || (index_count != TOTAL_OUTPUT))
            std::cout << "TEST FAILED at index: " << index_count << std::endl;
        else
            std::cout << "TEST PASSED " << std::endl;
        
    }
    
    
};


int main(int argc, char **argv)
{
#if 0
    std::cout << "MSB in 5:" << order_highest(5) << std::endl;
    std::cout << "LSB in 5:" << order_lowest(5) << std::endl;
    std::cout << "MSB in 17:" << order_highest(17) << std::endl;
    std::cout << "LSB in 17:" << order_lowest(17) << std::endl;
#endif
    
    AlgorithmTest *a = new AlgorithmTest();
    a->first_test();
    
    return 0;
    
}