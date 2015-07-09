//
//  algorithm_test.h
//  
//
//  Created by Faheem Sheikh on 7/3/15.
//
//

#ifndef ____algorithm_test__
#define ____algorithm_test__

#include <stdio.h>
#include <stdlib.h>
#include <x86intrin.h>

int order_highest(int num)
{
    
    return (sizeof(num)*8 - 1) - __builtin_clz(num);
}

int order_lowest(int num)
{
    return __builtin_ctz(num);
}

int min_of_two(int a, int b)
{
    
    if (a<b)
        return a;
    else
        return b;
}

template <typename closure>
int for_aligned_regions(unsigned base, unsigned size, closure sfunc)
{
 
    unsigned order_size_max, subsize;
    
    while(size)
    {
        order_size_max = order_highest(size);
        unsigned order_base_min {base ? order_lowest(base): order_size_max};
        unsigned order_min {static_cast <unsigned int>(min_of_two(order_size_max,order_base_min))};
        
        sfunc(base, order_min);
        
        subsize = static_cast <unsigned>(1ul << order_min);
        
        base+= subsize;
        size-= subsize;
        
    }
    
    return 0;
}
#endif /* defined(____algorithm_test__) */
