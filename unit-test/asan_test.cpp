#include <stdio.h>
#include <cstdlib>
#include <iostream>
#include <thread>
#include <functional>

class ClassUnderTest {

    int  *i_ptr = NULL;
    std::thread *t_ptr = NULL;

public:
    size_t i_size;

    void thread_func(void)
    {
        std::cout << "Hello from Thread " << std::endl;
    }

    ClassUnderTest(size_t s1)
    {
        i_size = s1;
    }

    void validate_ptr()
    {
        i_ptr = new int[i_size];
        t_ptr = new std::thread(&ClassUnderTest::thread_func, this);
    }
 
   ~ClassUnderTest()
    {
        /* Intentionally left empty */
        delete []i_ptr;
        delete t_ptr;
    }
    
};

int main(int argc, char *argv[])
{
    std::cout << "Instantiate an object" << std::endl;
    ClassUnderTest cut(4);
    //ClassUnderTest *cut = new ClassUnderTest(4);
    std::cout << "Allocating Memory " << std::endl;
    cut.validate_ptr();
    //cut->validate_ptr();
    std::cout << "Exiting without freeing" << std::endl;
}
