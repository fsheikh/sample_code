#include <iostream>

class Device
{

    static constexpr size_t N {10};
    static constexpr size_t M {4};
    uint8_t  reg8[10];
    uint32_t regs[4];

public:
    Device()
    {
        for (size_t index = 0; index < N; index++) {
            reg8[index] = 0;
        }
        for (size_t index = 0; index < M; index++) {
            regs[index] = 0;
        }
    }
    void Change_Device_State()
    {

    }

};


int main(int argc, char *argv[])
{
    Device dev;
    Device * p = new Device();
    p->Change_Device_State();
}
