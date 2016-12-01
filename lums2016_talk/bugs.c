#include <stdio.h>
#include <stdint.h>

struct Device {
    uint8_t  reg1;
    uint32_t reg2;
};
typedef struct Device Dev;

enum DevOperations
{
    DEVICE_RESET,
    DEVICE_POWER_ON,
    DEVICE_POWER_OFF,
};

typedef enum DevOperations DevOps;
Dev Change_Device_State(Dev *device, DevOps OP)
{
    if (device != NULL && OP == DEVICE_RESET) 
    {
        device->reg1 |= 0x5;
    }
    return *device;
}
uint8_t wrong_shift_parameter(Dev *device, uint8_t shift_amount)
{
    //if (shift_amount > 7) { shift_amount = 0; }; // Put an assert here
    printf("Shift amount %d\r\n", shift_amount);
    return device->reg1 | (1 << shift_amount);
}
int main(int argc, char *argv[])
{
    Dev d;
    d.reg1 = 0x63;
    d.reg2 = 0;
    //int bad_shift = -1;
    uint8_t ret_val  = wrong_shift_parameter(&d, -1);
    printf("Before Shift %c, After Shift %c\r\n", d.reg1, ret_val);
    return 0;
}
