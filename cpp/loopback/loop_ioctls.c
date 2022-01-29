#include <errno.h>
#include <fcntl.h>
#include <linux/loop.h>
#include <sys/ioctl.h>
#include <stdio.h>
#include <unistd.h>

// Sample program using loop interface to find a free loopback device
// in the system and send a bunch of ioctls to the found device.
// Please see: https://man7.org/linux/man-pages/man4/loop.4.html
// Purpose of this tiny program is to demonstrate missing support of loop
// ioctls in valgrind

// Author: Faheem Sheikh <faheem.sheikh@bmw.de>
// Date: 28.01.2022

int main()
{

    fprintf(stdout, "Valgrind loopback test starting...\n");

    int loopControl = open("/dev/loop-control", O_RDWR);

    if (loopControl == -1) {
        fprintf(stderr, "Failed to open loop control interfacei with error=%d\n", errno);
        return 1;

    } else {
        long loopDevNum = ioctl(loopControl, LOOP_CTL_GET_FREE);

        if (loopDevNum == -1) {
            fprintf(stderr, "Failed to find a free loopback device\n");
            close(loopControl);
            return 1;

        } else {
            char loopName[256];
            snprintf(loopName, sizeof(loopName), "/dev/loop%ld", loopDevNum);
            fprintf(stdout, "Free loopback device %s found\n", loopName);

            // open the loopdevice in order to test it with a bunch of ioctls
            // we don't care about success/failure of the ioctls, just want to
            // check if valgrind is able to instrument the code include those
            // ioctls

            int fdLoop = open(loopName, O_RDWR);
            if (fdLoop == -1) {
                fprintf(stderr, "Failed to open loopback device %s with error=%d\n", loopName, errno);
                close(loopControl);
                return 1;

            } else {
                // Just a small sequence of ioctls for the loopback device
                ioctl(fdLoop, LOOP_CLR_FD);
                ioctl(fdLoop, LOOP_SET_BLOCK_SIZE, 4096);
                struct loop_info64 statusStruct;
                ioctl(fdLoop, LOOP_GET_STATUS64, &statusStruct);
            }
        }
    }

    fprintf(stdout, "Valgrind loopback test done!\n");
    return 0;
}