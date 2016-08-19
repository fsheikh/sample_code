# A hurriedly put together make script

NASM := $(shell command -v nasm 2 < /dev/null)
ARM_GCC := $(shell command -v arm-none-eabi-gcc 2 < /dev/null)

all: helloworld.asm arm_first.s 
ifndef NASM
    $(error "NASM is not installed!")
endif
	nasm -f elf -l helloworld.lst helloworld.asm
	gcc -m32 -o hw helloworld.o
ifndef ARM_GCC
    $(error "ARM GNU Cross compiler not installed OR path not set!!")
endif
	arm-none-eabi-as -ggdb -o arm_first.o arm_first.s
	arm-none-eabi-ld -T linker.ld -o arm_first.elf arm_first.o
	arm-none-eabi-objcopy -O binary arm_first.elf arm_first.bin

clean:

	rm *.o *.lst *.elf *.bin



