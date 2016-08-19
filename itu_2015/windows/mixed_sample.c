#include <stdio.h>

/* AUTHOR: Faheem Sheikh
   Spring 2015 */


/* Requires Visual Studio Command Prompt (x86 Native Tools)
 CL /Famixed_sample.lst mixed_sample.c */

/* This function increments the value the integer pointed to by number_address
 by a value given in first argument count */
void inc_with_inline_asm(int count, int * number_address)
{

     _asm
    {
        push edi                    ; Store destination register
        mov  edi, number_address    ; Get the address of input number
        mov  eax, count             ; Get the increment value in accumulator
        add  eax, [edi]             ; Increment operation
        stosw                       ; Store result
        mov eax, 0                  ; Clear Accumulator
        pop edi                     ; Revert destinator register original value
    }

}
/* First x86 inline assembly program */
void main(void)
{

    int some_number = 100;
    printf("Number before increment %d\n", some_number);

  inc_with_inline_asm(5, &some_number);
  printf("The incremented value is %d\n", some_number);

}
