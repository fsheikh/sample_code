/* Taken from https://balau82.wordpress.com/2010/02/14/simplest-bare-metal-program-for-arm/ */
/* No License infringement intended, educational puprose only */

ENTRY(_Reset)
SECTIONS
{
	. = 0x10000;
    .arm_first . : {arm_first.o(.text)}
	.text : {*(.text)}
    .data : {*(.data)}
    .bss : {*(.bss COMMON)}
    . = ALIGN(8);
    . = . + 0x10000;
    stack_top = .;
}
