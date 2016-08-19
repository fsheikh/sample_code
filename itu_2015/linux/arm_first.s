@This ARM assembly scripts just loops for 100 iterations
@ and then stops in an infinite loop. Educational purpose only
@ for introducing students to ARM ISA.


@ Run with QEMU in debugging mode
@qemu-system-arm -m 128M -M versatilepb -nographic -s -S -kernel arm_first.bin

.data  
@ Nothing in this section

.text
       
.global _Reset
_Reset:
 
    MOV r0, #0
    MOV r1, #100
    MOV r3, #0

again: 
    ADD r0, r0, r1
    SUBS r1, r1, #1
    BNE again
stop:  B stop

MOV   r3, #100

