; Sample program to demonstrate usage of Windows Native APIs in assembly language program
; Requires Microsoft Visual Studio Command Prompt (Native Tools)
; Compile as: ml /Zi /Zd myprogam.asm -link -subsystem:console kernel32.li

.386
.MODEL small, stdcall

GetStdHandle PROTO NEAR32, device:dword

ExitProcess PROTO NEAR32, exitcode:dword

WriteConsoleA PROTO NEAR32, handle:dword,
                           buffer: ptr byte,
                           nChars:dword,
                           CharsWritten:ptr byte,
                           overlapped: ptr byte 
  .data
GETSTDOUT EQU -11
N         EQU  100


Block1   dw N DUP (12)
Block2   dw N DUP (13)
written  dd ?
comma    db 44
ascii_out db 4 DUP (?)
BITMASK_4 db 0FH

.code
main proc
    MOV AX, DS          ; Overlap DS and ES since source &
                        ; destination blocks are in same data segment

    MOV ES, AX
    CLD                     ; Select Auto-increment
    XOR ECX, ECX            ; Clear Counter
    MOV CX, N               ; Load Counter
    MOV ESI, OFFSET Block1  ; Address Block1
    MOV EDI, OFFSET Block2  ; Address Block2

    L1: LODSW              ; Load AX with element from block1
        ADD AX,[EDI]       ; Add element from block2
        STOSW              ; Save Answer
        LOOP L1            ; Repeat N times

    MOV EDX, offset ascii_out
    MOV BX, AX
    MOV CL, 16
    MOV ESI, OFFSET BITMASK_4
    MOV SI, [ESI]
    .WHILE CL > 0
    SUB CL, 4
    SHR AX, CL
    AND AX, SI
    ADD AL, 30H
    MOV BYTE PTR [EDX], AL
    INC EDX
    MOV AX, BX
    .ENDW

    SUB EDX, 4

    invoke GetStdHandle, GETSTDOUT         ; Grab output console device
    ; Write Contents at the start of Block2 to STDOUT

    invoke WriteConsoleA, eax, EDX, 4, offset written, 0
    invoke ExitProcess, 0
main endp
        end main

