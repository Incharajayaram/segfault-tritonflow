; --- Lowered Layer 1 Assembly (8x32 @ 32x64 on TRITONFLOW1) ---
00: MOVI value=31
01: MOVI value=0.0
02: MOVI value=1
03: MOVI value=0
04: MOVI value=32
05: EPI_ADD in0=%K in1=%c31_i32
06: EPI_DIV in0=%0 in1=%c32_i32
07: DMA1D src=global:%a_ptr[base=%a_ptr;sizes=[64, 32];strides=[%sam, %sak];offsets=[64*%sam*pid_x, 0];shape=[0, 0];order=[0, 1];dtype=f32;loop_carried=True;increment=32]
08: DMA1D src=global:%b_ptr[base=%b_ptr;sizes=[32, 64];strides=[%sbk, %sbn];offsets=[64*%sbn*pid_y, 0];shape=[0, 0];order=[0, 1];dtype=f32;loop_carried=True;increment=32]
09: MAC16 a=%a acc=%acc_36 b=%b
10: DMA1D dst=global:%c_ptr[base=%c_ptr;sizes=[64, 64];strides=[%scm, %scn];offsets=[64*%scm*pid_x + 64*%scn*pid_y, 0];shape=[0, 0];order=[0, 1];dtype=f32;loop_carried=False;increment=None] value=%acc_25#2