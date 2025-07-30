//
//  mykernel.metal
//  first-kernel
//
//  Created by Filip Skogh on 30.07.2025.
//



kernel void filips_first(
    device uint *inBuf[[buffer(0)]],
    device uint *outBuf[[buffer(1)]],
    uint gid        [[thread_position_in_grid]],
    uint threadid   [[thread_position_in_threadgroup]],
    uint blockDim   [[threads_per_threadgroup]],
    uint blockid    [[threadgroup_position_in_grid]]
){

    int i=blockid*blockDim+threadid;
    outBuf[i]=inBuf[i] * inBuf[i];
}


kernel void filip_second(
    device uint *inBuf[[buffer(0)]],
    device uint *outBuf[[buffer(1)]],
    uint gid        [[thread_position_in_grid]],
    uint threadid   [[thread_position_in_threadgroup]],
    uint blockDim   [[threads_per_threadgroup]],
    uint blockid    [[threadgroup_position_in_grid]]
){

    int i=blockid*blockDim+threadid;
    outBuf[i]=inBuf[i] + 10;
}
