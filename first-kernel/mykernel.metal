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
    // So metal has gid which is a simplified indexing scheme.
    // If we don't need to know which block etc we are in, it's cleaner.
    int v=blockid*blockDim+threadid;
    int i = gid;
    if (v!=i){
        outBuf[i]=inBuf[i] + 100000;
    }else {
        // will always go here
        outBuf[i]=inBuf[i] + 10;
    }
}


// Matrix multiplication: C = A × B where A is (m×k), B is (k×n), C is (m×n)
// All matrices are row-major
kernel void filip_matmul(
    device const float *A[[buffer(0)]],
    device const float *B[[buffer(1)]],
    device float *C[[buffer(2)]],
    constant int& m[[buffer(3)]],        // A dim0
    constant int& k[[buffer(4)]],        // A dim1 / B dim0
    constant int& n[[buffer(5)]],        // B dim1
    uint2 gid [[thread_position_in_grid]]
){
    // Each thread computes one element C[i][j]
    int i = gid.y;  // row index
    int j = gid.x;  // column index
    
    if (i >= m || j >= n) return;
    
    float acc = 0.0f;
    for (int kp = 0; kp < k; kp++) {
        // A[i][kp] = A[i*k + kp]
        // B[kp][j] = B[kp*n + j]
        acc += A[i*k + kp] * B[kp*n + j];
    }
    
    // C[i][j] = C[i*n + j]
    C[i*n + j] = acc;
}
