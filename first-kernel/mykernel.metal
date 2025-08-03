//
//  mykernel.metal
//  first-kernel
//
//  Created by Filip Skogh on 30.07.2025.
//

#include <metal_stdlib>
using namespace metal;


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


// Matrix multiplication: C = A × B where A is (m×k), B is (k×n), C is (m×n)
// A: row-major, B: col-major, C: row-major
kernel void filip_matmul_trans(
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
        // B[kp][j] = B[j*k + kp] (B is transposed)
        acc += A[i*k + kp] * B[j*k + kp];
    }
    // C[i][j] = C[i*n + j]
    C[i*n + j] = acc;
}

// It turns out this is slower. Even though at the thread level it's obvisouly better since
// both A and B is read with stride 1.
// TODO: think more about this at the threadgroup level



#define TILE_SIZE 16

kernel void filip_matmul_o1(
    device const float *A[[buffer(0)]],
    device const float *B[[buffer(1)]],
    device float *C[[buffer(2)]],
    constant int& m[[buffer(3)]],        // A dim0
    constant int& k[[buffer(4)]],        // A dim1 / B dim0
    constant int& n[[buffer(5)]],        // B dim1
    uint2 gid [[thread_position_in_grid]],
    uint2 lid [[thread_position_in_threadgroup]],
    uint2 group_id [[threadgroup_position_in_grid]]
) {
    // Threadgroup memory for caching tiles
    threadgroup float tileA[TILE_SIZE][TILE_SIZE];
    threadgroup float tileB[TILE_SIZE][TILE_SIZE];
    
    // Calculate global indices
    int globalRow = group_id.y * TILE_SIZE + lid.y;
    int globalCol = group_id.x * TILE_SIZE + lid.x;
    
    float acc = 0.0f;
    
    // Loop over tiles along the k dimension
    int numTiles = (k + TILE_SIZE - 1) / TILE_SIZE;
    
    for (int t = 0; t < numTiles; t++) {
        int tileK = t * TILE_SIZE + lid.x;
        int tileKRow = t * TILE_SIZE + lid.y;
        
        // load tiles of A and B into threadgroup memory
        if (globalRow < m && tileK < k) {
            tileA[lid.y][lid.x] = A[globalRow * k + tileK];
        } else {
            tileA[lid.y][lid.x] = 0.0f;
        }
        if (tileKRow < k && globalCol < n) {
            tileB[lid.y][lid.x] = B[tileKRow * n + globalCol];
        } else {
            tileB[lid.y][lid.x] = 0.0f;
        }
        
        // Synchronize to ensure all threads have loaded their data
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        // Compute partial dot product using cached tile data
        for (int kk = 0; kk < TILE_SIZE; kk++) {
            acc += tileA[lid.y][kk] * tileB[kk][lid.x];
        }
        
        // Synchronize before loading next tile
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    if (globalRow < m && globalCol < n) {
        C[globalRow * n + globalCol] = acc;
    }
}
