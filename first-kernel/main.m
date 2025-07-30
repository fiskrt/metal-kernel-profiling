//
//  main.m
//  first-kernel
//
//  Created by Filip Skogh on 30.07.2025.
//

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <simd/simd.h>


void run2kernels() {
    NSLog(@"Hello, World!!");
        id<MTLDevice> device=MTLCreateSystemDefaultDevice();
        id<MTLLibrary> defaultLibrary=[device newDefaultLibrary];
        id<MTLCommandQueue> commandQueue=[device newCommandQueue];
        
        int bufferSize = 32;
        
        id<MTLBuffer> inBuf=[device newBufferWithLength:bufferSize*sizeof(uint) options:MTLResourceStorageModeShared];
        id<MTLBuffer> outBuf=[device newBufferWithLength:bufferSize*sizeof(uint) options:MTLResourceStorageModeShared];
        
        
        //load buf with simple data
        uint *dataIn=(uint*)inBuf.contents;
        for (int i=0; i<bufferSize; i++) {
            dataIn[i]=i;
        }
        // Get the shared capture manager
        MTLCaptureManager *captureManager = [MTLCaptureManager sharedCaptureManager];
        
        // Create a capture descriptor
        MTLCaptureDescriptor *captureDescriptor = [[MTLCaptureDescriptor alloc] init];
        captureDescriptor.captureObject = device; // or commandQueue
        captureDescriptor.destination = MTLCaptureDestinationDeveloperTools;
        
        // Start capturing
        NSError *error = nil;
        if (![captureManager startCaptureWithDescriptor:captureDescriptor error:&error]) {
            NSLog(@"Failed to start capture: %@", error);
            return;
        }
        
        // Load both kernel functions
        id<MTLFunction> firstKernelFunc = [defaultLibrary newFunctionWithName:@"filips_first"];
        id<MTLFunction> secondKernelFunc = [defaultLibrary newFunctionWithName:@"filip_second"];
        
        // Create pipeline state objects for both kernels
        id<MTLComputePipelineState> firstKernelPSO = [device newComputePipelineStateWithFunction:firstKernelFunc error:&error];
        id<MTLComputePipelineState> secondKernelPSO = [device newComputePipelineStateWithFunction:secondKernelFunc error:&error];
        
        // Create an intermediate buffer for passing data between kernels
        id<MTLBuffer> intermediateBuf = [device newBufferWithLength:bufferSize * sizeof(uint) options:MTLResourceStorageModeShared];
        
        // Create command buffer
        id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
        
        // First kernel: inBuf -> intermediateBuf (squares the values)
        id<MTLComputeCommandEncoder> computeEncoder1 = [commandBuffer computeCommandEncoder];
        [computeEncoder1 setComputePipelineState:firstKernelPSO];
        [computeEncoder1 setBuffer:inBuf offset:0 atIndex:0];
        [computeEncoder1 setBuffer:intermediateBuf offset:0 atIndex:1];
        
        NSUInteger width1 = firstKernelPSO.threadExecutionWidth;
        MTLSize threadsPerThreadgroup1 = MTLSizeMake(width1, 1, 1);
        MTLSize threadsPerGrid = MTLSizeMake(bufferSize, 1, 1);
        
        [computeEncoder1 dispatchThreads:threadsPerGrid threadsPerThreadgroup:threadsPerThreadgroup1];
        [computeEncoder1 endEncoding];
        
        // Second kernel: intermediateBuf -> outBuf (adds 10)
        id<MTLComputeCommandEncoder> computeEncoder2 = [commandBuffer computeCommandEncoder];
        [computeEncoder2 setComputePipelineState:secondKernelPSO];
        [computeEncoder2 setBuffer:intermediateBuf offset:0 atIndex:0];
        [computeEncoder2 setBuffer:outBuf offset:0 atIndex:1];
        
        NSUInteger width2 = secondKernelPSO.threadExecutionWidth;
        MTLSize threadsPerThreadgroup2 = MTLSizeMake(width2, 1, 1);
        
        [computeEncoder2 dispatchThreads:threadsPerGrid threadsPerThreadgroup:threadsPerThreadgroup2];
        [computeEncoder2 endEncoding];
        
        // Commit and wait
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];
        
        // Stop capturing
        [captureManager stopCapture];
        
        // Print results
        uint *inBufData = (uint*)inBuf.contents;
        uint *outBufData = (uint*)outBuf.contents;
        NSLog(@"Results (first 10 elements):");
        for (int i = 0; i < MIN(10, bufferSize); i++) {
            NSLog(@"Input: %u -> Output: %u (should be %u² + 10 = %u)",
                  inBufData[i],
                  outBufData[i],
                  inBufData[i],
                  inBufData[i] * inBufData[i] + 10);
        }
}

void run1kernel() {
         // insert code here...
        NSLog(@"Hello, World!");
        id<MTLDevice> device=MTLCreateSystemDefaultDevice();
        id<MTLLibrary> defaultLibrary=[device newDefaultLibrary];
        id<MTLFunction> kernel=[defaultLibrary newFunctionWithName:@"filips_first"];
        id<MTLComputePipelineState> kernelPSO=[device newComputePipelineStateWithFunction:kernel error:nil];
        id<MTLCommandQueue> commandQueue=[device newCommandQueue];
        
        int bufferSize = 32;
        
        id<MTLBuffer> inBuf=[device newBufferWithLength:bufferSize*sizeof(uint) options:MTLResourceStorageModeShared];
        id<MTLBuffer> outBuf=[device newBufferWithLength:bufferSize*sizeof(uint) options:MTLResourceStorageModeShared];


        //load buf with simple data
        uint *dataIn=(uint*)inBuf.contents;
        for (int i=0; i<bufferSize; i++) {
            dataIn[i]=i;
        }
        
        // launching the kernel
        
        // Get the shared capture manager
        MTLCaptureManager *captureManager = [MTLCaptureManager sharedCaptureManager];

        // Create a capture descriptor
        MTLCaptureDescriptor *captureDescriptor = [[MTLCaptureDescriptor alloc] init];
        captureDescriptor.captureObject = device; // or commandQueue
        captureDescriptor.destination = MTLCaptureDestinationDeveloperTools; // or MTLCaptureDestinationGPUTraceDocument

        // Start capturing
        NSError *error = nil;
        if (![captureManager startCaptureWithDescriptor:captureDescriptor error:&error]) {
            NSLog(@"Failed to start capture: %@", error);
            return;
        }

        // Your existing code
        id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
        id<MTLComputeCommandEncoder> computeEncoder = [commandBuffer computeCommandEncoder];

        // Encode the pipeline state object
        [computeEncoder setComputePipelineState:kernelPSO];
        [computeEncoder setBuffer:inBuf offset:0 atIndex:0];
        [computeEncoder setBuffer:outBuf offset:0 atIndex:1];

        NSUInteger width = kernelPSO.threadExecutionWidth;
        MTLSize threadsPerThreadgroup = MTLSizeMake(width, 1, 1);
        MTLSize threadsPerGrid = MTLSizeMake(bufferSize, 1, 1);

        [computeEncoder dispatchThreads:threadsPerGrid threadsPerThreadgroup:threadsPerThreadgroup];
        [computeEncoder endEncoding];
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];
        [NSThread sleepForTimeInterval:0.5];  // 0.5 seconds = 500 ms

        // Stop capturing
        [captureManager stopCapture];

        // Print results
        uint *outBufData = (uint*)outBuf.contents;
        for (int i = 0; i < bufferSize; i++) {
            NSLog(@"%i", outBufData[i]);
        }
}

// Buffer allocation strategy: Three buffers are used to enable kernel chaining:
// - inBuf: Input data (persistent across iterations)
// - intermediateBuf: Output from first kernel, input to second kernel
// - outBuf: Final output from second kernel chain
// 
// Capture manager wraps the entire loop for GPU debugging in Xcode.
// Each iteration creates a new command buffer for proper GPU synchronization.
// Pipeline state objects are created once and reused for performance.
void runloopkernel(int n) {
    NSLog(@"Running kernels in loop %d times", n);
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    id<MTLLibrary> defaultLibrary = [device newDefaultLibrary];
    id<MTLCommandQueue> commandQueue = [device newCommandQueue];
    
    int bufferSize = 32;
    
    id<MTLBuffer> inBuf = [device newBufferWithLength:bufferSize*sizeof(uint) options:MTLResourceStorageModeShared];
    id<MTLBuffer> outBuf = [device newBufferWithLength:bufferSize*sizeof(uint) options:MTLResourceStorageModeShared];
    
    uint *dataIn = (uint*)inBuf.contents;
    for (int i = 0; i < bufferSize; i++) {
        dataIn[i] = i;
    }
    
    MTLCaptureManager *captureManager = [MTLCaptureManager sharedCaptureManager];
    MTLCaptureDescriptor *captureDescriptor = [[MTLCaptureDescriptor alloc] init];
    captureDescriptor.captureObject = device;
    captureDescriptor.destination = MTLCaptureDestinationDeveloperTools;
    
    NSError *error = nil;
    if (![captureManager startCaptureWithDescriptor:captureDescriptor error:&error]) {
        NSLog(@"Failed to start capture: %@", error);
        return;
    }
    
    id<MTLFunction> firstKernelFunc = [defaultLibrary newFunctionWithName:@"filips_first"];
    id<MTLFunction> secondKernelFunc = [defaultLibrary newFunctionWithName:@"filip_second"];
    
    id<MTLComputePipelineState> firstKernelPSO = [device newComputePipelineStateWithFunction:firstKernelFunc error:&error];
    id<MTLComputePipelineState> secondKernelPSO = [device newComputePipelineStateWithFunction:secondKernelFunc error:&error];
    
    id<MTLBuffer> intermediateBuf = [device newBufferWithLength:bufferSize * sizeof(uint) options:MTLResourceStorageModeShared];
    
    for (int iteration = 0; iteration < n; iteration++) {
        NSLog(@"Loop iteration: %d", iteration + 1);
        
        id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
        
        id<MTLComputeCommandEncoder> computeEncoder1 = [commandBuffer computeCommandEncoder];
        [computeEncoder1 setComputePipelineState:firstKernelPSO];
        [computeEncoder1 setBuffer:inBuf offset:0 atIndex:0];
        [computeEncoder1 setBuffer:intermediateBuf offset:0 atIndex:1];
        
        NSUInteger width1 = firstKernelPSO.threadExecutionWidth;
        MTLSize threadsPerThreadgroup1 = MTLSizeMake(width1, 1, 1);
        MTLSize threadsPerGrid = MTLSizeMake(bufferSize, 1, 1);
        
        [computeEncoder1 dispatchThreads:threadsPerGrid threadsPerThreadgroup:threadsPerThreadgroup1];
        [computeEncoder1 endEncoding];
        
        id<MTLComputeCommandEncoder> computeEncoder2 = [commandBuffer computeCommandEncoder];
        [computeEncoder2 setComputePipelineState:secondKernelPSO];
        [computeEncoder2 setBuffer:intermediateBuf offset:0 atIndex:0];
        [computeEncoder2 setBuffer:outBuf offset:0 atIndex:1];
        
        NSUInteger width2 = secondKernelPSO.threadExecutionWidth;
        MTLSize threadsPerThreadgroup2 = MTLSizeMake(width2, 1, 1);
        
        [computeEncoder2 dispatchThreads:threadsPerGrid threadsPerThreadgroup:threadsPerThreadgroup2];
        [computeEncoder2 endEncoding];
        
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];
        
        if (iteration == n - 1) {
            uint *inBufData = (uint*)inBuf.contents;
            uint *outBufData = (uint*)outBuf.contents;
            NSLog(@"Final results (first 10 elements):");
            for (int i = 0; i < MIN(10, bufferSize); i++) {
                NSLog(@"Input: %u -> Output: %u (should be %u² + 10 = %u)",
                      inBufData[i],
                      outBufData[i],
                      inBufData[i],
                      inBufData[i] * inBufData[i] + 10);
            }
        }
    }
    
    [captureManager stopCapture];
}

// Single compute encoder version - more efficient as it batches both kernel dispatches
// in one encoder session, reducing GPU command overhead
// Multi encoder uses n cmd buffers, and 2n compute encoders
// This uses n compute encoders instead!
void runloopkernel_single_encoder(int n) {
    NSLog(@"Running kernels in loop %d times (single encoder)", n);
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    id<MTLLibrary> defaultLibrary = [device newDefaultLibrary];
    id<MTLCommandQueue> commandQueue = [device newCommandQueue];
    
    // 100MB buffer = 100 * 1024 * 1024 bytes / 4 bytes per uint = 26,214,400 elements
    // Thread grid sizing: GPU threads are organized in threadgroups (blocks)
    // - threadsPerThreadgroup: Hardware-optimal size (typically 32-1024 threads)
    // - threadsPerGrid: Total work items (26M elements in this case)
    // GPU automatically distributes threadgroups across compute units for parallel execution
    int bufferSize = 26214400; // 100MB of uint data
    
    id<MTLBuffer> inBuf = [device newBufferWithLength:bufferSize*sizeof(uint) options:MTLResourceStorageModeShared];
    id<MTLBuffer> outBuf = [device newBufferWithLength:bufferSize*sizeof(uint) options:MTLResourceStorageModeShared];
    
    uint *dataIn = (uint*)inBuf.contents;
    for (int i = 0; i < bufferSize; i++) {
        dataIn[i] = i;
    }
    
    MTLCaptureManager *captureManager = [MTLCaptureManager sharedCaptureManager];
    MTLCaptureDescriptor *captureDescriptor = [[MTLCaptureDescriptor alloc] init];
    captureDescriptor.captureObject = device;
    captureDescriptor.destination = MTLCaptureDestinationDeveloperTools;
    
    NSError *error = nil;
    if (![captureManager startCaptureWithDescriptor:captureDescriptor error:&error]) {
        NSLog(@"Failed to start capture: %@", error);
        return;
    }
    
    id<MTLFunction> firstKernelFunc = [defaultLibrary newFunctionWithName:@"filips_first"];
    id<MTLFunction> secondKernelFunc = [defaultLibrary newFunctionWithName:@"filip_second"];
    
    id<MTLComputePipelineState> firstKernelPSO = [device newComputePipelineStateWithFunction:firstKernelFunc error:&error];
    id<MTLComputePipelineState> secondKernelPSO = [device newComputePipelineStateWithFunction:secondKernelFunc error:&error];
    
    id<MTLBuffer> intermediateBuf = [device newBufferWithLength:bufferSize * sizeof(uint) options:MTLResourceStorageModeShared];
    
    for (int iteration = 0; iteration < n; iteration++) {
        NSLog(@"Loop iteration: %d", iteration + 1);
        
        id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
        id<MTLComputeCommandEncoder> computeEncoder = [commandBuffer computeCommandEncoder];
        
        [computeEncoder setComputePipelineState:firstKernelPSO];
        [computeEncoder setBuffer:inBuf offset:0 atIndex:0];
        [computeEncoder setBuffer:intermediateBuf offset:0 atIndex:1];
        
        NSUInteger width1 = firstKernelPSO.threadExecutionWidth;
        MTLSize threadsPerThreadgroup1 = MTLSizeMake(width1, 1, 1);
        MTLSize threadsPerGrid = MTLSizeMake(bufferSize, 1, 1);
        
        [computeEncoder dispatchThreads:threadsPerGrid threadsPerThreadgroup:threadsPerThreadgroup1];
        
        [computeEncoder setComputePipelineState:secondKernelPSO];
        [computeEncoder setBuffer:intermediateBuf offset:0 atIndex:0];
        [computeEncoder setBuffer:outBuf offset:0 atIndex:1];
        
        NSUInteger width2 = secondKernelPSO.threadExecutionWidth;
        MTLSize threadsPerThreadgroup2 = MTLSizeMake(width2, 1, 1);
        
        [computeEncoder dispatchThreads:threadsPerGrid threadsPerThreadgroup:threadsPerThreadgroup2];
        [computeEncoder endEncoding];
        
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];
        
        if (iteration == n - 1) {
            uint *inBufData = (uint*)inBuf.contents;
            uint *outBufData = (uint*)outBuf.contents;
            NSLog(@"Final results (first 10 elements):");
            for (int i = 0; i < MIN(10, bufferSize); i++) {
                NSLog(@"Input: %u -> Output: %u (should be %u² + 10 = %u)",
                      inBufData[i],
                      outBufData[i],
                      inBufData[i],
                      inBufData[i] * inBufData[i] + 10);
            }
        }
    }
    
    [captureManager stopCapture];
}

int main(int argc, const char * argv[]) {
    @autoreleasepool {
        runloopkernel_single_encoder(5);
    }
    return 0;
}
