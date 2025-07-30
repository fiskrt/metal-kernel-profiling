# Metal GPU Profiling

## GPU Frame Capture
```objc
MTLCaptureManager *captureManager = [MTLCaptureManager sharedCaptureManager];
MTLCaptureDescriptor *captureDescriptor = [[MTLCaptureDescriptor alloc] init];
captureDescriptor.captureObject = device;
captureDescriptor.destination = MTLCaptureDestinationDeveloperTools;
[captureManager startCaptureWithDescriptor:captureDescriptor error:&error];
// ... GPU work ...
[captureManager stopCapture];
```

## Command Line Profiling
Set environment variable:
```bash
MTL_CAPTURE_ENABLED=1
```

## View Results
- Open Xcode → Window → Developer Tools → GPU Frame Debugger
- Analyze GPU timeline, memory usage, and kernel performance
- View shader assembly and register usage
