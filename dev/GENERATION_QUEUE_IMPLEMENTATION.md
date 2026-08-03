# GenerationQueue Implementation

## Overview

The `GenerationQueue` feature has been successfully implemented in Flux2Kit. This allows job stacking so multiple generation requests can be queued and executed sequentially without manual intervention.

## Files Created

### `/workspace/Sources/Flux2Kit/GenerationQueue.swift`

A complete, production-ready work queue implementation with the following features:

#### Core Types

1. **`JobID`** - Unique identifier for tracking jobs
   - UUID-based with short description format
   - Hashable and Equatable for use in collections

2. **`JobPriority`** - Three-tier priority system
   - `.low`, `.normal`, `.high`
   - Jobs are sorted by priority in the queue

3. **`JobStatus`** - Complete lifecycle tracking
   - `.pending` - Waiting in queue
   - `.running(progress:)` - Currently executing with progress updates
   - `.completed(result:)` - Successfully finished
   - `.failed(error:)` - Encountered an error
   - `.cancelled` - User cancelled

4. **`QueuedGenerationJob`** - Job definition
   - Contains all generation parameters
   - Tracks creation time and priority
   - Supports input images for img2img/inpainting

5. **`JobResult`** - Execution result container
   - Result image or error
   - Timing information
   - Success/failure status

#### GenerationQueue Class

**Thread-Safe Operations:**
- All public methods are thread-safe using NSLock
- Background processing on detached Task
- Proper cancellation propagation

**Key Methods:**

```swift
// Enqueue a job with full options
func enqueue(
    options: GenerationOptions,
    inputImages: [CGImage]? = nil,
    priority: JobPriority = .normal
) -> JobID

// Simple convenience enqueue
func enqueue(
    prompt: String,
    width: Int = defaultWidth,
    height: Int = defaultHeight,
    numSteps: Int = defaultSteps,
    seed: UInt64? = nil,
    priority: JobPriority = .normal
) -> JobID

// Cancel specific job
func cancel(_ jobId: JobID) -> Bool

// Cancel all pending jobs
func cancelAllPending()

// Check job status
func status(of jobId: JobID) -> JobStatus

// Get pending count
var pendingCount: Int { get }

// Check if idle
var isIdle: Bool { get }
```

**Callbacks:**
- `onProgress: JobProgressHandler` - Progress updates with job ID
- `onCompletion: JobCompletionHandler` - Job completion notification
- `onStatusChange: JobStatusHandler` - Status change events

## Usage Examples

### Basic Queue Setup

```swift
import Flux2Kit

// Create pipeline
let pipeline = try await Flux2Pipeline(
    repoId: "black-forest-labs/FLUX.2-klein-4B"
)

// Create queue
let queue = GenerationQueue(pipeline: pipeline)

// Set up callbacks
queue.onProgress = { jobId, progress in
    print("Job \(jobId): Step \(progress.step)/\(progress.totalSteps)")
}

queue.onCompletion = { jobId, result in
    if result.isSuccess {
        print("Job \(jobId) completed successfully")
    } else {
        print("Job \(jobId) failed: \(result.error?.localizedDescription ?? "unknown")")
    }
}
```

### Enqueue Multiple Jobs

```swift
// Queue several generations
let job1 = queue.enqueue(prompt: "A cat sitting on a couch")
let job2 = queue.enqueue(prompt: "A dog playing in the park")
let job3 = queue.enqueue(
    prompt: "A bird flying over mountains",
    priority: .high  // High priority jumps ahead
)

print("Pending jobs: \(queue.pendingCount)")
```

### Monitor Progress

```swift
// Check status anytime
switch queue.status(of: job1) {
case .pending:
    print("Waiting to start...")
case .running(let progress):
    if let progress = progress {
        let percent = Double(progress.step) / Double(progress.totalSteps) * 100
        print("Processing: \(percent)%")
    }
case .completed(let image):
    print("Done! Got image")
case .failed(let error):
    print("Failed: \(error)")
case .cancelled:
    print("Cancelled")
}
```

### Cancel Jobs

```swift
// Cancel a specific job
if queue.cancel(job2) {
    print("Job cancelled")
}

// Cancel all pending
queue.cancelAllPending()
```

### Wait for Completion (Testing)

```swift
// In async context, wait for job to finish
do {
    let image = try await queue.wait(for: job1, timeout: 300)
    // Use the image
} catch {
    print("Job failed or timed out: \(error)")
}
```

## Architecture

### Thread Safety

- Uses `NSLock` for all shared state access
- `@unchecked Sendable` with careful lock discipline
- Background processing on detached Task
- Cancellation tokens propagate to generation engine

### Priority Scheduling

Jobs are inserted in priority order:
- Higher priority jobs jump ahead of lower priority ones
- Within same priority, FIFO ordering is maintained
- Currently running job cannot be preempted (runs to completion or cancellation)

### Memory Management

- History cleanup keeps only last N completed/failed jobs (default: 100)
- Automatic cache clearing after each job via pipeline's existing mechanisms
- Weak self references in closures prevent retain cycles

### Integration with Existing Code

The queue leverages existing Flux2Kit infrastructure:
- Reuses `GenerationOptions` for job parameters
- Uses `GenerationCancellation` for cancellation support
- Integrates with existing progress callback mechanism
- Respects the pipeline's `generationLock` for serialization
- No breaking changes to existing APIs

## Testing Recommendations

### Unit Tests

1. **Priority ordering** - Verify high priority jobs execute before low priority
2. **Cancellation** - Test cancelling pending and running jobs
3. **Status tracking** - Verify all status transitions
4. **Thread safety** - Concurrent enqueue/cancel operations
5. **Error handling** - Failed jobs don't block queue

### Integration Tests

1. **Multiple sequential jobs** - Queue 5+ jobs, verify all complete
2. **Mixed priorities** - Interleave different priority levels
3. **Cancellation during execution** - Cancel mid-generation
4. **Memory limits** - Run many jobs, verify history cleanup works

## Performance Considerations

1. **Serial execution** - Matches existing lock semantics, prevents MLX RNG conflicts
2. **Background processing** - Doesn't block calling thread
3. **Minimal overhead** - Lock contention only during enqueue/status checks
4. **Memory efficient** - History cleanup prevents unbounded growth

## Future Enhancements (Not Implemented)

These could be added in future iterations:

1. **Persistence** - Save queue to disk for app restart survival
2. **Retry logic** - Automatic retry with exponential backoff
3. **Batch operations** - Grid generation, variations
4. **Rate limiting** - Throttle based on thermal/memory pressure
5. **Dependencies** - Job B depends on Job A's output
6. **Parallel execution** - Multiple pipelines for true parallelism (requires separate processes)

## Migration Path

No migration needed - this is an additive feature:

1. Existing code continues to work unchanged
2. Opt-in by creating a `GenerationQueue` instance
3. Can mix direct pipeline calls and queued jobs (with serialization)

## Error Handling

- Failed jobs report errors via `JobResult.error`
- Errors don't stop the queue - next job proceeds automatically
- Cancellation observed at each denoise step via existing mechanism
- Timeout protection in `wait(for:timeout:)` method

## Compatibility

- Requires Swift 5.9+ (for `Task.detached`)
- Compatible with both FLUX.2 [klein] 4B and [dev] 9B models
- Works with all generation modes (txt2img, img2img, inpaint, outpaint)
- No changes to existing public APIs
