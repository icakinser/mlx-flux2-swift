# Job Stacking (Work Queue) Implementation Plan

## Executive Summary

This document outlines a phased implementation plan to add job stacking capabilities to Flux2Kit, allowing users to queue multiple generation requests that execute sequentially without manual intervention between jobs. This feature is essential for batch processing, iterative workflows, and production applications.

---

## Current State Analysis

### Existing Infrastructure
- **Generation Lock**: `NSRecursiveLock` serializes all generation calls (line 112, Flux2Pipeline.swift)
- **Cancellation Support**: `GenerationCancellation` class provides thread-safe cancellation (GenerationEngine.swift:6-19)
- **Progress Callbacks**: All generation methods support `@Sendable (GenerationProgress) -> Void` callbacks
- **Memory Management**: Residency policies (`.keepResident`, `.unloadAfterUse`) manage GPU memory
- **Error Handling**: Comprehensive error types in `Flux2Error` enum

### Limitations
- No queue management - each call blocks until completion
- No job status tracking beyond progress callbacks
- No job prioritization or reordering
- No automatic retry on failure
- No job persistence across app restarts

---

## Phase 1: Core Queue Infrastructure (Foundation)

### Objectives
- Create thread-safe job queue data structures
- Define job states and lifecycle
- Implement basic enqueue/dequeue operations
- Add job identification and tracking

### Deliverables

#### 1.1 Job Model Definition
**File**: `Sources/Flux2Kit/GenerationQueue.swift`

```swift
/// Unique identifier for queued generation jobs
public struct JobID: Hashable, Sendable, CustomStringConvertible {
    public let uuid: UUID
    public var description: String { uuid.uuidString }
}

/// Represents a single generation request in the queue
public struct QueuedGenerationJob: Sendable {
    public let id: JobID
    public let mode: GenerationMode  // .textToImage, .img2img, .inpaint, .outpaint
    public let parameters: GenerationParameters  // Enum containing mode-specific params
    public let priority: JobPriority
    public let createdAt: Date
    public var status: JobStatus
    public var result: JobResult?
    public var error: Error?
    public var progress: Double  // 0.0 to 1.0
    public var startTime: Date?
    public var completionTime: Date?
}

/// Generation mode determines which pipeline method to call
public enum GenerationMode: Sendable {
    case textToImage(GenerationOptions)
    case img2img(ImageToImageOptions, source: CGImage)
    case inpaint(InpaintOptions, source: CGImage, mask: CGImage)
    case outpaint(OutpaintOptions, source: CGImage)
}

/// Job priority for queue ordering
public enum JobPriority: Int, Comparable, Sendable {
    case low = 0
    case normal = 1
    case high = 2
    case critical = 3
    
    public static func < (lhs: JobPriority, rhs: JobPriority) -> Bool {
        lhs.rawValue < rhs.rawValue
    }
}

/// Current state of a queued job
public enum JobStatus: Sendable {
    case pending
    case running(progress: Double)
    case completed
    case cancelled
    case failed(Error)
}

/// Result of a completed job
public enum JobResult: Sendable {
    case image(CGImage)
    case images([CGImage])
}
```

#### 1.2 Queue Manager Class
**File**: `Sources/Flux2Kit/GenerationQueue.swift`

```swift
/// Thread-safe generation job queue with priority support
public final class GenerationQueue: @unchecked Sendable {
    private let queueLock = NSLock()
    private var jobs: [QueuedGenerationJob] = []
    private var isProcessing = false
    private var currentJobID: JobID?
    
    /// Observers for job state changes
    private var statusObservers: [(JobID, JobStatus) -> Void] = []
    
    public init() {}
    
    /// Add a job to the queue
    public func enqueue(
        mode: GenerationMode,
        priority: JobPriority = .normal
    ) -> JobID
    
    /// Cancel a pending or running job
    public func cancel(_ jobID: JobID) throws
    
    /// Get current status of a job
    public func status(for jobID: JobID) -> JobStatus?
    
    /// Get all pending jobs
    public func pendingJobs() -> [QueuedGenerationJob]
    
    /// Clear all pending jobs
    public func clearPending()
    
    /// Register observer for status changes
    public func observeStatusChanges(_ handler: @escaping (JobID, JobStatus) -> Void)
}
```

#### 1.3 Integration with Flux2Pipeline
**File**: `Sources/Flux2Kit/Flux2Pipeline.swift`

Add queue property and processing loop:
```swift
public final class Flux2Pipeline: @unchecked Sendable {
    // ... existing properties ...
    
    /// Optional job queue for batch processing
    public let queue: GenerationQueue?
    
    /// Background dispatch queue for job processing
    private let jobProcessingQueue = DispatchQueue(label: "com.flux2kit.jobProcessor")
    
    /// Start processing queued jobs
    public func startQueueProcessing()
    
    /// Stop processing after current job completes
    public func stopQueueProcessing()
    
    /// Check if queue is actively processing
    public var isProcessingQueue: Bool { get }
}
```

### Testing Requirements
- Unit tests for thread safety (concurrent enqueue/cancel)
- Priority ordering verification
- Status transition correctness
- Memory leak detection with observers

---

## Phase 2: Job Execution Engine

### Objectives
- Implement background job processor
- Handle job lifecycle (pending → running → completed/failed)
- Integrate with existing progress callbacks
- Support graceful cancellation

### Deliverables

#### 2.1 Job Processor
**File**: `Sources/Flux2Kit/GenerationQueue.swift`

```swift
extension GenerationQueue {
    /// Internal worker that processes jobs sequentially
    private func processNextJob(with pipeline: Flux2Pipeline) async throws {
        guard let job = dequeueNextJob() else { return }
        
        updateJobStatus(job.id, .running(progress: 0.0))
        
        do {
            let result = try await executeJob(job, with: pipeline)
            updateJobStatus(job.id, .completed)
            markJobComplete(job.id, result: result)
        } catch let error as Flux2Error {
            updateJobStatus(job.id, .failed(error))
            markJobComplete(job.id, error: error)
        } catch {
            updateJobStatus(job.id, .failed(Flux2Error.generationFailed(error.localizedDescription)))
            markJobComplete(job.id, error: error)
        }
    }
    
    /// Execute a single job based on its mode
    private func executeJob(
        _ job: QueuedGenerationJob,
        with pipeline: Flux2Pipeline
    ) async throws -> JobResult {
        let cancellation = GenerationCancellation()
        job.cancellationHandle = cancellation
        
        var lastProgress: Double = 0
        let progressHandler: @Sendable (GenerationProgress) -> Void = { progress in
            let newProgress = Double(progress.step) / Double(progress.totalSteps)
            if newProgress > lastProgress {
                lastProgress = newProgress
                self.updateJobProgress(job.id, newProgress)
            }
        }
        
        switch job.mode {
        case .textToImage(let options):
            var modifiedOptions = options
            modifiedOptions.progress = progressHandler
            modifiedOptions.cancellation = cancellation
            let image = try pipeline.generate(modifiedOptions)
            return .image(image)
            
        case .img2img(let options, let source):
            var modifiedOptions = options
            modifiedOptions.denoise.progress = progressHandler
            modifiedOptions.denoise.cancellation = cancellation
            let image = try pipeline.generateImg2Img(modifiedOptions, source: source)
            return .image(image)
            
        // ... other modes ...
        }
    }
}
```

#### 2.2 Pipeline Integration
**File**: `Sources/Flux2Kit/Flux2Pipeline.swift`

```swift
extension Flux2Pipeline {
    public func startQueueProcessing() {
        guard let queue = queue else {
            fatalError("Queue not initialized")
        }
        
        jobProcessingQueue.async { [weak self] in
            Task {
                while self?.queue?.hasPendingJobs == true {
                    do {
                        try await queue.processNextJob(with: self!)
                    } catch {
                        self?.handleJobError(error)
                    }
                }
            }
        }
    }
}
```

### Testing Requirements
- End-to-end job execution tests
- Progress callback verification
- Cancellation during execution
- Error handling and recovery
- Concurrent queue access patterns

---

## Phase 3: Advanced Features

### Objectives
- Job persistence (survive app restarts)
- Retry logic for transient failures
- Batch operations
- Performance optimizations

### Deliverables

#### 3.1 Job Persistence
**File**: `Sources/Flux2Kit/GenerationQueue.swift`

```swift
/// Configuration for persistent queues
public struct QueuePersistenceConfiguration: Sendable {
    public let storageURL: URL
    public let persistCompletedJobs: Bool
    public let maxHistorySize: Int
    public let autosaveInterval: TimeInterval
    
    public static var `default`: QueuePersistenceConfiguration {
        .init(
            storageURL: FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
                .appendingPathComponent("GenerationQueue"),
            persistCompletedJobs: false,
            maxHistorySize: 100,
            autosaveInterval: 5.0
        )
    }
}

extension GenerationQueue {
    /// Initialize with persistence support
    public convenience init(persistence: QueuePersistenceConfiguration) {
        self.init()
        self.persistenceConfig = persistence
        try? loadPersistedJobs()
        startAutosaveTimer()
    }
    
    private func persistJobs() throws {
        // Serialize pending jobs to disk
    }
    
    private func loadPersistedJobs() throws {
        // Restore jobs from disk
    }
}
```

#### 3.2 Retry Logic
```swift
public struct RetryPolicy: Sendable {
    public let maxRetries: Int
    public let delayBetweenRetries: TimeInterval
    public let exponentialBackoff: Bool
    public let retryableErrors: [Flux2ErrorType]
    
    public static var `default`: RetryPolicy {
        .init(
            maxRetries: 3,
            delayBetweenRetries: 2.0,
            exponentialBackoff: true,
            retryableErrors: [.memoryPressure, .modelNotLoaded]
        )
    }
}

extension QueuedGenerationJob {
    public var retryCount: Int { get }
    public var canRetry: Bool { get }
}

extension GenerationQueue {
    public func retryJob(_ jobID: JobID) throws
    public func setRetryPolicy(_ policy: RetryPolicy)
}
```

#### 3.3 Batch Operations
```swift
extension Flux2Pipeline {
    /// Enqueue multiple variations of the same prompt
    public func enqueueBatch(
        prompt: String,
        variations: Int,
        seedRange: Range<UInt64>?,
        priority: JobPriority = .normal
    ) -> [JobID]
    
    /// Enqueue grid generation (multiple prompts × multiple seeds)
    public func enqueueGrid(
        prompts: [String],
        seeds: [UInt64],
        options: GenerationOptions
    ) -> [JobID]
}
```

### Testing Requirements
- Persistence round-trip tests
- Retry behavior under various error conditions
- Batch operation performance
- Disk space management

---

## Phase 4: API & Developer Experience

### Objectives
- Swift-friendly async/await APIs
- Combine publishers for reactive UIs
- Documentation and examples
- Performance monitoring

### Deliverables

#### 4.1 Async/Await API
```swift
extension Flux2Pipeline {
    /// Wait for job completion asynchronously
    public func wait(for jobID: JobID) async throws -> CGImage
    
    /// Stream results as jobs complete
    public func resultsStream() -> AsyncThrowingStream<(JobID, JobResult), Error>
}

// Usage example:
let pipeline = try await Flux2Pipeline(configuration: config)
let jobID = pipeline.queue.enqueue(mode: .textToImage(options))

do {
    let image = try await pipeline.wait(for: jobID)
    // Use image
} catch {
    // Handle error
}
```

#### 4.2 Combine Publishers (iOS/macOS)
```swift
import Combine

extension GenerationQueue {
    /// Publisher for job status changes
    public var statusPublisher: AnyPublisher<(JobID, JobStatus), Never>
    
    /// Publisher for completed jobs
    public var completedJobsPublisher: AnyPublisher<(JobID, JobResult), Never>
}
```

#### 4.3 Performance Monitoring
```swift
public struct QueueMetrics: Sendable {
    public let totalJobsEnqueued: Int
    public let totalJobsCompleted: Int
    public let totalJobsFailed: Int
    public let averageWaitTime: TimeInterval
    public let averageExecutionTime: TimeInterval
    public let currentQueueDepth: Int
    public let estimatedTimeToCompletion: TimeInterval?
}

extension GenerationQueue {
    public var metrics: QueueMetrics { get }
    public func resetMetrics()
}
```

### Testing Requirements
- API usability testing
- Documentation completeness
- Example project validation
- Performance benchmarking

---

## Phase 5: Optimization & Production Readiness

### Objectives
- Memory optimization for large queues
- Multi-instance coordination
- Logging and diagnostics
- Edge case handling

### Deliverables

#### 5.1 Memory Optimization
```swift
public struct QueueMemoryPolicy: Sendable {
    public let maxResidentImages: Int
    public let autoPurgeCompletedResults: Bool
    public let purgeDelay: TimeInterval
    
    public static var conservative: QueueMemoryPolicy {
        .init(maxResidentImages: 1, autoPurgeCompletedResults: true, purgeDelay: 0)
    }
    
    public static var balanced: QueueMemoryPolicy {
        .init(maxResidentImages: 5, autoPurgeCompletedResults: true, purgeDelay: 60)
    }
}
```

#### 5.2 Diagnostics
```swift
public struct QueueDiagnosticReport: Sendable {
    public let timestamp: Date
    public let queueDepth: Int
    public let memoryUsage: Int
    public let recentFailures: [(JobID, Error, Date)]
    public let performanceMetrics: QueueMetrics
    public let warnings: [String]
}

extension GenerationQueue {
    public func generateDiagnosticReport() -> QueueDiagnosticReport
}
```

#### 5.3 Logging
- Integration with OSLog
- Configurable log levels
- Structured logging for analytics

### Testing Requirements
- Stress testing with 100+ jobs
- Memory profiling
- Long-running stability tests
- Multi-process coordination tests

---

## Implementation Timeline

| Phase | Estimated Duration | Dependencies | Risk Level |
|-------|-------------------|--------------|------------|
| Phase 1 | 2-3 weeks | None | Low |
| Phase 2 | 2-3 weeks | Phase 1 | Medium |
| Phase 3 | 3-4 weeks | Phase 2 | Medium |
| Phase 4 | 2-3 weeks | Phase 2 | Low |
| Phase 5 | 2-3 weeks | Phase 3-4 | Medium |

**Total Estimated Duration**: 11-16 weeks

---

## Risk Mitigation

### Technical Risks
1. **Thread Safety Issues**
   - Mitigation: Extensive concurrency testing, use of TSan
   - Fallback: Serial queue for all operations

2. **Memory Pressure**
   - Mitigation: Aggressive purging policies, residency integration
   - Fallback: Queue size limits

3. **MLX Compatibility**
   - Mitigation: Isolate MLX interactions, version testing
   - Fallback: Document minimum MLX version

### Design Decisions

1. **Queue Location**: Embedded in `Flux2Pipeline` vs standalone
   - Decision: Optional property on pipeline for tight integration
   - Rationale: Shares model loading, memory management

2. **Persistence Strategy**: Core Data vs JSON vs SQLite
   - Decision: JSON for simplicity, SQLite optional for large-scale
   - Rationale: Jobs are simple structures, JSON is sufficient

3. **Execution Model**: Serial vs parallel
   - Decision: Serial execution (matches current lock semantics)
   - Rationale: MLX global RNG, shared model state

---

## Migration Path

### For Existing Users
1. Queue is opt-in (nullable property)
2. Existing APIs unchanged
3. Gradual rollout with feature flags

### Breaking Changes
- None planned for initial release
- Future: Potential API cleanup in major version

---

## Success Metrics

1. **Functional**: Can queue 100+ jobs without failure
2. **Performance**: <1ms enqueue latency, <10ms status lookup
3. **Reliability**: 99.9% job completion rate
4. **Memory**: <100MB overhead for queue management
5. **Adoption**: Used by 80% of example applications

---

## Appendix: Example Usage

### Basic Queue Usage
```swift
// Initialize pipeline with queue
var config = PipelineConfiguration.lowMemory()
let pipeline = try await Flux2Pipeline(configuration: config)
pipeline.queue = GenerationQueue()

// Enqueue jobs
let options1 = GenerationOptions(prompt: "A sunset over mountains", width: 512, height: 512)
let options2 = GenerationOptions(prompt: "A forest stream", width: 512, height: 512)
let options3 = GenerationOptions(prompt: "Ocean waves", width: 512, height: 512)

let job1 = pipeline.queue!.enqueue(mode: .textToImage(options1), priority: .normal)
let job2 = pipeline.queue!.enqueue(mode: .textToImage(options2), priority: .normal)
let job3 = pipeline.queue!.enqueue(mode: .textToImage(options3), priority: .high)

// Start processing
pipeline.startQueueProcessing()

// Monitor progress
pipeline.queue!.observeStatusChanges { jobID, status in
    switch status {
    case .running(let progress):
        print("Job \(jobID): \(Int(progress * 100))%")
    case .completed:
        print("Job \(jobID) completed!")
    case .failed(let error):
        print("Job \(jobID) failed: \(error)")
    default:
        break
    }
}

// Wait for specific job
let image = try await pipeline.wait(for: job3)
```

### Batch Processing
```swift
// Generate variations
let jobIDs = pipeline.enqueueBatch(
    prompt: "A cyberpunk cityscape",
    variations: 10,
    seedRange: 0..<1000
)

// Process all and save
for try await (jobID, result) in pipeline.resultsStream() {
    if case .image(let image) = result {
        try saveImage(image, to: outputDirectory.appendingPathComponent("\(jobID).png"))
    }
}
```

---

## Next Steps

1. **Immediate**: Begin Phase 1 implementation
2. **Week 1**: Complete job model and basic queue
3. **Week 2**: Add priority support and observers
4. **Week 3**: Integration tests and documentation
5. **Review**: Assess progress, adjust timeline if needed

This plan provides a structured approach to adding robust job stacking capabilities while maintaining compatibility with existing Flux2Kit functionality.
