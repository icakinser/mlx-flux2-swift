//
//  GenerationQueue.swift
//  Flux2Kit
//
//  Thread-safe work queue for sequential image generation jobs.
//  Allows job stacking so multiple generation requests can be queued
//  and executed one after another without manual intervention.
//

import Foundation
import CoreGraphics
import MLX

/// Unique identifier for a queued generation job
public struct JobID: Hashable, Equatable, CustomStringConvertible, Sendable {
    public let uuid: UUID
    
    public init(uuid: UUID = UUID()) {
        self.uuid = uuid
    }
    
    public var description: String {
        uuid.uuidString.prefix(8).description
    }
}

/// Priority level for job scheduling
public enum JobPriority: Int, Comparable, Sendable {
    case low = 0
    case normal = 1
    case high = 2
    
    public static func < (lhs: JobPriority, rhs: JobPriority) -> Bool {
        lhs.rawValue < rhs.rawValue
    }
}

/// Current status of a queued job
public enum JobStatus: Sendable {
    case pending
    case running(progress: GenerationProgress?)
    case completed(result: CGImage)
    case failed(error: Error)
    case cancelled
    
    public var isTerminal: Bool {
        switch self {
        case .completed, .failed, .cancelled:
            return true
        case .pending, .running:
            return false
        }
    }
}

/// A single generation job in the queue
public struct QueuedGenerationJob: Sendable {
    public let id: JobID
    public let options: GenerationOptions
    public let inputImages: [CGImage]?
    public let priority: JobPriority
    public let createdAt: Date
    
    // Computed properties for convenience
    public var status: JobStatus {
        // Status is tracked externally by GenerationQueue
        return .pending
    }
    
    public init(
        id: JobID = JobID(),
        options: GenerationOptions,
        inputImages: [CGImage]? = nil,
        priority: JobPriority = .normal
    ) {
        self.id = id
        self.options = options
        self.inputImages = inputImages
        self.priority = priority
        self.createdAt = Date()
    }
}

/// Result of a completed job
public struct JobResult: Sendable {
    public let jobId: JobID
    public let result: CGImage?
    public let error: Error?
    public let executionTime: TimeInterval
    public let startedAt: Date
    public let completedAt: Date
    
    public var isSuccess: Bool {
        error == nil && result != nil
    }
}

/// Progress callback with job context
public typealias JobProgressHandler = @Sendable (JobID, GenerationProgress) -> Void

/// Completion callback for when a job finishes
public typealias JobCompletionHandler = @Sendable (JobID, JobResult) -> Void

/// Status observer for UI updates
public typealias JobStatusHandler = @Sendable (JobID, JobStatus) -> Void

/// Thread-safe queue for managing sequential generation jobs
public final class GenerationQueue: @unchecked Sendable {
    private let lock = NSLock()
    private var pendingJobs: [QueuedGenerationJob] = []
    private var jobStatuses: [JobID: JobStatus] = [:]
    private var isProcessing = false
    private var runningJobId: JobID?
    private var runningCancellation: GenerationCancellation?
    /// Terminal job IDs in the order they finished, so history trimming drops the oldest first.
    private var terminalOrder: [JobID] = []
    
    /// Maximum number of jobs to keep in history (completed/failed)
    public var maxHistorySize: Int = 100
    
    /// Callbacks for job events
    public var onProgress: JobProgressHandler?
    public var onCompletion: JobCompletionHandler?
    public var onStatusChange: JobStatusHandler?
    
    /// The underlying pipeline used for generation
    private let pipeline: Flux2Pipeline
    
    /// Background task for processing jobs
    private var processingTask: Task<Void, Never>?
    
    public init(pipeline: Flux2Pipeline) {
        self.pipeline = pipeline
    }
    
    deinit {
        processingTask?.cancel()
    }
    
    // MARK: - Public API
    
    /// Enqueue a new generation job
    /// - Parameters:
    ///   - options: Generation options (prompt, dimensions, steps, etc.)
    ///   - inputImages: Optional input images for img2img/inpainting
    ///   - priority: Job priority (default: .normal)
    /// - Returns: The JobID for tracking/cancellation
    @discardableResult
    public func enqueue(
        options: GenerationOptions,
        inputImages: [CGImage]? = nil,
        priority: JobPriority = .normal
    ) -> JobID {
        let job = QueuedGenerationJob(
            options: options,
            inputImages: inputImages,
            priority: priority
        )
        
        lock.withLock {
            // Insert based on priority (higher priority first)
            let insertIndex = pendingJobs.firstIndex { $0.priority < priority } 
                ?? pendingJobs.count
            pendingJobs.insert(job, at: insertIndex)
            jobStatuses[job.id] = .pending
            
            // Start processing if not already running
            if !isProcessing {
                startProcessing()
            }
        }
        
        notifyStatusChange(job.id, .pending)
        return job.id
    }
    
    /// Cancel a pending or running job
    /// - Parameter jobId: The job to cancel
    /// - Returns: True if the job was found and cancelled
    @discardableResult
    public func cancel(_ jobId: JobID) -> Bool {
        enum Outcome { case running, removedPending, notFound }
        let outcome: Outcome = lock.withLock {
            // Running job: signal its cancellation token; the next denoise step observes it and
            // executeJob records the .cancelled status.
            if runningJobId == jobId {
                runningCancellation?.cancel()
                return .running
            }
            if let index = pendingJobs.firstIndex(where: { $0.id == jobId }) {
                pendingJobs.remove(at: index)
                setTerminalLocked(jobId, .cancelled)
                return .removedPending
            }
            return .notFound
        }
        switch outcome {
        case .running:
            return true
        case .removedPending:
            // Callbacks run outside the lock so a handler may safely call back into the queue.
            notifyStatusChange(jobId, .cancelled)
            return true
        case .notFound:
            return false
        }
    }
    
    /// Cancel all pending jobs
    public func cancelAllPending() {
        let cancelled: [JobID] = lock.withLock {
            let ids = pendingJobs.map(\.id)
            for id in ids { setTerminalLocked(id, .cancelled) }
            pendingJobs.removeAll()
            return ids
        }
        // Callbacks run outside the lock so a handler may safely call back into the queue.
        for id in cancelled { notifyStatusChange(id, .cancelled) }
    }
    
    /// Get the current status of a job
    /// - Parameter jobId: The job to check
    /// - Returns: Current job status
    public func status(of jobId: JobID) -> JobStatus {
        lock.withLock {
            return jobStatuses[jobId] ?? .cancelled
        }
    }
    
    /// Get all pending jobs
    public var pendingJobsList: [QueuedGenerationJob] {
        lock.withLock {
            return Array(pendingJobs)
        }
    }
    
    /// Get the currently running job ID (if any)
    public var currentJobId: JobID? {
        lock.withLock {
            return runningJobId
        }
    }
    
    /// Check if the queue is idle (no pending or running jobs)
    public var isIdle: Bool {
        lock.withLock {
            return pendingJobs.isEmpty && !isProcessing
        }
    }
    
    /// Number of pending jobs
    public var pendingCount: Int {
        lock.withLock {
            return pendingJobs.count
        }
    }
    
    // MARK: - Internal Processing
    
    private func startProcessing() {
        guard !isProcessing else { return }
        
        isProcessing = true
        processingTask = Task.detached { [weak self] in
            await self?.processQueue()
        }
    }
    
    private func processQueue() async {
        while let next = dequeueNextJob() {
            let (job, jobCancellation) = next
            notifyStatusChange(job.id, .running(progress: nil))
            await executeJob(job, cancellation: jobCancellation)
            lock.withLock {
                runningJobId = nil
                runningCancellation = nil
            }
        }
    }
    
    /// Pops the next job, or clears `isProcessing` in the same critical section when the queue is
    /// empty, so an `enqueue` racing with the drain can never be stranded.
    private func dequeueNextJob() -> (QueuedGenerationJob, GenerationCancellation)? {
        lock.withLock {
            guard !pendingJobs.isEmpty else {
                isProcessing = false
                runningJobId = nil
                runningCancellation = nil
                return nil
            }
            let job = pendingJobs.removeFirst()
            let jobCancellation = GenerationCancellation()
            runningJobId = job.id
            runningCancellation = jobCancellation
            jobStatuses[job.id] = .running(progress: nil)
            return (job, jobCancellation)
        }
    }
    
    private func executeJob(_ job: QueuedGenerationJob, cancellation jobCancellation: GenerationCancellation) async {
        let startTime = Date()
        
        do {
            // Check if cancelled before starting
            if Task.isCancelled || jobCancellation.isCancelled {
                throw Flux2Error.cancelled
            }
            
            // Set up progress handler that includes job ID
            let progressHandler: (@Sendable (GenerationProgress) -> Void) = { [weak self] progress in
                guard let self = self else { return }
                self.lock.withLock {
                    self.jobStatuses[job.id] = .running(progress: progress)
                }
                self.onProgress?(job.id, progress)
            }
            
            // Update options with our progress handler and cancellation
            var options = job.options
            options.progress = progressHandler
            options.cancellation = jobCancellation
            
            // Run generation (blocking call on background thread)
            let result = try pipeline.generate(options, inputImages: job.inputImages)
            
            let executionTime = Date().timeIntervalSince(startTime)
            let jobResult = JobResult(
                jobId: job.id,
                result: result,
                error: nil,
                executionTime: executionTime,
                startedAt: startTime,
                completedAt: Date()
            )
            
            lock.withLock {
                setTerminalLocked(job.id, .completed(result: result))
            }
            
            onCompletion?(job.id, jobResult)
            notifyStatusChange(job.id, .completed(result: result))
            
            // Clean up old history
            cleanupHistory()
            
        } catch {
            let executionTime = Date().timeIntervalSince(startTime)
            let jobResult = JobResult(
                jobId: job.id,
                result: nil,
                error: error,
                executionTime: executionTime,
                startedAt: startTime,
                completedAt: Date()
            )
            
            lock.withLock {
                if jobCancellation.isCancelled {
                    setTerminalLocked(job.id, .cancelled)
                } else {
                    setTerminalLocked(job.id, .failed(error: error))
                }
            }
            
            if jobCancellation.isCancelled {
                notifyStatusChange(job.id, .cancelled)
            } else {
                onCompletion?(job.id, jobResult)
                notifyStatusChange(job.id, .failed(error: error))
            }
            
            // Clean up old history
            cleanupHistory()
        }
    }
    
    private func notifyStatusChange(_ jobId: JobID, _ status: JobStatus) {
        onStatusChange?(jobId, status)
    }
    
    /// Record a terminal status and its finish order. Caller must hold `lock`.
    private func setTerminalLocked(_ jobId: JobID, _ status: JobStatus) {
        jobStatuses[jobId] = status
        terminalOrder.append(jobId)
    }

    /// Drop the oldest terminal jobs beyond `maxHistorySize` (oldest = finished first).
    private func cleanupHistory() {
        lock.withLock {
            let overflow = terminalOrder.count - max(0, maxHistorySize)
            guard overflow > 0 else { return }
            for jobId in terminalOrder.prefix(overflow) {
                jobStatuses.removeValue(forKey: jobId)
            }
            terminalOrder.removeFirst(overflow)
        }
    }
}

// MARK: - Convenience Extensions

extension GenerationQueue {
    /// Enqueue a simple text-to-image job with minimal parameters
    @discardableResult
    public func enqueue(
        prompt: String,
        width: Int = defaultWidth,
        height: Int = defaultHeight,
        numSteps: Int = defaultSteps,
        seed: UInt64? = nil,
        priority: JobPriority = .normal
    ) -> JobID {
        let options = GenerationOptions(
            prompt: prompt,
            width: width,
            height: height,
            numSteps: numSteps,
            seed: seed
        )
        return enqueue(options: options, priority: priority)
    }
    
    /// Wait for a job to complete (for testing/synchronous contexts)
    public func wait(for jobId: JobID, timeout: TimeInterval = 300) async throws -> CGImage {
        let startTime = Date()
        
        while Date().timeIntervalSince(startTime) < timeout {
            try Task.checkCancellation()
            
            switch status(of: jobId) {
            case .completed(let result):
                return result
            case .failed(let error):
                throw error
            case .cancelled:
                throw Flux2Error.cancelled
            case .pending, .running:
                try await Task.sleep(nanoseconds: 100_000_000) // 100ms
            }
        }
        
        throw Flux2Error.generationFailed("Job timed out")
    }
}
