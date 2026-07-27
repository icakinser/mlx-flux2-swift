import Foundation
import os.signpost

/// Instruments signposts under the `com.flux2kit.performance` subsystem. They are always cheap to
/// emit and become visible when recording the Points of Interest instrument in Instruments.
package enum Flux2Signpost {
    package static let log = OSLog(
        subsystem: "com.flux2kit.performance",
        category: .pointsOfInterest)

    package static func begin(_ name: StaticString) -> OSSignpostID {
        let id = OSSignpostID(log: log)
        os_signpost(.begin, log: log, name: name, signpostID: id)
        return id
    }

    package static func beginStep(_ step: Int) -> OSSignpostID {
        let id = OSSignpostID(log: log)
        os_signpost(
            .begin,
            log: log,
            name: "DenoiseStep",
            signpostID: id,
            "step=%{public}d",
            step)
        return id
    }

    package static func end(_ name: StaticString, _ id: OSSignpostID) {
        os_signpost(.end, log: log, name: name, signpostID: id)
    }

    package static func measure<T>(
        _ name: StaticString,
        _ body: () throws -> T
    ) rethrows -> T {
        let id = begin(name)
        defer { end(name, id) }
        return try body()
    }
}
