// Flux2Kit — package-scoped, testable parsing helpers for the CLI. They throw `CLIParseError` on malformed input
// instead of silently falling back to a default, so a typo in a numeric flag is a hard error rather
// than a surprising run with the wrong value.

import Foundation

/// Errors produced by the CLI argument parsers. Carries the offending flag/value for clear messages.
package enum CLIParseError: Error, Equatable, CustomStringConvertible {
    case invalidInt(flag: String, value: String)
    case invalidDouble(flag: String, value: String)
    case invalidUInt(flag: String, value: String)
    case invalidDimensions(String)
    case invalidTuple(flag: String, value: String, expected: Int)

    package var description: String {
        switch self {
        case let .invalidInt(flag, value):
            return "invalid integer for \(flag): '\(value)'"
        case let .invalidDouble(flag, value):
            return "invalid number for \(flag): '\(value)'"
        case let .invalidUInt(flag, value):
            return "invalid unsigned integer for \(flag): '\(value)'"
        case let .invalidDimensions(value):
            return "expected WxH (e.g. 512x768), got: '\(value)'"
        case let .invalidTuple(flag, value, expected):
            return "\(flag) expects \(expected) comma-separated integers, got: '\(value)'"
        }
    }
}

/// Parse a required integer flag value, throwing on anything non-numeric.
package func parseIntArg(_ flag: String, _ value: String) throws -> Int {
    guard let n = Int(value.trimmingCharacters(in: .whitespaces)) else {
        throw CLIParseError.invalidInt(flag: flag, value: value)
    }
    return n
}

/// Parse a required floating-point flag value, throwing on anything non-numeric.
package func parseDoubleArg(_ flag: String, _ value: String) throws -> Double {
    guard let d = Double(value.trimmingCharacters(in: .whitespaces)) else {
        throw CLIParseError.invalidDouble(flag: flag, value: value)
    }
    return d
}

/// Parse a required `Float` flag value, throwing on anything non-numeric.
package func parseFloatArg(_ flag: String, _ value: String) throws -> Float {
    guard let f = Float(value.trimmingCharacters(in: .whitespaces)) else {
        throw CLIParseError.invalidDouble(flag: flag, value: value)
    }
    return f
}

/// Parse a required unsigned integer flag value (e.g. a seed), throwing on anything invalid.
package func parseUInt64Arg(_ flag: String, _ value: String) throws -> UInt64 {
    guard let n = UInt64(value.trimmingCharacters(in: .whitespaces)) else {
        throw CLIParseError.invalidUInt(flag: flag, value: value)
    }
    return n
}

/// Parse "WxH" (e.g. "512x768") into two ints; throws on a malformed spec.
package func parseWxHArg(_ s: String) throws -> (Int, Int) {
    let p = s.lowercased().split(separator: "x").map {
        Int($0.trimmingCharacters(in: .whitespaces))
    }
    guard p.count == 2, let w = p[0], let h = p[1] else {
        throw CLIParseError.invalidDimensions(s)
    }
    return (w, h)
}

/// Parse exactly four comma-separated ints (e.g. crop / mask box "x,y,w,h").
package func parse4Arg(_ flag: String, _ s: String) throws -> (Int, Int, Int, Int) {
    let p = s.split(separator: ",").map { Int($0.trimmingCharacters(in: .whitespaces)) }
    guard p.count == 4, let a = p[0], let b = p[1], let c = p[2], let d = p[3] else {
        throw CLIParseError.invalidTuple(flag: flag, value: s, expected: 4)
    }
    return (a, b, c, d)
}

/// Parse outpaint margins: "L,R,T,B" or a single value applied to all sides.
package func parseOutpaintArg(_ s: String) throws -> (Int, Int, Int, Int) {
    let raw = s.split(separator: ",").map { Int($0.trimmingCharacters(in: .whitespaces)) }
    if raw.count == 1, let v = raw[0] { return (v, v, v, v) }
    guard raw.count == 4, let l = raw[0], let r = raw[1], let t = raw[2], let b = raw[3] else {
        throw CLIParseError.invalidTuple(flag: "--outpaint", value: s, expected: 4)
    }
    return (l, r, t, b)
}

/// The five recolor components plus any warnings (unknown keys / unparseable values) so the caller
/// can surface them instead of silently ignoring a typo like `hue=0.2`.
package struct RecolorSpec: Equatable {
    package var hue: Float
    package var sat: Float
    package var exp: Float
    package var contrast: Float
    package var gamma: Float
    package var warnings: [String]
}

/// Parse a `--recolor` spec like "hue=0.2,sat=1.1,exp=0.3,contrast=1.1,gamma=1.0". Unknown keys and
/// unparseable values are collected into `warnings` rather than silently dropped.
package func parseRecolorArg(_ s: String) -> RecolorSpec {
    var spec = RecolorSpec(hue: 0, sat: 1, exp: 0, contrast: 1, gamma: 1, warnings: [])
    for part in s.split(separator: ",") {
        let kv = part.split(separator: "=", maxSplits: 1)
        guard kv.count == 2 else {
            spec.warnings.append("ignored malformed --recolor term '\(part)' (expected key=value)")
            continue
        }
        let key = kv[0].trimmingCharacters(in: .whitespaces).lowercased()
        let rawVal = kv[1].trimmingCharacters(in: .whitespaces)
        guard let val = Float(rawVal) else {
            spec.warnings.append("ignored --recolor '\(key)': '\(rawVal)' is not a number")
            continue
        }
        switch key {
        case "hue": spec.hue = val
        case "sat", "saturation": spec.sat = val
        case "exp", "exposure": spec.exp = val
        case "contrast": spec.contrast = val
        case "gamma": spec.gamma = val
        default:
            spec.warnings.append("ignored unknown --recolor key '\(key)'")
        }
    }
    return spec
}
