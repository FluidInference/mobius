// Iter3Bench: load each iteration_3 .mlmodelc with the placement
// recommended by `_STAGE_COMPUTE` in coreml/inference.py, run a couple
// of warm predictions on synthesised inputs (shape resolved from the
// model description itself), and report per-stage timing.
//
// Build & run:
//   cd iteration_3/swift
//   swift build -c release
//   .build/release/iter3-bench --compiled ../compiled
//
// To produce real audio, swap the synthetic inputs for tensors captured
// from `coreml.inference` (or write the eager-glue stages — alignment
// matmul, asr-shift, s/ref split — in Swift). This binary is intended
// as a scaffolding sanity check that all 8 stages load and predict in
// the placement matrix the Python pipeline recommends.

import CoreML
import Foundation

// MARK: - Stage manifest

struct StageSpec {
    let name: String
    let modelFile: String
    let computeUnits: MLComputeUnits
}

let MANIFEST: [StageSpec] = [
    StageSpec(name: "text_encoder",
              modelFile: "text_encoder_fp16.mlmodelc",
              computeUnits: .cpuOnly),
    StageSpec(name: "bert",
              modelFile: "bert_fp16.mlmodelc",
              computeUnits: .all),
    StageSpec(name: "ref_encoder",
              modelFile: "ref_encoder_fp16.mlmodelc",
              computeUnits: .cpuAndGPU),
    StageSpec(name: "fused_diffusion_sampler",
              modelFile: "fused_diffusion_sampler_fp16.mlmodelc",
              computeUnits: .all),
    StageSpec(name: "duration_predictor",
              modelFile: "duration_predictor_fp16.mlmodelc",
              computeUnits: .cpuOnly),
    StageSpec(name: "fused_f0n_har_source",
              modelFile: "fused_f0n_har_source.mlmodelc",
              computeUnits: .cpuOnly),
    StageSpec(name: "decoder_pre",
              modelFile: "decoder_pre_fp16.mlmodelc",
              computeUnits: .cpuAndNeuralEngine),
    StageSpec(name: "decoder_upsample",
              modelFile: "decoder_upsample_fp16.mlmodelc",
              computeUnits: .cpuOnly),
]

// MARK: - Helpers

enum BenchError: Error {
    case unresolvedShape(String)
    case unsupportedFeatureType(String)
    case missingInput(String)
}

extension MLComputeUnits {
    var label: String {
        switch self {
        case .cpuOnly: return "CPU_ONLY"
        case .cpuAndGPU: return "CPU_AND_GPU"
        case .cpuAndNeuralEngine: return "CPU_AND_NE"
        case .all: return "ALL"
        @unknown default: return "?"
        }
    }
}

/// Resolve a possibly-flexible NSArray<NSNumber> shape to a concrete
/// `[Int]`. RangeDim placeholders sometimes show up as `0` on the
/// description; we substitute a representative default so the Swift
/// caller can build a tensor of that size.
func concreteShape(
    inputName: String,
    constraint: MLMultiArrayConstraint
) throws -> [Int] {
    let raw = constraint.shape.map { $0.intValue }
    var resolved: [Int] = []
    for (axis, dim) in raw.enumerated() {
        if dim > 0 {
            resolved.append(dim)
        } else {
            // Try the enumerated/range constraint to get a concrete pick.
            let shapeConstraint = constraint.shapeConstraint
            switch shapeConstraint.type {
            case .enumerated:
                guard let candidate = shapeConstraint.enumeratedShapes.first else {
                    throw BenchError.unresolvedShape(
                        "\(inputName): empty enumerated shape on axis \(axis)")
                }
                resolved.append(candidate[axis].intValue)
            case .range:
                let ranges = shapeConstraint.sizeRangeForDimension
                let r = ranges[axis].rangeValue
                let lower = r.location
                let upper = r.location + r.length
                // Pick a representative size (147 frames is the sample
                // used in the Python conversion script).
                let repValue = max(lower, min(upper, 147))
                resolved.append(repValue)
            case .unspecified:
                throw BenchError.unresolvedShape(
                    "\(inputName): unspecified shape on axis \(axis)")
            @unknown default:
                throw BenchError.unresolvedShape(
                    "\(inputName): unknown shape constraint")
            }
        }
    }
    return resolved
}

/// Build a synthetic MLMultiArray matching a constraint, filled with
/// modest pseudo-random values. Integer-typed inputs are filled with
/// 0..<vocabSize indices; floating types with N(0, 1).
func makeArray(
    inputName: String,
    constraint: MLMultiArrayConstraint
) throws -> MLMultiArray {
    let shape = try concreteShape(inputName: inputName, constraint: constraint)
    let nsShape = shape.map { NSNumber(value: $0) }
    let array = try MLMultiArray(shape: nsShape, dataType: constraint.dataType)
    let count = array.count
    switch constraint.dataType {
    case .float32:
        let ptr = array.dataPointer.bindMemory(to: Float.self, capacity: count)
        for i in 0..<count {
            // Box–Muller to keep dependencies zero
            let u1 = Float.random(in: 1e-7..<1)
            let u2 = Float.random(in: 0..<1)
            ptr[i] = (-2 * log(u1)).squareRoot() * cos(2 * .pi * u2)
        }
    case .float16:
        // Pack via Float32 → Float16 bit-cast (UInt16 storage)
        let ptr = array.dataPointer.bindMemory(to: UInt16.self, capacity: count)
        for i in 0..<count {
            let u1 = Float.random(in: 1e-7..<1)
            let u2 = Float.random(in: 0..<1)
            let f32 = (-2 * log(u1)).squareRoot() * cos(2 * .pi * u2)
            ptr[i] = float16Bits(f32)
        }
    case .double:
        let ptr = array.dataPointer.bindMemory(to: Double.self, capacity: count)
        for i in 0..<count {
            ptr[i] = Double.random(in: -1...1)
        }
    case .int32:
        let ptr = array.dataPointer.bindMemory(to: Int32.self, capacity: count)
        for i in 0..<count {
            // Phoneme vocab in StyleTTS2 has 178 symbols. Cap to that
            // for inputs that look like token IDs.
            ptr[i] = Int32.random(in: 0..<178)
        }
    default:
        throw BenchError.unsupportedFeatureType("dataType=\(constraint.dataType)")
    }
    return array
}

/// Round-trip Float32 → IEEE-754 binary16 (half).
func float16Bits(_ value: Float) -> UInt16 {
    let bits = value.bitPattern
    let sign = UInt16((bits >> 31) & 0x1) << 15
    let exp = Int((bits >> 23) & 0xff) - 127 + 15
    let frac = bits & 0x7fffff
    if exp <= 0 {
        return sign
    }
    if exp >= 0x1f {
        return sign | (0x1f << 10)
    }
    return sign | (UInt16(exp) << 10) | UInt16(frac >> 13)
}

func buildInputs(model: MLModel) throws -> MLDictionaryFeatureProvider {
    var dict: [String: MLFeatureValue] = [:]
    for (name, desc) in model.modelDescription.inputDescriptionsByName {
        guard let constraint = desc.multiArrayConstraint else {
            // String / image / sequence inputs: not used by StyleTTS2 stages.
            throw BenchError.unsupportedFeatureType("\(name) is non-multiArray")
        }
        let array = try makeArray(inputName: name, constraint: constraint)
        dict[name] = MLFeatureValue(multiArray: array)
    }
    return try MLDictionaryFeatureProvider(dictionary: dict)
}

// MARK: - Runner

func benchOne(
    spec: StageSpec,
    compiledRoot: URL,
    iterations: Int = 4
) -> Bool {
    let modelURL = compiledRoot.appendingPathComponent(spec.modelFile)
    guard FileManager.default.fileExists(atPath: modelURL.path) else {
        print("  [\(spec.name)] missing: \(modelURL.path)")
        return false
    }
    let cfg = MLModelConfiguration()
    cfg.computeUnits = spec.computeUnits

    let loadStart = DispatchTime.now()
    let model: MLModel
    do {
        model = try MLModel(contentsOf: modelURL, configuration: cfg)
    } catch {
        print("  [\(spec.name)] load failed: \(error)")
        return false
    }
    let loadMs =
        Double(DispatchTime.now().uptimeNanoseconds - loadStart.uptimeNanoseconds) / 1e6

    fputs("  [\(spec.name)] loaded; building inputs…\n", stderr)
    let inputs: MLDictionaryFeatureProvider
    do {
        inputs = try buildInputs(model: model)
    } catch {
        print("  [\(spec.name)] input synth failed: \(error)")
        return false
    }
    fputs("  [\(spec.name)] inputs built; warmup predict…\n", stderr)

    // Warmup
    do {
        _ = try model.prediction(from: inputs)
    } catch {
        print("  [\(spec.name)] warmup predict failed: \(error)")
        return false
    }
    fputs("  [\(spec.name)] warmup done\n", stderr)

    var times: [Double] = []
    times.reserveCapacity(iterations)
    for i in 0..<iterations {
        fputs("  [\(spec.name)] iter \(i + 1)…\n", stderr)
        let t0 = DispatchTime.now()
        do {
            _ = try model.prediction(from: inputs)
        } catch {
            print("  [\(spec.name)] predict failed: \(error)")
            return false
        }
        let ms =
            Double(DispatchTime.now().uptimeNanoseconds - t0.uptimeNanoseconds) / 1e6
        times.append(ms)
    }
    fputs("  [\(spec.name)] timed loop done\n", stderr)
    let mn = times.min() ?? 0
    let av = times.reduce(0, +) / Double(times.count)
    let mx = times.max() ?? 0

    let outputs = model.modelDescription.outputDescriptionsByName.keys.sorted().joined(
        separator: ", ")
    func pad(_ s: String, _ w: Int) -> String {
        s.count >= w ? s : s + String(repeating: " ", count: w - s.count)
    }
    func num(_ d: Double, _ frac: Int = 1) -> String {
        String(format: "%.\(frac)f", d)
    }
    let line =
        "  [\(pad(spec.name, 25)) | \(pad(spec.computeUnits.label, 11))] "
        + "load=\(num(loadMs, 0))ms  warm: min=\(num(mn)) avg=\(num(av)) max=\(num(mx)) ms"
    print(line)
    print("    inputs:  \(model.modelDescription.inputDescriptionsByName.keys.sorted().joined(separator: ", "))")
    print("    outputs: \(outputs)")
    return true
}

// MARK: - Entry

func parseCompiledRoot() -> URL {
    let args = CommandLine.arguments
    if let i = args.firstIndex(of: "--compiled"), i + 1 < args.count {
        return URL(fileURLWithPath: args[i + 1])
    }
    // Default: ../compiled relative to swift/ folder
    let exe = URL(fileURLWithPath: args[0])
    let candidate = exe
        .deletingLastPathComponent()  // .build/release
        .deletingLastPathComponent()  // .build
        .deletingLastPathComponent()  // swift
        .appendingPathComponent("compiled")
    return candidate
}

let compiledRoot = parseCompiledRoot()
print("iter3-bench")
print("  compiled root: \(compiledRoot.path)")
print("")

var ok = 0
var fail = 0
for spec in MANIFEST {
    fputs(">>> running \(spec.name)\n", stderr)
    if benchOne(spec: spec, compiledRoot: compiledRoot) {
        ok += 1
    } else {
        fail += 1
    }
    fputs("<<< done \(spec.name)\n", stderr)
}

print("")
print("\(ok)/\(MANIFEST.count) stages OK; \(fail) failed")
exit(fail == 0 ? 0 : 1)
