// Interleaved A/B benchmark: Parakeet EOU RNNT decode step.
//   A = two-model reference (decoder.prediction + joint_decision.prediction)
//   B = decoder_joint_decision_pipeline (1 dispatch, bit-exact)
//   C = decoder_joint_decision_fused (1 dispatch, MIL-fused, fp16-tie drift)
//
// 10 warmup + 200 timed per variant, interleaved (A,B,C,A,B,C,...) per the
// mobius Trial 15 methodology. Reports median / p95 per compute-unit config.
//
// Usage:
//   swiftc -O bench_fused_decode.swift -o /tmp/bench_eou
//   /tmp/bench_eou <model_dir_with_ref_mlmodelc> <pipeline.mlpackage> <fused.mlpackage>

import CoreML
import Foundation

let warmup = 10
let iters = 200

func makeArray(_ shape: [NSNumber], _ fill: (Int) -> Float) -> MLMultiArray {
    let arr = try! MLMultiArray(shape: shape, dataType: .float32)
    let ptr = arr.dataPointer.bindMemory(to: Float.self, capacity: arr.count)
    for i in 0..<arr.count { ptr[i] = fill(i) }
    return arr
}

func median(_ xs: [Double]) -> Double {
    let s = xs.sorted()
    return s[s.count / 2]
}
func p95(_ xs: [Double]) -> Double {
    let s = xs.sorted()
    return s[Int(Double(s.count) * 0.95)]
}

let args = CommandLine.arguments
guard args.count == 4 else {
    print("usage: bench_eou <ref_model_dir> <pipeline.mlpackage> <fused.mlpackage>")
    exit(1)
}
let refDir = URL(fileURLWithPath: args[1])
let pipelinePkg = URL(fileURLWithPath: args[2])
let fusedPkg = URL(fileURLWithPath: args[3])

// Compile mlpackages once.
let pipelineC = try! MLModel.compileModel(at: pipelinePkg)
let fusedC = try! MLModel.compileModel(at: fusedPkg)

// Deterministic pseudo-random encoder step.
var seed: UInt64 = 0x9E37_79B9_7F4A_7C15
func rnd() -> Float {
    seed = seed &* 6_364_136_223_846_793_005 &+ 1_442_695_040_888_963_407
    return Float((seed >> 33) & 0xFFFF) / 65536.0 - 0.5
}

let encoderStep = makeArray([1, 512, 1]) { _ in rnd() * 4.0 }
let hIn = makeArray([1, 1, 640]) { _ in rnd() * 0.2 }
let cIn = makeArray([1, 1, 640]) { _ in rnd() * 0.2 }
let targets = try! MLMultiArray(shape: [1, 1], dataType: .int32)
targets[0] = 1026
let targetLength = try! MLMultiArray(shape: [1], dataType: .int32)
targetLength[0] = 1

let configs: [(String, MLComputeUnits)] = [
    ("CPU_ONLY", .cpuOnly),
    ("CPU_AND_NE", .cpuAndNeuralEngine),
    ("ALL", .all),
]

print("variant            cu          median_ms   p95_ms")
for (cuName, cu) in configs {
    let cfg = MLModelConfiguration()
    cfg.computeUnits = cu

    let decoder = try! MLModel(contentsOf: refDir.appendingPathComponent("decoder.mlmodelc"), configuration: cfg)
    let joint = try! MLModel(
        contentsOf: refDir.appendingPathComponent("joint_decision.mlmodelc"), configuration: cfg)
    let pipeline = try! MLModel(contentsOf: pipelineC, configuration: cfg)
    let fused = try! MLModel(contentsOf: fusedC, configuration: cfg)

    func runRef() -> Int32 {
        let decIn = try! MLDictionaryFeatureProvider(dictionary: [
            "targets": MLFeatureValue(multiArray: targets),
            "target_length": MLFeatureValue(multiArray: targetLength),
            "h_in": MLFeatureValue(multiArray: hIn),
            "c_in": MLFeatureValue(multiArray: cIn),
        ])
        let decOut = try! decoder.prediction(from: decIn)
        let decStep = decOut.featureValue(for: "decoder")!.multiArrayValue!
        let jIn = try! MLDictionaryFeatureProvider(dictionary: [
            "encoder_step": MLFeatureValue(multiArray: encoderStep),
            "decoder_step": MLFeatureValue(multiArray: decStep),
        ])
        let jOut = try! joint.prediction(from: jIn)
        return jOut.featureValue(for: "token_id")!.multiArrayValue![0].int32Value
    }

    func runOne(_ model: MLModel, needsTargetLength: Bool) -> Int32 {
        var dict: [String: MLFeatureValue] = [
            "targets": MLFeatureValue(multiArray: targets),
            "h_in": MLFeatureValue(multiArray: hIn),
            "c_in": MLFeatureValue(multiArray: cIn),
            "encoder_step": MLFeatureValue(multiArray: encoderStep),
        ]
        if needsTargetLength {
            dict["target_length"] = MLFeatureValue(multiArray: targetLength)
        }
        let input = try! MLDictionaryFeatureProvider(dictionary: dict)
        let out = try! model.prediction(from: input)
        return out.featureValue(for: "token_id")!.multiArrayValue![0].int32Value
    }

    // Sanity: all three agree on this input (pipeline must; fused should on non-ties).
    let tRef = runRef()
    let tPipe = runOne(pipeline, needsTargetLength: true)
    let tFused = runOne(fused, needsTargetLength: false)
    if tRef != tPipe { print("WARNING [\(cuName)]: pipeline token \(tPipe) != ref \(tRef)") }
    if tRef != tFused { print("note [\(cuName)]: fused token \(tFused) != ref \(tRef) (fp16 tie)") }

    var tRefMs: [Double] = []
    var tPipeMs: [Double] = []
    var tFusedMs: [Double] = []
    for i in 0..<(warmup + iters) {
        let a0 = DispatchTime.now().uptimeNanoseconds
        _ = runRef()
        let a1 = DispatchTime.now().uptimeNanoseconds
        _ = runOne(pipeline, needsTargetLength: true)
        let a2 = DispatchTime.now().uptimeNanoseconds
        _ = runOne(fused, needsTargetLength: false)
        let a3 = DispatchTime.now().uptimeNanoseconds
        if i >= warmup {
            tRefMs.append(Double(a1 - a0) / 1e6)
            tPipeMs.append(Double(a2 - a1) / 1e6)
            tFusedMs.append(Double(a3 - a2) / 1e6)
        }
    }

    func row(_ name: String, _ xs: [Double]) {
        let n = name.padding(toLength: 18, withPad: " ", startingAt: 0)
        let c = cuName.padding(toLength: 11, withPad: " ", startingAt: 0)
        print(n + " " + c + " " + String(format: "%9.4f %8.4f", median(xs), p95(xs)))
    }
    row("ref(dec+joint)", tRefMs)
    row("pipeline", tPipeMs)
    row("fused(MIL)", tFusedMs)
}
