// Swift CoreML harness: latency + memory (phys_footprint, the jetsam metric)
// for the LuxTTS TextEncoder/FmDecoder at each pipeline stage.
//
// Build:  swiftc -O swift/RssBench.swift -o build/rss_bench
// Run:    build/rss_bench <model-dir> <frame-bucket> <token-bucket> <cpu|gpu|ane|all> [steps]

import CoreML
import Foundation

func memMB() -> (footprint: Double, resident: Double) {
    var info = task_vm_info_data_t()
    var count = mach_msg_type_number_t(
        MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<integer_t>.size)
    let kr = withUnsafeMutablePointer(to: &info) {
        $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
            task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count)
        }
    }
    guard kr == KERN_SUCCESS else { return (-1, -1) }
    return (Double(info.phys_footprint) / 1_048_576, Double(info.resident_size) / 1_048_576)
}

func report(_ stage: String) {
    let m = memMB()
    let pad = stage.padding(toLength: 28, withPad: " ", startingAt: 0)
    print(pad + String(format: "footprint %7.1f MB  resident %7.1f MB", m.footprint, m.resident))
    fflush(stdout)
}

func randArray(_ shape: [NSNumber]) throws -> MLMultiArray {
    let a = try MLMultiArray(shape: shape, dataType: .float32)
    let p = a.dataPointer.bindMemory(to: Float32.self, capacity: a.count)
    var seed: UInt64 = 0x9E3779B97F4A7C15
    for i in 0..<a.count {
        seed = seed &* 6364136223846793005 &+ 1442695040888963407
        p[i] = Float32(Double(seed >> 11) / Double(UInt64.max >> 11)) * 2 - 1
    }
    return a
}

let args = CommandLine.arguments
guard args.count >= 5 else {
    print("usage: rss_bench <model-dir> <frame-bucket> <token-bucket> <cpu|gpu|ane|all> [steps]")
    exit(1)
}
let dir = args[1]
let frames = Int(args[2])!
let tokens = Int(args[3])!
let steps = args.count > 5 ? Int(args[5])! : 4

let config = MLModelConfiguration()
switch args[4] {
case "cpu": config.computeUnits = .cpuOnly
case "gpu": config.computeUnits = .cpuAndGPU
case "ane": config.computeUnits = .cpuAndNeuralEngine
default: config.computeUnits = .all
}

print("== \(dir) frames=\(frames) tokens=\(tokens) cu=\(args[4]) steps=\(steps)")
report("baseline")

var t0 = Date()
let te = try MLModel(
    contentsOf: URL(fileURLWithPath: "\(dir)/TextEncoder.mlmodelc"), configuration: config)
let fm = try MLModel(
    contentsOf: URL(fileURLWithPath: "\(dir)/FmDecoder.mlmodelc"), configuration: config)
print(String(format: "model load: %.0f ms", -t0.timeIntervalSinceNow * 1000))
report("after load")

// text encoder inputs
let tokArr = try MLMultiArray(shape: [1, NSNumber(value: tokens)], dataType: .int32)
let tokMask = try MLMultiArray(shape: [1, NSNumber(value: tokens)], dataType: .float32)
for i in 0..<tokens {
    tokArr[i] = 0
    tokMask[i] = i < 200 ? 0.0 : 1.0
}
let teIn = try MLDictionaryFeatureProvider(dictionary: [
    "tokens": tokArr, "padding_mask": tokMask,
])

// fm decoder inputs
let shape: [NSNumber] = [1, NSNumber(value: frames), 100]
let tArr = try MLMultiArray(shape: [1], dataType: .float32)
tArr[0] = 0.5
let gArr = try MLMultiArray(shape: [1], dataType: .float32)
gArr[0] = 3.0
let fMask = try MLMultiArray(shape: [1, NSNumber(value: frames)], dataType: .float32)
for i in 0..<frames { fMask[i] = 0.0 }
let fmIn = try MLDictionaryFeatureProvider(dictionary: [
    "t": tArr, "x": try randArray(shape),
    "text_condition": try randArray(shape), "speech_condition": try randArray(shape),
    "guidance_scale": gArr, "padding_mask": fMask,
])

t0 = Date()
_ = try te.prediction(from: teIn)
_ = try fm.prediction(from: fmIn)
print(String(format: "first predict (warm ANE/GPU): %.0f ms", -t0.timeIntervalSinceNow * 1000))
report("after first predict")

// timed: text encoder
var times: [Double] = []
for _ in 0..<10 {
    t0 = Date()
    _ = try te.prediction(from: teIn)
    times.append(-t0.timeIntervalSinceNow * 1000)
}
print(String(format: "text_encoder: %.2f ms (min %.2f)", times.reduce(0, +) / 10, times.min()!))

// timed: decoder steps
times = []
for _ in 0..<10 {
    t0 = Date()
    _ = try fm.prediction(from: fmIn)
    times.append(-t0.timeIntervalSinceNow * 1000)
}
let stepMs = times.reduce(0, +) / 10
print(String(format: "fm_decoder/step: %.2f ms (min %.2f)", stepMs, times.min()!))

// full synth loop
t0 = Date()
_ = try te.prediction(from: teIn)
for _ in 0..<steps { _ = try fm.prediction(from: fmIn) }
let coreMs = -t0.timeIntervalSinceNow * 1000
let audioSec = Double(frames - 469) / 93.75
print(String(format: "core pipeline (te+%d steps): %.1f ms -> core RTFx %.1fx (for %.1fs gen)",
             steps, coreMs, audioSec * 1000 / coreMs, audioSec))
report("steady state")
