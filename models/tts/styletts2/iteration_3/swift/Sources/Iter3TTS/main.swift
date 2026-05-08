// Iter3TTS: side-loaded CoreML pipeline.
//
// Reads .npy fixtures dumped by `dump_intermediates.py`, runs each
// iteration_3 .mlmodelc stage's predict in Swift with the documented
// placement, and writes a 24 kHz mono WAV from decoder_upsample's
// output. Inter-stage glue (alignment matmul, asr-shift, s/ref split)
// is *not* re-implemented here — the dumper precomputes each stage's
// inputs in Python.
//
// Build & run:
//   cd iteration_3/swift
//   swift build -c release
//   .build/release/iter3-tts \
//       --compiled ../compiled \
//       --fixtures fixtures \
//       --output fixtures_swift.wav

import CoreML
import Foundation

// MARK: - Stage placement (mirrors Iter3Bench)

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

let SAMPLE_RATE = 24_000

// MARK: - Errors

enum TTSError: Error, CustomStringConvertible {
    case missing(String)
    case unsupportedDtype(String)
    case manifestShape(String)
    case predict(String)
    case io(String)
    case parse(String)

    var description: String {
        switch self {
        case .missing(let s): return "missing: \(s)"
        case .unsupportedDtype(let s): return "unsupported dtype: \(s)"
        case .manifestShape(let s): return "manifest/shape mismatch: \(s)"
        case .predict(let s): return "predict: \(s)"
        case .io(let s): return "io: \(s)"
        case .parse(let s): return "parse: \(s)"
        }
    }
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

// MARK: - .npy reader (v1.0/v2.0/v3.0, C-contiguous, '<f4' or '<i4')

struct NpyArray {
    let shape: [Int]
    let dtype: String         // '<f4' or '<i4'
    let mlDataType: MLMultiArrayDataType
    let dataPointer: UnsafeMutableRawPointer
    let dataCount: Int        // element count
    let backing: Data         // keeps the buffer alive
}

func loadNpy(at url: URL) throws -> NpyArray {
    let blob = try Data(contentsOf: url, options: .alwaysMapped)
    if blob.count < 10 { throw TTSError.parse("\(url.path): too small") }

    // Magic
    let magic: [UInt8] = [0x93, 0x4E, 0x55, 0x4D, 0x50, 0x59]
    for i in 0..<6 where blob[i] != magic[i] {
        throw TTSError.parse("\(url.path): bad magic")
    }
    let major = blob[6]
    let _ = blob[7]
    var headerLen: Int
    var headerStart: Int
    switch major {
    case 1:
        let lo = Int(blob[8])
        let hi = Int(blob[9])
        headerLen = lo | (hi << 8)
        headerStart = 10
    case 2, 3:
        let b8  = Int(blob[8])
        let b9  = Int(blob[9])
        let b10 = Int(blob[10])
        let b11 = Int(blob[11])
        headerLen = b8 | (b9 << 8) | (b10 << 16) | (b11 << 24)
        headerStart = 12
    default:
        throw TTSError.parse("\(url.path): unsupported npy version \(major)")
    }
    let headerEnd = headerStart + headerLen
    guard headerEnd <= blob.count else {
        throw TTSError.parse("\(url.path): truncated header")
    }
    let headerData = blob[headerStart..<headerEnd]
    guard let header = String(data: headerData, encoding: .ascii) else {
        throw TTSError.parse("\(url.path): non-ascii header")
    }

    // Parse fields. The header is a Python repr like:
    //   {'descr': '<f4', 'fortran_order': False, 'shape': (1, 57), }
    func extract(_ key: String) throws -> String {
        guard let r = header.range(of: "'\(key)'") else {
            throw TTSError.parse("\(url.path): missing key '\(key)'")
        }
        let after = header[r.upperBound...]
        guard let colon = after.firstIndex(of: ":") else {
            throw TTSError.parse("\(url.path): malformed '\(key)'")
        }
        let rest = after[after.index(after: colon)...].drop(while: { $0 == " " })
        // Value is up to next comma at depth 0 (parens count).
        var depth = 0
        var end = rest.startIndex
        for idx in rest.indices {
            let c = rest[idx]
            if c == "(" || c == "[" { depth += 1 }
            else if c == ")" || c == "]" { depth -= 1 }
            else if c == "," && depth == 0 { end = idx; break }
            end = rest.index(after: idx)
        }
        return String(rest[rest.startIndex..<end]).trimmingCharacters(in: .whitespaces)
    }

    let descrRaw = try extract("descr")          // "'<f4'"
    let fortranRaw = try extract("fortran_order")
    let shapeRaw = try extract("shape")           // "(1, 57)"

    let dtype = descrRaw.replacingOccurrences(of: "'", with: "")
    let fortran = fortranRaw == "True"
    if fortran {
        throw TTSError.parse("\(url.path): fortran_order=True not supported")
    }
    var shapeInner = shapeRaw
    shapeInner.removeAll(where: { $0 == "(" || $0 == ")" || $0 == " " })
    let shape: [Int] =
        shapeInner.isEmpty
        ? []
        : shapeInner
            .split(separator: ",", omittingEmptySubsequences: true)
            .compactMap { Int($0) }

    let mlDtype: MLMultiArrayDataType
    let elemSize: Int
    switch dtype {
    case "<f4":
        mlDtype = .float32
        elemSize = 4
    case "<i4":
        mlDtype = .int32
        elemSize = 4
    default:
        throw TTSError.unsupportedDtype("\(dtype) in \(url.lastPathComponent)")
    }

    let count = shape.reduce(1, *)
    let dataStart = headerEnd
    let needBytes = count * elemSize
    guard dataStart + needBytes <= blob.count else {
        throw TTSError.parse("\(url.path): truncated data")
    }

    // Copy into an aligned buffer so MLMultiArray's strides+lifetime
    // are independent of the mmap window.
    let buf = malloc(needBytes)!
    blob.copyBytes(
        to: UnsafeMutableBufferPointer(
            start: buf.bindMemory(to: UInt8.self, capacity: needBytes),
            count: needBytes),
        from: dataStart..<(dataStart + needBytes))

    let owning = Data(
        bytesNoCopy: buf,
        count: needBytes,
        deallocator: .free)

    return NpyArray(
        shape: shape,
        dtype: dtype,
        mlDataType: mlDtype,
        dataPointer: buf,
        dataCount: count,
        backing: owning)
}

func makeMultiArray(_ npy: NpyArray) throws -> MLMultiArray {
    let nsShape = npy.shape.map { NSNumber(value: $0) }
    let strides = computeStrides(npy.shape).map { NSNumber(value: $0) }
    return try MLMultiArray(
        dataPointer: npy.dataPointer,
        shape: nsShape,
        dataType: npy.mlDataType,
        strides: strides,
        deallocator: nil)
}

func computeStrides(_ shape: [Int]) -> [Int] {
    var strides = Array(repeating: 1, count: shape.count)
    if shape.count <= 1 { return strides }
    for i in (0..<(shape.count - 1)).reversed() {
        strides[i] = strides[i + 1] * shape[i + 1]
    }
    return strides
}

// MARK: - Manifest

struct StageCall {
    struct Field {
        let name: String
        let shape: [Int]
        let dtype: String
    }
    let dir: String
    let inputs: [Field]
    let outputs: [Field]
}

struct ManifestData {
    let stageOrder: [String]
    let calls: [String: StageCall]
}

func loadManifest(_ url: URL) throws -> ManifestData {
    let data = try Data(contentsOf: url)
    guard let any = try? JSONSerialization.jsonObject(with: data) else {
        throw TTSError.parse("manifest.json: not JSON")
    }
    guard let root = any as? [String: Any],
          let order = root["stage_order"] as? [String],
          let stages = root["stages"] as? [String: Any]
    else {
        throw TTSError.parse("manifest.json: missing stage_order/stages")
    }

    var out: [String: StageCall] = [:]
    for s in order {
        guard let stageDict = stages[s] as? [String: Any],
              let calls = stageDict["calls"] as? [[String: Any]],
              let call = calls.first
        else {
            throw TTSError.parse("manifest.json: missing stage \(s)")
        }
        let dir = (call["dir"] as? String) ?? s
        func parseFields(_ key: String) throws -> [StageCall.Field] {
            guard let arr = call[key] as? [[String: Any]] else {
                throw TTSError.parse("manifest.json: \(s).\(key) malformed")
            }
            return try arr.map { d in
                guard let n = d["name"] as? String,
                      let sh = d["shape"] as? [Int],
                      let dt = d["dtype"] as? String
                else {
                    throw TTSError.parse("manifest.json: bad field in \(s).\(key)")
                }
                return StageCall.Field(name: n, shape: sh, dtype: dt)
            }
        }
        out[s] = StageCall(
            dir: dir,
            inputs: try parseFields("inputs"),
            outputs: try parseFields("outputs"))
    }
    return ManifestData(stageOrder: order, calls: out)
}

// MARK: - WAV writer (mono float32 → int16 little-endian PCM)

func writeWavMonoF32(samples: [Float], sampleRate: Int, to url: URL) throws {
    let n = samples.count
    let byteRate = sampleRate * 2
    let dataSize = n * 2
    let chunkSize = 36 + dataSize

    var data = Data()
    data.reserveCapacity(44 + dataSize)

    func appendString(_ s: String) {
        data.append(s.data(using: .ascii)!)
    }
    func appendU32LE(_ v: UInt32) {
        var x = v.littleEndian
        withUnsafeBytes(of: &x) { data.append(contentsOf: $0) }
    }
    func appendU16LE(_ v: UInt16) {
        var x = v.littleEndian
        withUnsafeBytes(of: &x) { data.append(contentsOf: $0) }
    }

    appendString("RIFF")
    appendU32LE(UInt32(chunkSize))
    appendString("WAVE")
    appendString("fmt ")
    appendU32LE(16)            // PCM fmt-chunk size
    appendU16LE(1)             // PCM format
    appendU16LE(1)             // mono
    appendU32LE(UInt32(sampleRate))
    appendU32LE(UInt32(byteRate))
    appendU16LE(2)             // block align
    appendU16LE(16)            // bits per sample
    appendString("data")
    appendU32LE(UInt32(dataSize))

    // Float32 [-1, 1] → Int16
    var pcm = [Int16](repeating: 0, count: n)
    for i in 0..<n {
        let v = max(-1.0, min(1.0, samples[i]))
        pcm[i] = Int16((v * 32767.0).rounded())
    }
    pcm.withUnsafeBufferPointer { buf in
        data.append(Data(buffer: buf))
    }

    try data.write(to: url, options: .atomic)
}

// MARK: - Pipeline runner

struct Args {
    var compiledRoot: URL
    var fixtures: URL
    var output: URL
}

func parseArgs() -> Args {
    let argv = CommandLine.arguments
    func read(_ flag: String, _ defaultURL: URL) -> URL {
        if let i = argv.firstIndex(of: flag), i + 1 < argv.count {
            return URL(fileURLWithPath: argv[i + 1])
        }
        return defaultURL
    }
    let cwd = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
    return Args(
        compiledRoot: read("--compiled", cwd.appendingPathComponent("../compiled")),
        fixtures: read("--fixtures", cwd.appendingPathComponent("fixtures")),
        output: read("--output", cwd.appendingPathComponent("fixtures_swift.wav")))
}

func runStage(
    spec: StageSpec,
    call: StageCall,
    fixtures: URL,
    compiledRoot: URL
) throws -> [String: MLMultiArray] {
    let modelURL = compiledRoot.appendingPathComponent(spec.modelFile)
    guard FileManager.default.fileExists(atPath: modelURL.path) else {
        throw TTSError.missing(modelURL.path)
    }
    let cfg = MLModelConfiguration()
    cfg.computeUnits = spec.computeUnits

    let loadStart = DispatchTime.now()
    let model = try MLModel(contentsOf: modelURL, configuration: cfg)
    let loadMs =
        Double(DispatchTime.now().uptimeNanoseconds - loadStart.uptimeNanoseconds) / 1e6

    // Build inputs from .npy fixtures.
    let stageDir = fixtures.appendingPathComponent(call.dir)
    var feed: [String: MLFeatureValue] = [:]
    var heldArrays: [NpyArray] = []
    for f in call.inputs {
        let url = stageDir.appendingPathComponent("in_\(f.name).npy")
        let npy = try loadNpy(at: url)
        if npy.shape != f.shape {
            throw TTSError.manifestShape(
                "\(spec.name).\(f.name): manifest \(f.shape) vs npy \(npy.shape)")
        }
        let arr = try makeMultiArray(npy)
        feed[f.name] = MLFeatureValue(multiArray: arr)
        heldArrays.append(npy)  // keep buffer alive
    }
    let provider = try MLDictionaryFeatureProvider(dictionary: feed)

    let predStart = DispatchTime.now()
    let result: MLFeatureProvider
    do {
        result = try model.prediction(from: provider)
    } catch {
        throw TTSError.predict("\(spec.name): \(error)")
    }
    let predMs =
        Double(DispatchTime.now().uptimeNanoseconds - predStart.uptimeNanoseconds) / 1e6

    var outputs: [String: MLMultiArray] = [:]
    for f in call.outputs {
        guard let v = result.featureValue(for: f.name)?.multiArrayValue else {
            throw TTSError.predict("\(spec.name): missing output \(f.name)")
        }
        outputs[f.name] = v
    }

    func pad(_ s: String, _ w: Int) -> String {
        s.count >= w ? s : s + String(repeating: " ", count: w - s.count)
    }
    print(
        "  [\(pad(spec.name, 25)) | \(pad(spec.computeUnits.label, 11))] "
        + "load=\(Int(loadMs))ms predict=\(String(format: "%.1f", predMs))ms")

    _ = heldArrays  // explicit: buffers outlive predict()
    return outputs
}

// MARK: - Entry

let args = parseArgs()
print("iter3-tts")
print("  compiled root: \(args.compiledRoot.path)")
print("  fixtures:      \(args.fixtures.path)")
print("  output:        \(args.output.path)")
print("")

let manifestURL = args.fixtures.appendingPathComponent("manifest.json")
let manifest = try loadManifest(manifestURL)
let stageByName = Dictionary(uniqueKeysWithValues: MANIFEST.map { ($0.name, $0) })

var lastOutputs: [String: MLMultiArray] = [:]
let totalStart = DispatchTime.now()
for stageName in manifest.stageOrder {
    guard let spec = stageByName[stageName] else {
        throw TTSError.parse("no MANIFEST entry for \(stageName)")
    }
    guard let call = manifest.calls[stageName] else {
        throw TTSError.parse("no manifest call for \(stageName)")
    }
    lastOutputs = try runStage(
        spec: spec,
        call: call,
        fixtures: args.fixtures,
        compiledRoot: args.compiledRoot)
}
let totalMs =
    Double(DispatchTime.now().uptimeNanoseconds - totalStart.uptimeNanoseconds) / 1e6
print("\nPipeline total: \(Int(totalMs))ms")

// Final stage's first (and only) output is the audio: shape (1, 1, T).
guard let lastSpec = manifest.stageOrder.last,
      let call = manifest.calls[lastSpec],
      let firstOutName = call.outputs.first?.name,
      let audioArr = lastOutputs[firstOutName]
else {
    throw TTSError.parse("no audio output from final stage")
}
let audioCount = audioArr.count
var samples = [Float](repeating: 0, count: audioCount)
switch audioArr.dataType {
case .float32:
    let p = audioArr.dataPointer.bindMemory(to: Float.self, capacity: audioCount)
    for i in 0..<audioCount { samples[i] = p[i] }
case .float16:
    let p = audioArr.dataPointer.bindMemory(to: Float16.self, capacity: audioCount)
    for i in 0..<audioCount { samples[i] = Float(p[i]) }
default:
    throw TTSError.unsupportedDtype("audio dtype \(audioArr.dataType.rawValue)")
}

// Mirror Python's tail trim of 50 samples (no other trim needed).
let trimmedCount = max(0, samples.count - 50)
samples = Array(samples.prefix(trimmedCount))

try writeWavMonoF32(samples: samples, sampleRate: SAMPLE_RATE, to: args.output)

let durationS = Double(samples.count) / Double(SAMPLE_RATE)
print(String(format: "Wrote %@ (%.2fs @ %d Hz)",
             args.output.path, durationS, SAMPLE_RATE))
