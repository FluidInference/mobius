// Measure CoreML model RAM usage without Python overhead
// Usage: ./measure_coreml_ram <model.mlpackage>

import CoreML
import Foundation

func getMemoryFootprint() -> UInt64 {
    var info = mach_task_basic_info()
    var count = mach_msg_type_number_t(MemoryLayout<mach_task_basic_info>.size) / 4
    let result = withUnsafeMutablePointer(to: &info) {
        $0.withMemoryRebound(to: integer_t.self, capacity: 1) {
            task_info(mach_task_self_, task_flavor_t(MACH_TASK_BASIC_INFO), $0, &count)
        }
    }
    return result == KERN_SUCCESS ? info.resident_size : 0
}

func formatBytes(_ bytes: UInt64) -> String {
    let mb = Double(bytes) / (1024 * 1024)
    return String(format: "%.1f MB", mb)
}

func compileModelIfNeeded(at path: String) throws -> URL {
    let url = URL(fileURLWithPath: path)

    // If already compiled (.mlmodelc), use directly
    if path.hasSuffix(".mlmodelc") {
        return url
    }

    // Compile .mlpackage to temporary .mlmodelc
    print("  Compiling model...")
    let compiledURL = try MLModel.compileModel(at: url)
    return compiledURL
}

func measureModel(at path: String, computeUnits: MLComputeUnits) async throws -> (load: UInt64, inference: UInt64) {
    // Compile the model first
    let compiledURL = try compileModelIfNeeded(at: path)

    // Measure baseline AFTER compilation
    let baseline = getMemoryFootprint()

    // Configure and load model
    let config = MLModelConfiguration()
    config.computeUnits = computeUnits

    let loadStart = Date()
    let model = try MLModel(contentsOf: compiledURL, configuration: config)
    let loadTime = Date().timeIntervalSince(loadStart)

    let afterLoad = getMemoryFootprint()
    let loadRAM = afterLoad - baseline

    print("  Load time: \(String(format: "%.2f", loadTime))s")
    print("  RAM after load: \(formatBytes(loadRAM))")

    // Try a simple inference to measure peak RAM
    let desc = model.modelDescription
    let inputNames = desc.inputDescriptionsByName.keys.sorted()
    print("  Inputs: \(inputNames.prefix(5).joined(separator: ", "))\(inputNames.count > 5 ? "..." : "")")

    // Create dummy inputs
    var inputs: [String: MLFeatureValue] = [:]
    for (name, inputDesc) in desc.inputDescriptionsByName {
        if let constraint = inputDesc.multiArrayConstraint {
            _ = constraint.shape.map { $0.intValue }
            let array = try MLMultiArray(shape: constraint.shape, dataType: constraint.dataType)
            inputs[name] = MLFeatureValue(multiArray: array)
        }
    }

    if !inputs.isEmpty {
        let provider = try MLDictionaryFeatureProvider(dictionary: inputs)
        let inferStart = Date()
        _ = try await model.prediction(from: provider)
        let inferTime = Date().timeIntervalSince(inferStart)
        print("  Inference time: \(String(format: "%.3f", inferTime))s")
    }

    let afterInference = getMemoryFootprint()
    let inferRAM = afterInference - baseline

    return (loadRAM, inferRAM)
}

func directorySize(at url: URL) -> UInt64 {
    let fm = FileManager.default
    guard let enumerator = fm.enumerator(at: url, includingPropertiesForKeys: [.fileSizeKey], options: [.skipsHiddenFiles]) else {
        return 0
    }
    var total: UInt64 = 0
    for case let fileURL as URL in enumerator {
        if let size = try? fileURL.resourceValues(forKeys: [.fileSizeKey]).fileSize {
            total += UInt64(size)
        }
    }
    return total
}

@main
struct MeasureCoreMLRAM {
    static func main() async {
        let args = CommandLine.arguments

        if args.count < 2 {
            print("Usage: ./measure_coreml_ram <model.mlpackage> [model2.mlpackage ...]")
            print("")
            print("Measures pure CoreML RAM usage for each model.")
            print("Tests with different compute units: CPU_AND_NE, CPU_AND_GPU, ALL")
            return
        }

        let modelPaths = Array(args.dropFirst())

        for path in modelPaths {
            print("\n" + String(repeating: "=", count: 60))
            print("Model: \(URL(fileURLWithPath: path).lastPathComponent)")
            print(String(repeating: "=", count: 60))

            // Check directory size (mlpackage is a directory)
            let url = URL(fileURLWithPath: path)
            let size = directorySize(at: url)
            print("Disk size: \(formatBytes(size))")

            let computeOptions: [(MLComputeUnits, String)] = [
                (.cpuAndNeuralEngine, "CPU_AND_NE"),
                (.cpuAndGPU, "CPU_AND_GPU"),
                (.all, "ALL")
            ]

            for (units, name) in computeOptions {
                print("\n[\(name)]")
                do {
                    let (_, inferRAM) = try await measureModel(at: path, computeUnits: units)
                    print("  Peak RAM (after inference): \(formatBytes(inferRAM))")
                } catch {
                    print("  Error: \(error.localizedDescription)")
                }
            }
        }

        print("\n" + String(repeating: "=", count: 60))
        print("Done.")
    }
}
