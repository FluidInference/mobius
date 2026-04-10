import Foundation
import CoreML

print("=======================================================================")
print("Flow Decoder CoreML Test")
print("=======================================================================")

// Test: Compile and load flow decoder (23MB - smaller than vocoder)
print("\n[1] Compiling flow decoder...")
let flowURL = URL(fileURLWithPath: "flow_decoder.mlpackage")
let startCompile = Date()

do {
    let compiledURL = try MLModel.compileModel(at: flowURL)
    let compileTime = Date().timeIntervalSince(startCompile)
    print("✓ Compiled in \(String(format: "%.2f", compileTime))s")

    print("\n[2] Loading compiled flow decoder...")
    let config = MLModelConfiguration()
    config.computeUnits = .cpuOnly
    let startLoad = Date()

    let flow = try MLModel(contentsOf: compiledURL, configuration: config)
    let loadTime = Date().timeIntervalSince(startLoad)
    print("✓ Loaded in \(String(format: "%.2f", loadTime))s")
    print("  Total time: \(String(format: "%.2f", compileTime + loadTime))s")
} catch {
    print("✗ Failed: \(error)")
}

print("\n=======================================================================")
print("Test complete")
print("=======================================================================")
