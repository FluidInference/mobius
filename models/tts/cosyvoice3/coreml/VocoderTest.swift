import Foundation
import CoreML

print("=======================================================================")
print("Vocoder CoreML Test")
print("=======================================================================")

// Test: Compile and load vocoder
print("\n[1] Compiling vocoder model...")
let vocoderURL = URL(fileURLWithPath: "converted/hift_vocoder.mlpackage")
let startCompile = Date()

do {
    let compiledURL = try MLModel.compileModel(at: vocoderURL)
    let compileTime = Date().timeIntervalSince(startCompile)
    print("✓ Compiled in \(String(format: "%.2f", compileTime))s")
    print("  Compiled to: \(compiledURL.path)")

    print("\n[2] Loading compiled vocoder...")
    let config = MLModelConfiguration()
    config.computeUnits = .cpuOnly
    let startLoad = Date()

    let vocoder = try MLModel(contentsOf: compiledURL, configuration: config)
    let loadTime = Date().timeIntervalSince(startLoad)
    print("✓ Loaded in \(String(format: "%.2f", loadTime))s")
    print("  Total time: \(String(format: "%.2f", compileTime + loadTime))s")
    print("  Model: \(vocoder)")
} catch {
    print("✗ Failed: \(error)")
}

print("\n=======================================================================")
print("Test complete")
print("=======================================================================")
