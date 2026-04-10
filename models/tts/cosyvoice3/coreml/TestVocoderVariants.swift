import Foundation
import CoreML

print("=======================================================================")
print("Test Different Vocoder Variants")
print("=======================================================================")

let variants = [
    "converted/hift_vocoder.mlpackage",
    "converted/hift_vocoder_fp32.mlpackage",
    "converted/hift_vocoder_validated.mlpackage",
]

for (index, path) in variants.enumerated() {
    print("\n[\(index + 1)/\(variants.count)] Testing: \(path)")

    let url = URL(fileURLWithPath: path)

    // Check if exists
    if !FileManager.default.fileExists(atPath: url.path) {
        print("  ✗ File does not exist")
        continue
    }

    // Try to compile
    print("  Compiling...")
    let startCompile = Date()

    do {
        let compiledURL = try MLModel.compileModel(at: url)
        let compileTime = Date().timeIntervalSince(startCompile)
        print("  ✓ Compiled in \(String(format: "%.2f", compileTime))s")

        // Try to load (with 10s timeout)
        print("  Loading (10s timeout)...")
        let config = MLModelConfiguration()
        config.computeUnits = .cpuOnly
        let startLoad = Date()

        // We can't actually timeout in Swift, so this is just for testing
        let model = try MLModel(contentsOf: compiledURL, configuration: config)
        let loadTime = Date().timeIntervalSince(startLoad)

        print("  ✓ Loaded in \(String(format: "%.2f", loadTime))s")
        print("  SUCCESS! This variant works")

    } catch {
        print("  ✗ Failed: \(error)")
    }
}

print("\n=======================================================================")
print("Test complete")
print("=======================================================================")
