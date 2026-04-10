import Foundation
import CoreML

print("=======================================================================")
print("Simple CoreML Load Test")
print("=======================================================================")

// Test 1: Compile and load embedding model (smallest, simplest)
print("\n[1] Compiling embedding model...")
let embURL = URL(fileURLWithPath: "cosyvoice_llm_embedding.mlpackage")
let startCompile = Date()

do {
    let compiledURL = try MLModel.compileModel(at: embURL)
    let compileTime = Date().timeIntervalSince(startCompile)
    print("✓ Compiled in \(String(format: "%.2f", compileTime))s")

    print("\n[2] Loading compiled model...")
    let config = MLModelConfiguration()
    config.computeUnits = .cpuOnly
    let startLoad = Date()

    let embModel = try MLModel(contentsOf: compiledURL, configuration: config)
    let loadTime = Date().timeIntervalSince(startLoad)
    print("✓ Loaded in \(String(format: "%.2f", loadTime))s")
    print("  Total time: \(String(format: "%.2f", compileTime + loadTime))s")
} catch {
    print("✗ Failed: \(error)")
}

print("\n=======================================================================")
print("Test complete")
print("=======================================================================")
