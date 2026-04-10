import Foundation
import CoreML

print("=======================================================================")
print("LM Head CoreML Test")
print("=======================================================================")

// Test: Compile and load LM head (260MB - same as embedding)
print("\n[1] Compiling LM head model...")
let lmheadURL = URL(fileURLWithPath: "cosyvoice_llm_lm_head.mlpackage")
let startCompile = Date()

do {
    let compiledURL = try MLModel.compileModel(at: lmheadURL)
    let compileTime = Date().timeIntervalSince(startCompile)
    print("✓ Compiled in \(String(format: "%.2f", compileTime))s")

    print("\n[2] Loading compiled LM head...")
    let config = MLModelConfiguration()
    config.computeUnits = .cpuOnly
    let startLoad = Date()

    let lmhead = try MLModel(contentsOf: compiledURL, configuration: config)
    let loadTime = Date().timeIntervalSince(startLoad)
    print("✓ Loaded in \(String(format: "%.2f", loadTime))s")
    print("  Total time: \(String(format: "%.2f", compileTime + loadTime))s")
} catch {
    print("✗ Failed: \(error)")
}

print("\n=======================================================================")
print("Test complete")
print("=======================================================================")
