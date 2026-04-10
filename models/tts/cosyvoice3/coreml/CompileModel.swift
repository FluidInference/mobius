import Foundation
import CoreML

print("Compiling hift_vocoder.mlpackage...")

let sourceURL = URL(fileURLWithPath: "converted/hift_vocoder.mlpackage")
let startCompile = Date()

do {
    let compiledURL = try MLModel.compileModel(at: sourceURL)
    let compileTime = Date().timeIntervalSince(startCompile)

    print("✓ Compiled in \(String(format: "%.2f", compileTime))s")
    print("✓ Compiled model at: \(compiledURL.path)")
} catch {
    print("✗ Compilation failed: \(error)")
    exit(1)
}
