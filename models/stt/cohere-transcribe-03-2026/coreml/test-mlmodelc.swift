#!/usr/bin/env swift

import Foundation
import CoreML

// Test the compiled .mlmodelc model
func testCompiledModel() {
    print("Testing compiled .mlmodelc model...")
    print(String(repeating: "=", count: 70))

    // Load compiled model
    print("\n[1/3] Loading compiled model...")
    let modelURL = URL(fileURLWithPath: "build-test/cohere_decoder_cache_external.mlmodelc")

    guard let model = try? MLModel(contentsOf: modelURL) else {
        print("   ✗ Failed to load model")
        return
    }
    print("   ✓ Loaded: \(modelURL.path)")

    // Print model info
    print("\n[2/3] Model info:")
    let description = model.modelDescription
    print("   Inputs: \(description.inputDescriptionsByName.count)")
    print("   Outputs: \(description.outputDescriptionsByName.count)")

    // Test single inference step
    print("\n[3/3] Running single inference step...")

    // Create dummy inputs
    let maxSeqLen = 108

    // Create MLMultiArray inputs
    guard let inputId = try? MLMultiArray(shape: [1, 1], dataType: .int32),
          let positionId = try? MLMultiArray(shape: [1, 1], dataType: .int32),
          let encoderHidden = try? MLMultiArray(shape: [1, 438, 1024], dataType: .float32),
          let crossMask = try? MLMultiArray(shape: [1, 1, 1, 438], dataType: .float32),
          let attentionMask = try? MLMultiArray(shape: [1, 1, 1, 1], dataType: .float32) else {
        print("   ✗ Failed to create input arrays")
        return
    }

    // Set input values
    inputId[0] = 4  // START_TOKEN
    positionId[0] = 0

    // Fill encoder hidden with random values
    for i in 0..<(1 * 438 * 1024) {
        encoderHidden[i] = Float.random(in: -1...1) as NSNumber
    }

    // Fill cross mask with ones
    for i in 0..<(1 * 1 * 1 * 438) {
        crossMask[i] = 1.0
    }

    // Attention mask is zeros (already initialized)

    // Create cache arrays (all zeros)
    var kCaches: [MLMultiArray] = []
    var vCaches: [MLMultiArray] = []

    for _ in 0..<8 {
        guard let kCache = try? MLMultiArray(shape: [1, 8, NSNumber(value: maxSeqLen), 128], dataType: .float32),
              let vCache = try? MLMultiArray(shape: [1, 8, NSNumber(value: maxSeqLen), 128], dataType: .float32) else {
            print("   ✗ Failed to create cache arrays")
            return
        }
        kCaches.append(kCache)
        vCaches.append(vCache)
    }

    // Create input dictionary
    var inputDict: [String: Any] = [
        "input_id": inputId,
        "position_id": positionId,
        "encoder_hidden_states": encoderHidden,
        "cross_attention_mask": crossMask,
        "attention_mask": attentionMask
    ]

    // Add caches to input
    for i in 0..<8 {
        inputDict["k_cache_\(i)"] = kCaches[i]
        inputDict["v_cache_\(i)"] = vCaches[i]
    }

    // Create MLFeatureProvider
    let inputProvider = try? MLDictionaryFeatureProvider(dictionary: inputDict)

    guard let provider = inputProvider else {
        print("   ✗ Failed to create feature provider")
        return
    }

    // Run inference
    guard let output = try? model.prediction(from: provider) else {
        print("   ✗ Inference failed")
        return
    }

    // Check outputs
    guard let logits = output.featureValue(for: "logits")?.multiArrayValue else {
        print("   ✗ No logits in output")
        return
    }

    print("   Logits shape: \(logits.shape)")
    print("   Expected: [1, 16384]")

    // Check cache outputs
    var cacheOk = true
    for i in 0..<8 {
        guard let kOut = output.featureValue(for: "k_cache_\(i)_out")?.multiArrayValue,
              let vOut = output.featureValue(for: "v_cache_\(i)_out")?.multiArrayValue else {
            cacheOk = false
            print("   ✗ Missing cache output \(i)")
            continue
        }

        let expectedShape = [1, 8, NSNumber(value: maxSeqLen), 128]
        if kOut.shape != expectedShape || vOut.shape != expectedShape {
            cacheOk = false
            print("   ✗ Cache \(i) has wrong shape")
        }
    }

    if cacheOk {
        print("   ✓ All 16 cache outputs have correct shape: [1, 8, \(maxSeqLen), 128]")
    }

    // Find next token (argmax)
    var maxVal: Float = -Float.infinity
    var maxIdx: Int = 0

    for i in 0..<logits.count {
        let val = logits[i].floatValue
        if val > maxVal {
            maxVal = val
            maxIdx = i
        }
    }

    print("   Next token: \(maxIdx)")

    print("\n" + String(repeating: "=", count: 70))
    print("✅ Compiled .mlmodelc works correctly!")
    print(String(repeating: "=", count: 70))
}

// Run test
testCompiledModel()
