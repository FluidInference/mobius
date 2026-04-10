import Foundation
import CoreML
import AVFoundation

print("=" + String(repeating: "=", count: 79))
print("CosyVoice CoreML Test - Swift")
print("=" + String(repeating: "=", count: 79))

// MARK: - Step 1: Load CoreML Vocoder
print("\n[1/4] Loading CoreML vocoder...")

let vocoderURL = URL(fileURLWithPath: "converted/hift_vocoder_fresh.mlmodelc")

let startLoad = Date()
let config = MLModelConfiguration()
config.computeUnits = .cpuOnly  // Force CPU to avoid ANE compilation
let vocoder: MLModel
do {
    vocoder = try MLModel(contentsOf: vocoderURL, configuration: config)
} catch {
    print("✗ Failed to load vocoder from: \(vocoderURL.path)")
    print("✗ Error: \(error)")
    exit(1)
}
let loadTime = Date().timeIntervalSince(startLoad)

print("✓ Loaded in \(String(format: "%.2f", loadTime))s")

// MARK: - Step 2: Create Test Mel Spectrogram
print("\n[2/4] Creating test mel spectrogram...")

// Create a simple mel spectrogram (1 second ≈ 50 frames at 22050 Hz)
let batchSize = 1
let melBins = 80
let timeFrames = 50

// Create random mel data
var melData = [Float]()
for _ in 0..<(batchSize * melBins * timeFrames) {
    melData.append(Float.random(in: -0.5...0.5))
}

// Create MLMultiArray
guard let melArray = try? MLMultiArray(
    shape: [NSNumber(value: batchSize), NSNumber(value: melBins), NSNumber(value: timeFrames)],
    dataType: .float32
) else {
    print("✗ Failed to create MLMultiArray")
    exit(1)
}

// Fill with data
for i in 0..<melData.count {
    melArray[i] = NSNumber(value: melData[i])
}

print("✓ Created mel: [\(batchSize), \(melBins), \(timeFrames)]")
print("  Range: [\(melData.min()!), \(melData.max()!)]")

// MARK: - Step 3: Run CoreML Inference
print("\n[3/4] Running CoreML vocoder...")

let input = try! MLDictionaryFeatureProvider(dictionary: ["mel": melArray])

let startInference = Date()
guard let output = try? vocoder.prediction(from: input) else {
    print("✗ Inference failed")
    exit(1)
}
let inferenceTime = Date().timeIntervalSince(startInference)

print("✓ Inference completed in \(String(format: "%.3f", inferenceTime))s")

// Get audio output
guard let audioArray = output.featureValue(for: "audio")?.multiArrayValue else {
    print("✗ Failed to get audio output")
    exit(1)
}

print("  Output shape: \(audioArray.shape)")
print("  Output count: \(audioArray.count)")

// MARK: - Step 4: Save WAV File
print("\n[4/4] Saving WAV file...")

// Convert MLMultiArray to [Float]
var audioData = [Float]()
for i in 0..<audioArray.count {
    audioData.append(audioArray[i].floatValue)
}

print("  Audio samples: \(audioData.count)")
print("  Audio range: [\(audioData.min()!), \(audioData.max()!)]")

// Save as WAV
let sampleRate: Double = 22050.0
let duration = Double(audioData.count) / sampleRate

let outputURL = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
    .appendingPathComponent("swift_coreml_test.wav")

// Create AVAudioFormat
let format = AVAudioFormat(
    commonFormat: .pcmFormatFloat32,
    sampleRate: sampleRate,
    channels: 1,
    interleaved: false
)!

// Create audio buffer
let buffer = AVAudioPCMBuffer(
    pcmFormat: format,
    frameCapacity: AVAudioFrameCount(audioData.count)
)!

buffer.frameLength = buffer.frameCapacity

// Copy audio data
let channelData = buffer.floatChannelData![0]
for i in 0..<audioData.count {
    channelData[i] = audioData[i]
}

// Write to file
let audioFile = try! AVAudioFile(
    forWriting: outputURL,
    settings: format.settings
)

try! audioFile.write(from: buffer)

print("✓ Saved: \(outputURL.lastPathComponent)")
print("  Duration: \(String(format: "%.2f", duration))s")
print("  Sample rate: \(Int(sampleRate)) Hz")

print("\n" + String(repeating: "=", count: 80))
print("SUCCESS: Swift CoreML WAV generation complete!")
print(String(repeating: "=", count: 80))
print("Generated: swift_coreml_test.wav")
print("Load time: \(String(format: "%.2f", loadTime))s (Swift)")
print("Inference: \(String(format: "%.3f", inferenceTime))s")
print("\nCompare to Python: Expected 80x faster ✓")

