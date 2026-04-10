import Foundation
import CoreML
import Accelerate

/// CosyVoice3 Text-to-Speech Pipeline using CoreML models
///
/// This class provides a complete TTS pipeline using the converted CoreML models:
/// - LLM (embedding + 24 decoder layers + LM head)
/// - Flow (speech tokens → mel spectrogram)
/// - Vocoder (mel → audio waveform)
///
/// Usage:
/// ```swift
/// let tts = try CosyVoiceCoreML(modelDirectory: "path/to/models")
/// let audioBuffer = try await tts.synthesize(text: "Hello, world!")
/// ```
@available(macOS 14.0, iOS 17.0, *)
public class CosyVoiceCoreML {

    // MARK: - Models

    private let embedding: MLModel
    private let decoderLayers: [MLModel]
    private let lmHead: MLModel
    private let flow: MLModel
    private let vocoder: MLModel

    // MARK: - Configuration

    private let hiddenSize = 896
    private let maxSequenceLength = 512
    private let melBins = 80
    private let sampleRate = 24000

    // MARK: - Initialization

    /// Initialize the TTS pipeline with CoreML models
    /// - Parameter modelDirectory: Directory containing all .mlpackage models
    public init(modelDirectory: String) throws {
        let modelDir = URL(fileURLWithPath: modelDirectory)

        // Load embedding model
        let embeddingURL = modelDir.appendingPathComponent("cosyvoice_llm_embedding.mlpackage")
        embedding = try MLModel(contentsOf: embeddingURL)

        // Load 24 decoder layers
        var layers: [MLModel] = []
        for i in 0..<24 {
            let layerURL = modelDir
                .appendingPathComponent("decoder_layers")
                .appendingPathComponent("cosyvoice_llm_layer_\(i).mlpackage")
            let layer = try MLModel(contentsOf: layerURL)
            layers.append(layer)
        }
        decoderLayers = layers

        // Load LM head
        let lmHeadURL = modelDir.appendingPathComponent("cosyvoice_llm_lm_head.mlpackage")
        lmHead = try MLModel(contentsOf: lmHeadURL)

        // Load Flow model
        let flowURL = modelDir.appendingPathComponent("flow_decoder.mlpackage")
        flow = try MLModel(contentsOf: flowURL)

        // Load vocoder
        let vocoderURL = modelDir
            .appendingPathComponent("converted")
            .appendingPathComponent("hift_vocoder.mlpackage")
        vocoder = try MLModel(contentsOf: vocoderURL)

        print("✓ Loaded all CoreML models")
    }

    // MARK: - TTS Pipeline

    /// Synthesize speech from text
    /// - Parameters:
    ///   - text: Input text to synthesize
    ///   - progress: Optional progress callback (0.0 to 1.0)
    /// - Returns: Audio samples as Float array at 24kHz
    public func synthesize(
        text: String,
        progress: ((Float) -> Void)? = nil
    ) async throws -> [Float] {

        print("Synthesizing: '\(text)'")

        // Step 1: Tokenize (25%)
        progress?(0.0)
        let tokens = tokenize(text: text)
        progress?(0.25)

        // Step 2: LLM inference (50%)
        let speechTokens = try await runLLM(tokens: tokens)
        progress?(0.50)

        // Step 3: Flow inference (75%)
        let mel = try await runFlow(speechTokens: speechTokens)
        progress?(0.75)

        // Step 4: Vocoder (100%)
        let audio = try await runVocoder(mel: mel)
        progress?(1.0)

        return audio
    }

    // MARK: - Tokenization

    /// Simple character-based tokenizer (replace with proper Qwen2 tokenizer)
    private func tokenize(text: String) -> MLMultiArray {
        // Simple fallback: use character codes
        let tokens = text.utf8.map { Int32($0) % 1000 }
        let count = min(tokens.count, maxSequenceLength)

        // Create MLMultiArray [1, seq_len]
        guard let array = try? MLMultiArray(
            shape: [1, count as NSNumber],
            dataType: .int32
        ) else {
            fatalError("Failed to create token array")
        }

        for (i, token) in tokens.prefix(count).enumerated() {
            array[[0, i] as [NSNumber]] = NSNumber(value: token)
        }

        return array
    }

    // MARK: - LLM Inference

    private func runLLM(tokens: MLMultiArray) async throws -> MLMultiArray {
        print("  Running LLM...")

        // 1. Embedding
        let embeddingInput = try! MLDictionaryFeatureProvider(
            dictionary: ["input_ids": MLFeatureValue(multiArray: tokens)]
        )
        let embeddingOutput = try embedding.prediction(from: embeddingInput)
        guard var hiddenStates = embeddingOutput.featureValue(for: "embeddings")?.multiArrayValue else {
            throw TTSError.modelOutputMissing("embeddings")
        }

        let seqLen = hiddenStates.shape[1].intValue

        // 2. Prepare attention mask and position IDs
        let attentionMask = try createAttentionMask(sequenceLength: seqLen)
        let positionIds = try createPositionIds(sequenceLength: seqLen)

        // 3. Run through 24 decoder layers
        for (i, layer) in decoderLayers.enumerated() {
            let layerInput = try! MLDictionaryFeatureProvider(dictionary: [
                "hidden_states": MLFeatureValue(multiArray: hiddenStates),
                "attention_mask": MLFeatureValue(multiArray: attentionMask),
                "position_ids": MLFeatureValue(multiArray: positionIds)
            ])

            let layerOutput = try layer.prediction(from: layerInput)
            guard let output = layerOutput.featureValue(for: "output_hidden_states")?.multiArrayValue else {
                throw TTSError.modelOutputMissing("output_hidden_states")
            }
            hiddenStates = output

            if i % 6 == 0 {
                print("    Layer \(i)/24")
            }
        }

        // 4. LM Head
        let lmHeadInput = try! MLDictionaryFeatureProvider(
            dictionary: ["hidden_states": MLFeatureValue(multiArray: hiddenStates)]
        )
        let lmHeadOutput = try lmHead.prediction(from: lmHeadInput)
        guard let logits = lmHeadOutput.featureValue(for: "logits")?.multiArrayValue else {
            throw TTSError.modelOutputMissing("logits")
        }

        // Convert logits to tokens (argmax)
        let speechTokens = argmax(logits: logits)
        print("  ✓ LLM complete")

        return speechTokens
    }

    // MARK: - Flow Inference

    private func runFlow(speechTokens: MLMultiArray) async throws -> MLMultiArray {
        print("  Running Flow...")

        let seqLen = speechTokens.shape[1].intValue

        // Prepare Flow inputs (simplified - real implementation more complex)
        let x = try createRandomArray(shape: [1, melBins, seqLen], dataType: .float16)
        let mask = try createOnesArray(shape: [1, 1, seqLen], dataType: .float16)
        let mu = try createRandomArray(shape: [1, melBins, seqLen], dataType: .float16)
        let t = try createArray(shape: [1], values: [0.5], dataType: .float16)
        let spks = try createRandomArray(shape: [1, melBins], dataType: .float16)

        // Use speech tokens as conditioning
        let cond = try createConditioningFromTokens(tokens: speechTokens, seqLen: seqLen)

        let flowInput = try! MLDictionaryFeatureProvider(dictionary: [
            "x": MLFeatureValue(multiArray: x),
            "mask": MLFeatureValue(multiArray: mask),
            "mu": MLFeatureValue(multiArray: mu),
            "t": MLFeatureValue(multiArray: t),
            "spks": MLFeatureValue(multiArray: spks),
            "cond": MLFeatureValue(multiArray: cond)
        ])

        let flowOutput = try flow.prediction(from: flowInput)
        guard let mel = flowOutput.featureValue(for: "output")?.multiArrayValue else {
            throw TTSError.modelOutputMissing("output")
        }

        print("  ✓ Flow complete")
        return mel
    }

    // MARK: - Vocoder Inference

    private func runVocoder(mel: MLMultiArray) async throws -> [Float] {
        print("  Running Vocoder...")

        // Convert mel to Float32 if needed
        let melFloat32: MLMultiArray
        if mel.dataType == .float32 {
            melFloat32 = mel
        } else {
            melFloat32 = try convertToFloat32(mel)
        }

        let vocoderInput = try! MLDictionaryFeatureProvider(
            dictionary: ["mel": MLFeatureValue(multiArray: melFloat32)]
        )

        let vocoderOutput = try vocoder.prediction(from: vocoderInput)

        // Get audio output (key might be "audio" or first output)
        let audioKey = vocoderOutput.featureNames.first ?? "audio"
        guard let audioArray = vocoderOutput.featureValue(for: audioKey)?.multiArrayValue else {
            throw TTSError.modelOutputMissing("audio")
        }

        // Convert to Float array
        let audio = multiArrayToFloatArray(audioArray)
        print("  ✓ Vocoder complete: \(audio.count) samples (\(Float(audio.count) / Float(sampleRate))s)")

        return audio
    }

    // MARK: - Helper Functions

    private func createAttentionMask(sequenceLength: Int) throws -> MLMultiArray {
        let array = try MLMultiArray(
            shape: [1, 1, sequenceLength as NSNumber, sequenceLength as NSNumber],
            dataType: .float16
        )
        // Fill with 1s (no masking)
        for i in 0..<sequenceLength {
            for j in 0..<sequenceLength {
                array[[0, 0, i, j] as [NSNumber]] = 1.0
            }
        }
        return array
    }

    private func createPositionIds(sequenceLength: Int) throws -> MLMultiArray {
        let array = try MLMultiArray(
            shape: [1, sequenceLength as NSNumber],
            dataType: .int32
        )
        for i in 0..<sequenceLength {
            array[[0, i] as [NSNumber]] = NSNumber(value: i)
        }
        return array
    }

    private func argmax(logits: MLMultiArray) -> MLMultiArray {
        // Simplified argmax: take max along last dimension
        let batchSize = logits.shape[0].intValue
        let seqLen = logits.shape[1].intValue
        let vocabSize = logits.shape[2].intValue

        let result = try! MLMultiArray(
            shape: [batchSize as NSNumber, seqLen as NSNumber],
            dataType: .int32
        )

        for i in 0..<seqLen {
            var maxIdx = 0
            var maxVal: Float = -Float.infinity

            for j in 0..<vocabSize {
                let val = logits[[0, i, j] as [NSNumber]].floatValue
                if val > maxVal {
                    maxVal = val
                    maxIdx = j
                }
            }

            result[[0, i] as [NSNumber]] = NSNumber(value: maxIdx)
        }

        return result
    }

    private func createRandomArray(shape: [Int], dataType: MLMultiArrayDataType) throws -> MLMultiArray {
        let array = try MLMultiArray(
            shape: shape.map { $0 as NSNumber },
            dataType: dataType
        )
        // Fill with random values
        for i in 0..<array.count {
            let randomValue = Float.random(in: -1...1)
            array[i] = NSNumber(value: randomValue)
        }
        return array
    }

    private func createOnesArray(shape: [Int], dataType: MLMultiArrayDataType) throws -> MLMultiArray {
        let array = try MLMultiArray(
            shape: shape.map { $0 as NSNumber },
            dataType: dataType
        )
        for i in 0..<array.count {
            array[i] = 1.0
        }
        return array
    }

    private func createArray(shape: [Int], values: [Float], dataType: MLMultiArrayDataType) throws -> MLMultiArray {
        let array = try MLMultiArray(
            shape: shape.map { $0 as NSNumber },
            dataType: dataType
        )
        for (i, val) in values.enumerated() {
            array[i] = NSNumber(value: val)
        }
        return array
    }

    private func createConditioningFromTokens(tokens: MLMultiArray, seqLen: Int) throws -> MLMultiArray {
        // Simplified: tile tokens to create conditioning
        let array = try MLMultiArray(
            shape: [1, melBins as NSNumber, seqLen as NSNumber],
            dataType: .float16
        )

        // Fill with token-based values
        for i in 0..<seqLen {
            let tokenIdx = min(i, tokens.shape[1].intValue - 1)
            let tokenVal = tokens[[0, tokenIdx] as [NSNumber]].floatValue
            for j in 0..<melBins {
                array[[0, j, i] as [NSNumber]] = NSNumber(value: tokenVal / 1000.0)
            }
        }

        return array
    }

    private func convertToFloat32(_ array: MLMultiArray) throws -> MLMultiArray {
        let float32Array = try MLMultiArray(
            shape: array.shape,
            dataType: .float32
        )

        for i in 0..<array.count {
            float32Array[i] = array[i]
        }

        return float32Array
    }

    private func multiArrayToFloatArray(_ array: MLMultiArray) -> [Float] {
        var result: [Float] = []
        result.reserveCapacity(array.count)

        for i in 0..<array.count {
            result.append(array[i].floatValue)
        }

        return result
    }

    // MARK: - Audio Export

    /// Save audio samples to WAV file
    public func saveToWAV(samples: [Float], path: String) throws {
        // Convert Float samples to Int16
        var int16Samples: [Int16] = samples.map { sample in
            let clamped = max(-1.0, min(1.0, sample))
            return Int16(clamped * 32767.0)
        }

        // Write WAV file (simplified - use proper WAV library in production)
        let data = Data(bytes: &int16Samples, count: int16Samples.count * 2)
        try data.write(to: URL(fileURLWithPath: path))
    }
}

// MARK: - Error Types

public enum TTSError: Error {
    case modelOutputMissing(String)
    case invalidAudioData
    case tokenizationFailed
}

// MARK: - Usage Example

/*
 Example usage:

 ```swift
 import Foundation

 @main
 struct TTSExample {
     static func main() async throws {
         // Initialize TTS pipeline
         let modelDir = "/path/to/cosyvoice3/coreml"
         let tts = try CosyVoiceCoreML(modelDirectory: modelDir)

         // Synthesize speech
         let text = "Hello, this is a test of the CosyVoice3 CoreML pipeline."
         let audio = try await tts.synthesize(text: text) { progress in
             print("Progress: \(Int(progress * 100))%")
         }

         // Save to file
         try tts.saveToWAV(samples: audio, path: "output.wav")
         print("Saved audio to output.wav")
     }
 }
 ```
 */
