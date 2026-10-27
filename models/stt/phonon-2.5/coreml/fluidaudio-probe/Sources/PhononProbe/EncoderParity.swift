import CoreML
import FluidAudio
import Foundation

extension PhononProbe {
    // Feed identical saved mel features through the encoder loaded by FluidAudio.
    // This separates encoder numerics from preprocessing and TDT decoding.
    static func encoderParity(_ args: [String]) throws {
        guard args.count == 4 else { throw ProbeError.usage }
        let units: MLComputeUnits
        switch args[1] {
        case "ane": units = .cpuAndNeuralEngine
        case "gpu": units = .cpuAndGPU
        case "cpu": units = .cpuOnly
        default: throw ProbeError.usage
        }
        let fixture = URL(fileURLWithPath: args[2])
        let metadata = try JSONSerialization.jsonObject(
            with: Data(contentsOf: fixture.appendingPathComponent("metadata.json"))
        ) as! [String: Any]
        let melShape = metadata["mel_shape"] as! [Int]
        let mel = try MLMultiArray(shape: melShape.map(NSNumber.init(value:)), dataType: .float32)
        let melData = try floats(fixture.appendingPathComponent("mel.f32"))
        let expected = try floats(fixture.appendingPathComponent("encoder.f32"))
        guard mel.count == melData.count else { throw ProbeError.usage }
        for i in melData.indices { mel[i] = NSNumber(value: melData[i]) }
        let length = try MLMultiArray(shape: [1], dataType: .int32)
        length[0] = NSNumber(value: metadata["mel_length"] as! Int)
        let models = try AsrModels.loadLocal(
            from: URL(fileURLWithPath: args[0]), version: .v3, encoderComputeUnits: units
        )
        guard let encoder = models.encoder else { throw ProbeError.usage }
        let input = try MLDictionaryFeatureProvider(dictionary: ["mel": mel, "mel_length": length])
        let output = try encoder.prediction(from: input)
        guard let actual = output.featureValue(for: "encoder")?.multiArrayValue,
              actual.shape.map({ $0.intValue }) == metadata["encoder_shape"] as! [Int],
              actual.count == expected.count,
              let actualLength = output.featureValue(for: "encoder_length")?.multiArrayValue,
              actualLength[0].intValue == metadata["encoder_length"] as! Int
        else { throw ProbeError.usage }
        var dot = 0.0, normActual = 0.0, normExpected = 0.0, error = 0.0
        var values: [Float] = []
        values.reserveCapacity(actual.count)
        for i in expected.indices {
            let a = actual[i].doubleValue, b = Double(expected[i])
            guard a.isFinite && b.isFinite else { throw ProbeError.usage }
            values.append(Float(a))
            dot += a * b
            normActual += a * a
            normExpected += b * b
            error += (a - b) * (a - b)
        }
        let cosine = dot / sqrt(normActual * normExpected)
        let nrmse = sqrt(error / normExpected)
        let report: [String: Any] = [
            "fluid_audio": "0.17.7", "units": args[1], "cosine": cosine, "nrmse": nrmse,
            "encoder_frames": actualLength[0].intValue,
            "gate_passed": cosine > 0.999 && nrmse < 0.05,
            "loaded_encoder_compute_units": encoder.configuration.computeUnits.rawValue,
            "allow_low_precision_accumulation_on_gpu": encoder.configuration.allowLowPrecisionAccumulationOnGPU
        ]
        let target = URL(fileURLWithPath: args[3])
        try values.withUnsafeBytes { try Data($0).write(to: target.appendingPathExtension("f32")) }
        try JSONSerialization.data(withJSONObject: report, options: [.prettyPrinted, .sortedKeys])
            .write(to: target, options: .atomic)
        print(String(data: try JSONSerialization.data(withJSONObject: report, options: [.sortedKeys]), encoding: .utf8)!)
    }

    private static func floats(_ url: URL) throws -> [Float] {
        let data = try Data(contentsOf: url)
        guard data.count % MemoryLayout<Float>.size == 0 else { throw ProbeError.usage }
        return data.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
    }
}
