import CoreML
import FluidAudio
import Foundation

@main
struct PhononProbe {
    static func main() async throws {
        let args = Array(CommandLine.arguments.dropFirst())
        if args.first == "encoder-parity" {
            try encoderParity(Array(args.dropFirst()))
            return
        }
        guard args.count >= 4 else {
            throw ProbeError.usage
        }
        let units: MLComputeUnits
        switch args[1] {
        case "cpu": units = .cpuOnly
        case "ane": units = .cpuAndNeuralEngine
        case "gpu": units = .cpuAndGPU
        default: throw ProbeError.usage
        }
        let started = Date()
        // Load through the unmodified FluidAudio backend. The checkpoint has
        // the v3 token/blank/decoder contract; do not download stock weights.
        let models = try AsrModels.loadLocal(
            from: URL(fileURLWithPath: args[0]), version: .v3,
            encoderComputeUnits: units
        )
        let manager = AsrManager(config: .default)
        try await manager.loadModels(models)
        let loadSeconds = Date().timeIntervalSince(started)
        var rows: [[String: Any]] = []
        for path in args.dropFirst(3) {
            var state = TdtDecoderState.make(decoderLayers: await manager.decoderLayerCount)
            let start = Date()
            let result = try await manager.transcribe(URL(fileURLWithPath: path), decoderState: &state)
            rows.append([
                "file": path, "text": result.text, "audio_seconds": result.duration,
                "wall_seconds": Date().timeIntervalSince(start)
            ])
        }
        let report: [String: Any] = [
            "fluid_audio": "0.17.7", "units": args[1], "load_seconds": loadSeconds, "rows": rows
        ]
        let data = try JSONSerialization.data(withJSONObject: report, options: [.prettyPrinted, .sortedKeys])
        try data.write(to: URL(fileURLWithPath: args[2]), options: .atomic)
    }
    enum ProbeError: Error {
        case usage // phonon-probe MODEL_DIR cpu|ane|gpu REPORT.json AUDIO...
    }
}
