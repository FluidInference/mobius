import CryptoKit
import FluidAudio
import Foundation

struct Model: Decodable {
    let name: String
    let path: String
}
struct Configuration: Decodable {
    let dataset: String
    let datasetSHA256: String
    let rows: Int
    let models: [Model]
    let trace: String
}
struct Row: Decodable {
    let context: String
    let options: [String]
    let label: Int
}
struct Record: Encodable {
    let row: Int
    let model: String
    let label: Int
    let selected: Int
    let rawSelected: Int
    let probabilities: [Float]
    let rawProbabilities: [Float]
    let logits: [Float]
}
struct ValidationError: Error { let message: String }
func require(_ value: Bool, _ message: String) throws {
    guard value else { throw ValidationError(message: message) }
}
let config = try JSONDecoder().decode(
    Configuration.self, from: Data(contentsOf: URL(fileURLWithPath: CommandLine.arguments[1])))
let data = try Data(contentsOf: URL(fileURLWithPath: config.dataset))
let digest = SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
try require(digest == config.datasetSHA256, "Dataset hash mismatch")
let rows = try data.split(separator: 10).map { try JSONDecoder().decode(Row.self, from: Data($0)) }
try require(rows.count == config.rows, "Dataset row count mismatch")
ModelHub.offlineMode = true
var models: [(String, CuaS1FormsManager)] = []
for model in config.models {
    let manager = try await CuaS1FormsManager.load(from: URL(fileURLWithPath: model.path))
    models.append((model.name, manager))
}
try require(!FileManager.default.fileExists(atPath: config.trace), "Use a fresh trace path")
try require(FileManager.default.createFile(atPath: config.trace, contents: nil), "Cannot create trace")
let trace = try FileHandle(forWritingTo: URL(fileURLWithPath: config.trace))
defer { try? trace.close() }
let encoder = JSONEncoder()
for (index, row) in rows.enumerated() {
    for (name, manager) in models {
        let result = try await manager.score(context: row.context, options: row.options)
        try require(!result.contextWasTruncated && result.truncatedOptionIndices.isEmpty, "Input truncated")
        try require(result.probabilities.count == row.options.count, "Incorrect output count")
        try require(result.rawProbabilities.count == row.options.count, "Incorrect raw output count")
        try require(abs(result.probabilities.reduce(0, +) - 1) <= 0.001, "Runtime probability sum failed")
        try require(result.selectedOption == row.options[result.selectedIndex], "Selected string changed")
        var rawSelected = 0
        for option in result.rawProbabilities.indices.dropFirst()
        where result.rawProbabilities[option] > result.rawProbabilities[rawSelected] {
            rawSelected = option
        }
        let record = Record(
            row: index, model: name, label: row.label, selected: result.selectedIndex, rawSelected: rawSelected,
            probabilities: result.probabilities, rawProbabilities: result.rawProbabilities, logits: result.logits)
        try trace.write(contentsOf: encoder.encode(record))
        try trace.write(contentsOf: Data([10]))
    }
    if (index + 1).isMultiple(of: 1000) || index + 1 == rows.count {
        print("Swift runtime: \(index + 1)/\(rows.count) rows, \(models.count) models", terminator: "\n")
    }
}
