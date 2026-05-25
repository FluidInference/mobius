import Foundation
import CoreML

@main
struct ComputePlanDiag {
    static func main() async throws {
        guard CommandLine.arguments.count >= 2 else {
            print("usage: compute_plan <model-dir>")
            return
        }
        let dir = URL(fileURLWithPath: CommandLine.arguments[1])
        let models = ["encoder.mlmodelc", "decoder.mlmodelc", "joint.mlmodelc",
                     "decoder_joint_argmax.mlmodelc", "joint_noencproj_batched.mlmodelc"]
        for name in models {
            let url = dir.appendingPathComponent(name)
            guard FileManager.default.fileExists(atPath: url.path) else { continue }
            print("=== \(name) ===")
            do {
                let cuArg = ProcessInfo.processInfo.environment["CU"] ?? "all"
                let cfg = MLModelConfiguration()
                switch cuArg {
                case "cpu_and_ne": cfg.computeUnits = .cpuAndNeuralEngine
                case "cpu": cfg.computeUnits = .cpuOnly
                case "cpu_and_gpu": cfg.computeUnits = .cpuAndGPU
                default: cfg.computeUnits = .all
                }
                print("  [config: \(cuArg)]")
                let plan = try await MLComputePlan.load(contentsOf: url, configuration: cfg)
                let program = plan.modelStructure
                if case .program(let prog) = program {
                    var anneCount = 0
                    var cpuCount = 0
                    var gpuCount = 0
                    var ops: [(String, String)] = []
                    for (_, function) in prog.functions {
                        for op in function.block.operations {
                            if op.operatorName == "const" { continue }
                            let usage = plan.deviceUsage(for: op)
                            let dev: String
                            if let preferred = usage?.preferred {
                                let desc = String(describing: preferred)
                                if desc.contains("CPU") || desc.contains("cpu") { dev = "CPU"; cpuCount += 1 }
                                else if desc.contains("GPU") || desc.contains("gpu") { dev = "GPU"; gpuCount += 1 }
                                else if desc.contains("Neural") || desc.contains("neural") || desc.contains("ANE") { dev = "ANE"; anneCount += 1 }
                                else { dev = "?(\(desc))"; }
                            } else {
                                dev = "?"
                            }
                            ops.append((op.operatorName, dev))
                            if ops.count <= 5 {
                                let supported = usage?.supported.map { String(describing: $0) } ?? []
                                let prefDesc: String
                                if let u = usage { prefDesc = String(describing: u.preferred) } else { prefDesc = "nil" }
                                print("    \(op.operatorName): preferred=\(prefDesc) supported=\(supported)")
                            }
                        }
                    }
                    print("  Total: ANE=\(anneCount) CPU=\(cpuCount) GPU=\(gpuCount)")
                    let cpuOps = ops.filter { $0.1 == "CPU" }
                    if !cpuOps.isEmpty {
                        print("  CPU ops:")
                        for (name, _) in cpuOps.prefix(20) {
                            print("    - \(name)")
                        }
                    }
                }
            } catch {
                print("  ERROR: \(error.localizedDescription)")
            }
        }
    }
}
