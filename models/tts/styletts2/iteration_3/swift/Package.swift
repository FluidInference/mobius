// swift-tools-version:5.10
import PackageDescription

let package = Package(
    name: "Iter3Bench",
    platforms: [.macOS(.v14)],
    products: [
        .executable(name: "iter3-bench", targets: ["Iter3Bench"]),
        .executable(name: "iter3-tts",   targets: ["Iter3TTS"]),
    ],
    targets: [
        .executableTarget(name: "Iter3Bench", path: "Sources/Iter3Bench"),
        .executableTarget(name: "Iter3TTS",   path: "Sources/Iter3TTS"),
    ]
)
