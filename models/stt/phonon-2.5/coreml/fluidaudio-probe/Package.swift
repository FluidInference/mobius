// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "PhononCoreMLProbe",
    platforms: [.macOS(.v15)],
    products: [.executable(name: "phonon-probe", targets: ["PhononProbe"])],
    dependencies: [.package(url: "https://github.com/FluidInference/FluidAudio.git", exact: "0.17.7")],
    targets: [.executableTarget(name: "PhononProbe", dependencies: [.product(name: "FluidAudio", package: "FluidAudio")])]
)
