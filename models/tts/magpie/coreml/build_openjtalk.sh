#!/bin/bash
#
# Build OpenJTalk as an XCFramework for iOS and macOS.
#
# Usage: ./build_openjtalk.sh
#
# Prerequisites:
#   - Xcode with iOS SDK installed
#   - CMake (brew install cmake)
#   - Git
#
# Output: ios/OpenJTalk.xcframework/

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BUILD_ROOT="${SCRIPT_DIR}/build_openjtalk"
OPENJTALK_REPO="${BUILD_ROOT}/open_jtalk"
OPENJTALK_DIR="${OPENJTALK_REPO}/src"
WRAPPER_H="${SCRIPT_DIR}/openjtalk_wrapper.h"
WRAPPER_CPP="${SCRIPT_DIR}/openjtalk_wrapper.cpp"
OUTPUT_DIR="${SCRIPT_DIR}/ios/OpenJTalk.xcframework"

# Minimum deployment targets
IOS_DEPLOYMENT_TARGET="17.0"
MACOS_DEPLOYMENT_TARGET="14.0"

# Targets to build
TARGETS=(
    "ios-arm64"
    "ios-arm64-simulator"
    "macos-arm64"
)

echo "=== Building OpenJTalk XCFramework ==="

# Step 1: Clone open_jtalk if needed
if [ ! -d "$OPENJTALK_REPO" ]; then
    echo "Cloning r9y9/open_jtalk..."
    mkdir -p "$BUILD_ROOT"
    git clone --depth 1 https://github.com/r9y9/open_jtalk.git "$OPENJTALK_REPO"
else
    echo "Using existing open_jtalk at $OPENJTALK_REPO"
fi

# Step 2: Build for each target
build_target() {
    local target=$1
    local build_dir="${BUILD_ROOT}/build-${target}"
    local install_dir="${BUILD_ROOT}/install-${target}"

    echo ""
    echo "--- Building for ${target} ---"

    rm -rf "$build_dir"
    mkdir -p "$build_dir"

    local cmake_args=(
        -DCMAKE_BUILD_TYPE=Release
        -DBUILD_SHARED_LIBS=OFF
        -DCMAKE_INSTALL_PREFIX="$install_dir"
    )

    case "$target" in
        ios-arm64)
            local sdk_path
            sdk_path=$(xcrun --sdk iphoneos --show-sdk-path)
            cmake_args+=(
                -DCMAKE_SYSTEM_NAME=iOS
                -DCMAKE_OSX_ARCHITECTURES=arm64
                -DCMAKE_OSX_DEPLOYMENT_TARGET="$IOS_DEPLOYMENT_TARGET"
                -DCMAKE_OSX_SYSROOT="$sdk_path"
                -DCMAKE_C_FLAGS="-fembed-bitcode"
                -DCMAKE_CXX_FLAGS="-fembed-bitcode"
            )
            ;;
        ios-arm64-simulator)
            local sdk_path
            sdk_path=$(xcrun --sdk iphonesimulator --show-sdk-path)
            cmake_args+=(
                -DCMAKE_SYSTEM_NAME=iOS
                -DCMAKE_OSX_ARCHITECTURES=arm64
                -DCMAKE_OSX_DEPLOYMENT_TARGET="$IOS_DEPLOYMENT_TARGET"
                -DCMAKE_OSX_SYSROOT="$sdk_path"
                -DCMAKE_C_FLAGS="-target arm64-apple-ios${IOS_DEPLOYMENT_TARGET}-simulator"
                -DCMAKE_CXX_FLAGS="-target arm64-apple-ios${IOS_DEPLOYMENT_TARGET}-simulator"
            )
            ;;
        macos-arm64)
            cmake_args+=(
                -DCMAKE_SYSTEM_NAME=Darwin
                -DCMAKE_OSX_ARCHITECTURES=arm64
                -DCMAKE_OSX_DEPLOYMENT_TARGET="$MACOS_DEPLOYMENT_TARGET"
            )
            ;;
    esac

    # Configure
    cmake -S "$OPENJTALK_DIR" -B "$build_dir" "${cmake_args[@]}"

    # Build
    cmake --build "$build_dir" --config Release -j "$(sysctl -n hw.ncpu)"

    # Find the built static library
    local lib_path
    lib_path=$(find "$build_dir" -name "libopenjtalk.a" | head -1)
    if [ -z "$lib_path" ]; then
        echo "ERROR: libopenjtalk.a not found in $build_dir"
        find "$build_dir" -name "*.a" -print
        exit 1
    fi

    # Compile the wrapper
    local wrapper_obj="${build_dir}/openjtalk_wrapper.o"
    local include_dirs=(
        -I"${OPENJTALK_DIR}/njd"
        -I"${OPENJTALK_DIR}/mecab/src"
        -I"${OPENJTALK_DIR}/jpcommon"
        -I"${OPENJTALK_DIR}/text2mecab"
        -I"${OPENJTALK_DIR}/mecab2njd"
        -I"${OPENJTALK_DIR}/njd2jpcommon"
        -I"${OPENJTALK_DIR}/njd_set_pronunciation"
        -I"${OPENJTALK_DIR}/njd_set_digit"
        -I"${OPENJTALK_DIR}/njd_set_accent_phrase"
        -I"${OPENJTALK_DIR}/njd_set_accent_type"
        -I"${OPENJTALK_DIR}/njd_set_unvoiced_vowel"
        -I"${OPENJTALK_DIR}/njd_set_long_vowel"
    )

    local cc_flags=()
    case "$target" in
        ios-arm64)
            cc_flags+=(-isysroot "$(xcrun --sdk iphoneos --show-sdk-path)")
            cc_flags+=(-target arm64-apple-ios${IOS_DEPLOYMENT_TARGET})
            ;;
        ios-arm64-simulator)
            cc_flags+=(-isysroot "$(xcrun --sdk iphonesimulator --show-sdk-path)")
            cc_flags+=(-target arm64-apple-ios${IOS_DEPLOYMENT_TARGET}-simulator)
            ;;
        macos-arm64)
            cc_flags+=(-target arm64-apple-macos${MACOS_DEPLOYMENT_TARGET})
            ;;
    esac

    xcrun clang++ -c "$WRAPPER_CPP" -o "$wrapper_obj" \
        "${include_dirs[@]}" "${cc_flags[@]}" \
        -std=c++17 -O2

    # Merge wrapper into static library
    local merged_lib="${build_dir}/libopenjtalk_merged.a"
    libtool -static -o "$merged_lib" "$lib_path" "$wrapper_obj"

    # Store for xcframework creation
    mkdir -p "$install_dir"
    cp "$merged_lib" "$install_dir/libopenjtalk.a"

    echo "Built: $install_dir/libopenjtalk.a"
}

for target in "${TARGETS[@]}"; do
    build_target "$target"
done

# Step 3: Create xcframework (static libs only, no headers — headers are in COpenJTalk/)
echo ""
echo "--- Creating XCFramework ---"

# Remove old xcframework
rm -rf "$OUTPUT_DIR"

# Create xcframework without headers (avoids module.modulemap collision with NemoTextProcessing)
# Headers are provided separately via ios/COpenJTalk/ and SWIFT_INCLUDE_PATHS
xcodebuild -create-xcframework \
    -library "${BUILD_ROOT}/install-ios-arm64/libopenjtalk.a" \
    -library "${BUILD_ROOT}/install-ios-arm64-simulator/libopenjtalk.a" \
    -library "${BUILD_ROOT}/install-macos-arm64/libopenjtalk.a" \
    -output "$OUTPUT_DIR"

# Step 4: Create COpenJTalk module directory for Swift import
COPENJTALK_DIR="${SCRIPT_DIR}/ios/COpenJTalk"
mkdir -p "$COPENJTALK_DIR"
cp "$WRAPPER_H" "$COPENJTALK_DIR/openjtalk_wrapper.h"
cat > "$COPENJTALK_DIR/module.modulemap" << 'EOF'
module COpenJTalk {
    header "openjtalk_wrapper.h"
    export *
}
EOF

echo ""
echo "=== XCFramework created at: $OUTPUT_DIR ==="
echo "=== COpenJTalk module at: $COPENJTALK_DIR ==="
echo ""

# Verify
echo "Contents:"
find "$OUTPUT_DIR" -type f | sort
echo ""
echo "Architecture check:"
for target in "${TARGETS[@]}"; do
    lib=$(find "$OUTPUT_DIR" -path "*${target}*" -name "libopenjtalk.a" | head -1)
    if [ -n "$lib" ]; then
        echo "  ${target}: $(lipo -info "$lib" 2>/dev/null)"
    fi
done
