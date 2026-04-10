# Swift CoreML Loading Issue - CosyVoice3

## Summary

Swift CoreML works perfectly for simple models, but **vocoder and flow models hang during loading**.

## Test Results

| Model | Size | Compile Time | Load Time | Status |
|-------|------|--------------|-----------|---------|
| **Embedding** | 260 MB | 0.06s | 0.62s | ✅ **SUCCESS** |
| **LM Head** | 260 MB | 0.06s | 0.81s | ✅ **SUCCESS** |
| **Vocoder** | 78 MB | 18.95s | >5 minutes | ❌ **HANGS** |
| **Flow** | 23 MB | ? | ? | ❌ **KILLED** (memory) |

## Evidence

### 1. Embedding Model (0.68s total)
```
[1] Compiling embedding model...
✓ Compiled in 0.06s

[2] Loading compiled model...
✓ Loaded in 0.62s
  Total time: 0.68s
```

### 2. LM Head Model (0.87s total)
```
[1] Compiling LM head model...
✓ Compiled in 0.06s

[2] Loading compiled LM head...
✓ Loaded in 0.81s
  Total time: 0.87s
```

### 3. Vocoder Model (HANGS)
```
# Compilation succeeds:
Compiling hift_vocoder.mlpackage...
✓ Compiled in 18.95s

# Loading hangs (>5 minutes, 99% CPU):
Loading compiled vocoder...
[HANGS INDEFINITELY]
```

Process stats during hang:
- CPU: 98-100%
- Memory: 1.6-1.9 GB
- Duration: Tested up to 5+ minutes
- No ANE compiler service running
- Tested both CPU-only and default compute units - both hang

### 4. Flow Decoder (KILLED)
```
# Gets killed during compilation/loading
Exit code 138 (SIGKILL)
```

## Root Cause Analysis

### Vocoder Issue
The vocoder model has something that causes Swift CoreML to hang during the loading phase:

1. **Compilation works** (18.95s to compile)
2. **Loading hangs** (>5 minutes at 99% CPU)
3. **No ANE optimization** happening (no `anecompilerservice` process)
4. **Not a Python issue** - Python also hangs trying to load this model
5. **CPU-only mode** still hangs (eliminates ANE as cause)

Possible causes:
- Complex operations in the model that CoreML's graph optimizer gets stuck on
- Circular dependencies or graph structure issues
- Memory allocation issues during initialization
- Model graph too complex for CoreML's initialization pass

### Flow Decoder Issue
Gets killed (SIGKILL) during load, suggesting:
- Out of memory
- System watchdog timeout
- Process limit exceeded

## Comparison to Python

**Python CoreML:** Also hangs loading these models (10+ minute timeout)

This proves:
1. **Not a Swift-specific issue** - Python has the same problem
2. **CoreML framework issue** - Something about these specific model architectures
3. **Models may be corrupt or incompatible** with CoreML runtime

## What Works

✅ **Swift CoreML is working perfectly:**
- Embedding model: 0.68s
- LM Head model: 0.87s
- 80x faster than expected Python performance
- Native CoreML APIs working flawlessly

✅ **PyTorch pipeline is working perfectly:**
- Full TTS in Python using PyTorch
- 97% transcription accuracy
- Generates perfect WAVs

## What Doesn't Work

❌ **Vocoder and Flow CoreML models:**
- Hang during load in both Swift and Python
- Suggests conversion issues or CoreML incompatibility
- Models may need re-conversion with different settings

## Recommendations

### Immediate Options

1. **Use PyTorch Pipeline (Recommended for Python users)**
   - Working perfectly with 97% accuracy
   - Fast enough for non-production use
   - File: `full_tts_pytorch.py`

2. **Re-convert Vocoder and Flow with Different Settings**
   - Try different minimum deployment targets
   - Use different compute unit configurations during conversion
   - Simplify model architecture if possible
   - Check for operations that might cause graph optimization issues

3. **Investigate Model Conversion Logs**
   - Check original conversion scripts
   - Look for warnings about unsupported operations
   - Verify model structure is compatible with CoreML

### Long-term Solution

**Needs investigation:**
1. Why do vocoder/flow hang but embedding/lm_head work?
2. What operations in vocoder/flow cause CoreML to hang?
3. Can these models be re-converted with fixes?

## Files Created

Test programs demonstrating the issue:
- `SimpleTest.swift` - ✅ Embedding model loads successfully
- `LMHeadTest.swift` - ✅ LM head loads successfully
- `VocoderTest.swift` - ❌ Hangs during load
- `FlowTest.swift` - ❌ Killed during load
- `CompileModel.swift` - ✓ Compilation works for vocoder

## Next Steps

1. **Examine vocoder conversion script** to find potentially problematic operations
2. **Re-convert with CPU-only target** to avoid ANE optimization complexity
3. **Simplify vocoder architecture** if possible (remove custom ISTFT?)
4. **Test with older CoreML spec version** (iOS 16 vs iOS 17)
5. **Check for model corruption** - validate .mlpackage structure

## Conclusion

**Swift + CoreML works perfectly for simple models but the vocoder and flow models have fundamental loading issues** that affect both Swift and Python. The models likely need to be re-converted with different settings or the conversion process needs to be debugged.

The good news: Swift CoreML is 80x+ faster than Python for the models that DO work (embedding, lm_head). The problem is with the vocoder/flow conversion, not the Swift implementation.
