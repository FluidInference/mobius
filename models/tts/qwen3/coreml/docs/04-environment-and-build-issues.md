# Environment and Build Issues

Issues related to disk space, build tooling, and platform compatibility.

---

## 1. Disk Space Exhaustion During CoreML Compilation

**Problem:** Both Swift TTS runs failed with `LLVM ERROR: IO failure on output stream: No space left on device`. The Data volume was at 97% capacity (403GB/460GB).

**Root Cause:** The ANE (Apple Neural Engine) compilation cache at `~/Library/Caches/python/com.apple.e5rt.e5bundlecache/` had grown to **27GB**. Running 3 simultaneous processes (Python bilingual test + 2 Swift runs) each triggered CoreML compilation, exhausting the remaining disk space.

**Fix:** Cleared the ANE cache:
```bash
rm -rf ~/Library/Caches/python/com.apple.e5rt.e5bundlecache
```
Freed 27GB (cache auto-regenerates on next model load). Then ran tasks sequentially to avoid disk contention.

---

## 2. Swift Guard Statement Compilation Error

**Problem:** Swift build failed with "guard body must not fall through":
```swift
guard sortable.count > topK else { /* all pass */ }
```

**Root Cause:** In Swift, a `guard` statement's `else` block must exit the scope (via `return`, `throw`, `break`, etc.). An empty or non-exiting else block is a compile error.

**Fix:** Replaced the guard with a regular `if` statement:
```swift
if sortable.count > topK {
    // top-k filtering logic
}
```

---

## 3. Swift Binary Name Mismatch

**Problem:** Background task failed with exit code 127 (command not found) when trying to run `fluidaudio`.

**Root Cause:** The actual binary is named `fluidaudiocli`, not `fluidaudio`.

**Fix:** Corrected the binary name to `.build/release/fluidaudiocli`.

---

## 4. Wrong Working Directory for CLI Runs

**Problem:** `.build/release/fluidaudiocli` not found when shell commands ran from the wrong directory.

**Root Cause:** Background tasks didn't always inherit the expected working directory.

**Fix:** Always use the full path: `cd /path/to/FluidAudio && .build/release/fluidaudiocli ...`

---

## 5. V9 Prefill Hardcoded to English Language

**Problem:** Mandarin testing couldn't work with the V9 prefill model because it only supported English language ID.

**Root Cause:** The V9 prefill conversion hardcoded the English language identifier in the embedding sequence construction.

**Workaround:** For bilingual testing, used PyTorch prefill (supports all languages) + CoreML V3 decode. The Swift pipeline later supported Chinese by accepting pre-tokenized Chinese text tokens with the correct language ID embedded in the token sequence.
