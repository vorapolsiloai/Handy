# Handy AMD (GPU): build from source

This document is written so that a person **or a coding agent (e.g. Claude Code)** can follow it top to bottom.
Every step has a command and a checkpoint. Stop at the first failed checkpoint and fix it before continuing.

## 0. Do you need to build at all?

| Your machine | What to do |
|---|---|
| Windows 11 x64, any GPU with a current driver (AMD Radeon, NVIDIA GeForce/RTX, Intel Arc) | **No build needed.** Use the prebuilt installer `Handy-AMD-GPU-0.9.8-setup.exe`. It uses Vulkan, which works on all of these. |
| Windows 11 x64, NVIDIA GPU, and you want CUDA instead of Vulkan | Build, with backend `cuda` (section 3). |
| Linux x64 | Build (section 8), or use official Handy for Linux. |
| macOS (Apple Silicon) | **No build needed.** Use the official Handy release (https://github.com/cjpais/Handy/releases); it already runs on the Apple GPU (Metal). |
| You want to change the code | Build. |

What this build adds on top of upstream Handy (https://github.com/cjpais/Handy, MIT):
- Every speech model runs on the GPU (the upcoming Handy release's transcribe.cpp engine), available today.
- GPU warm-up right after a model loads, so the first dictation is fast (about 0.1-0.3 s instead of 1.3-2.4 s).
- transcribe.cpp fixes: Vulkan long-clip attention fix (no CPU fallback for clips over 51 s) and a better CPU thread default.

Measured on a Radeon 8060S (Ryzen AI MAX+ 395), Parakeet TDT 0.6B v2: 9 s of speech in ~0.08 s, 30 s in ~0.2 s.

## 1. Sources

Two repositories are cloned **side by side** in the same parent folder (the Handy build points at `../../transcribe.cpp` relative to `Handy/src-tauri`).

| Repo | URL | Branch |
|---|---|---|
| Handy (app) | https://github.com/vorapolsiloai/Handy | `amd/gpu-warmup` |
| transcribe.cpp (speech engine) | https://github.com/vorapolsiloai/transcribe.cpp | `amd/handy-integration-0.4.0` |

Upstream bases: Handy `cjpais/Handy` main at `7bfb2e2`; transcribe.cpp `handy-computer/transcribe.cpp` tag `v0.4.0`.

## 2. Prerequisites (Windows)

Run in a normal PowerShell or cmd window. The Visual Studio Build Tools and Vulkan SDK installers ask for administrator approval (UAC).

```bat
winget install --id Git.Git --accept-source-agreements --accept-package-agreements
winget install --id Kitware.CMake --accept-source-agreements --accept-package-agreements
winget install --id Microsoft.VisualStudio.2022.BuildTools --accept-source-agreements --accept-package-agreements --override "--quiet --wait --norestart --add Microsoft.VisualStudio.Workload.VCTools --add Microsoft.VisualStudio.Component.VC.Tools.x86.x64 --add Microsoft.VisualStudio.Component.Windows11SDK.22621 --includeRecommended"
winget install --id KhronosGroup.VulkanSDK --accept-source-agreements --accept-package-agreements
winget install --id Rustlang.Rustup --accept-source-agreements --accept-package-agreements
winget install --id Oven-sh.Bun --accept-source-agreements --accept-package-agreements
```

Then open a **new** terminal (so PATH and `VULKAN_SDK` are picked up) and run `rustup default stable`.

Extra, only for backend `cuda`: install the NVIDIA CUDA Toolkit 12.x (`winget install --id Nvidia.CUDA`).

**Checkpoint 2:** each of these prints a version:
```bat
git --version && cmake --version && cargo --version && bun --version && echo %VULKAN_SDK%
```
`%VULKAN_SDK%` must print a folder such as `C:\VulkanSDK\1.4.x.y`.

Corporate networks that inspect TLS: if `bun install` or `cargo` fail with certificate errors, export your company root certificate to a `.pem` file and `set NODE_EXTRA_CA_CERTS=C:\path\to\root.pem` before building. (Cargo on Windows uses the Windows certificate store and normally works.)

## 3. Get the code and choose the GPU backend

```bat
mkdir %USERPROFILE%\src && cd /d %USERPROFILE%\src
git clone -b amd/gpu-warmup https://github.com/vorapolsiloai/Handy.git
git clone -b amd/handy-integration-0.4.0 https://github.com/vorapolsiloai/transcribe.cpp.git
```

The backend is chosen in `Handy\src-tauri\Cargo.toml`, block `[target.'cfg(all(windows, target_arch = "x86_64"))'.dependencies]`:

```toml
transcribe-cpp = { version = "0.4.0", default-features = false, features = [
  "dynamic-backends",
  "vulkan",
] }
```

| GPU | features list | Notes |
|---|---|---|
| Any (default) | `"dynamic-backends", "vulkan"` | Recommended. Works on AMD, NVIDIA and Intel. |
| NVIDIA | `"dynamic-backends", "vulkan", "cuda"` | Needs the CUDA Toolkit at build time; adds large CUDA DLLs to the package. Keeps Vulkan as fallback. |
| AMD, ROCm/HIP | `"dynamic-backends", "rocm"` | **Not recommended**: on a Radeon 8060S it measured 2-2.7x slower than Vulkan, and needs the ROCm SDK. |

**Checkpoint 3:** `Handy\src-tauri\Cargo.toml` contains, under `[patch.crates-io]`, a line pointing at the engine fork:
```toml
transcribe-cpp = { path = "../../transcribe.cpp/bindings/rust/transcribe-cpp" }
```
and the file `Handy\src-tauri\tauri.amd.conf.json` exists.

## 4. Build

Use a **"x64 Native Tools Command Prompt for VS 2022"** (Start menu), or run `"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"` first in cmd.

```bat
cd /d %USERPROFILE%\src\Handy
set HANDY_VC_REDIST_DIRS=%VCToolsRedistDir%x64\Microsoft.VC143.CRT;%VCToolsRedistDir%x64\Microsoft.VC143.OpenMP
rem Optional privacy: keep your user name out of the binaries (remove these three lines if you don't care)
set CARGO_TARGET_DIR=C:\hb\t
set RUSTFLAGS=--remap-path-prefix=%USERPROFILE%=C:\Users\user
set CL=/d1trimfile:%USERPROFILE%\
bun install
cd src-tauri && cargo update -p transcribe-cpp && cd ..
bun run tauri build --config src-tauri\tauri.amd.conf.json --no-sign
```

The first build takes 20-40 minutes (Rust crates, the C++ engine and the Vulkan shaders).

**Checkpoint 4:** the log ends with a line containing `Finished` and `Built application at`, and these exist (with `CARGO_TARGET_DIR=C:\hb\t`; otherwise under `Handy\src-tauri\target\`):
- `C:\hb\t\release\handy.exe`
- `C:\hb\t\release\ggml-vulkan.dll` and `transcribe.dll`
- `C:\hb\t\release\bundle\nsis\Handy AMD (GPU)_0.9.8_x64-setup.exe` (the installer to share)

For a runnable folder without an installer, use `--no-bundle` instead of `--no-sign`; then keep all files of `release\` together (`handy.exe` is not standalone: it loads `transcribe.dll`, the `ggml-*.dll` GPU/CPU backends, the VC++ runtime DLLs and `resources\`).

## 5. Verify on the target machine

```bat
"C:\hb\t\release\handy.exe" --list-devices
```
**Checkpoint 5a:** the output lists your GPU, e.g. `index=0 kind=vulkan name=AMD Radeon(TM) 8060S Graphics` (or your NVIDIA/Intel GPU), plus a `kind=cpu` line. If only CPU appears, update the GPU driver.

Then install and start the app, download **Parakeet TDT 0.6B v2** in *Models*, and transcribe a test WAV (16 kHz mono) headlessly (quit the app first; a running instance captures the command):
```bat
"%LOCALAPPDATA%\Handy AMD (GPU)\handy.exe" -f C:\path\to\test.wav --model handy-computer/parakeet-tdt-0.6b-v2-gguf/parakeet-tdt-0.6b-v2-Q8_0.gguf --device-index 0 --repeat 3 --json
```
**Checkpoint 5b:** JSON with `"bound_backend": "Vulkan0"` (or your GPU backend) and a sensible `text`. For a ~9 s clip on a recent GPU, `best_ms` is in the tens to low hundreds.

## 6. Install and use

- Installer: per-user, no admin rights. Windows SmartScreen shows "unknown publisher" because the build is unsigned: *More info* then *Run anyway* (follow your IT policy).
- **Quit any running Handy before installing** (both use `handy.exe` and Ctrl+Space; the installer asks to close it).
- Settings, history and downloaded models live in `%APPDATA%\com.pais.handy.amd` and `%LOCALAPPDATA%\com.pais.handy.amd`, separate from official Handy and never inside the app folder.
- Dictate: hold **Ctrl+Space**, speak, release.

## 7. Troubleshooting

| Symptom | Cause and fix |
|---|---|
| Launching the new exe opens the old window | Single-instance: an older copy with the same identity is running. Quit it from the tray first. |
| Grey overlay for ~2 s before recording starts | Bluetooth headset mic switching to hands-free mode. Pick the laptop's built-in mic in Settings, or enable *Always-on microphone* (press Ctrl+Shift+D for the Debug section). |
| First dictation after launch slow | Expected once after install or driver update (GPU shader cache). Later launches are fast. |
| `PowerShell ... running scripts is disabled` | Use cmd.exe and `.bat` files instead of `.ps1`. |
| Build fails in `transcribe-cpp-sys` with Vulkan errors | `VULKAN_SDK` not set in this terminal: open a new one after installing the SDK. |
| Build fails with `PATCH NOT APPLIED` / uses crates.io transcribe-cpp | Repos not side by side, or wrong branch of transcribe.cpp. Re-check section 1 and checkpoint 3. |
| `bun install` certificate errors | Set `NODE_EXTRA_CA_CERTS` (section 2). |

## 8. Linux (x64)

```bash
sudo apt install build-essential cmake git curl libvulkan-dev glslc libwebkit2gtk-4.1-dev libayatana-appindicator3-dev librsvg2-dev libasound2-dev libssl-dev
curl -fsSL https://bun.sh/install | bash && curl https://sh.rustup.rs -sSf | sh
mkdir -p ~/src && cd ~/src
git clone -b amd/gpu-warmup https://github.com/vorapolsiloai/Handy.git
git clone -b amd/handy-integration-0.4.0 https://github.com/vorapolsiloai/transcribe.cpp.git
cd Handy && bun install && bun run tauri build --config src-tauri/tauri.amd.conf.json
```
Linux already uses `"dynamic-backends", "vulkan"`. See the upstream Handy README for Wayland notes (global hotkeys and typing tools).

## 9. Provenance

- Handy: `cjpais/Handy` main `7bfb2e2` + GPU warm-up commit (worker: warm up GPU pipelines right after a model load; product name in window title and tray).
- transcribe.cpp: `handy-computer/transcribe.cpp` `v0.4.0` + Vulkan windowed-attention fix (no `ne[3]` broadcast) + CPU default threads = physical cores (cap 16).
- Licenses: Handy and transcribe.cpp MIT; Parakeet models CC-BY-4.0 (NVIDIA), downloaded by the app.
