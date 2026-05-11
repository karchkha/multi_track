# multi_track

A Max/MSP external for multi-track audio source separation using neural networks.
Separates audio into four instrument stems — **bass, drums, guitar, piano** — by communicating with a Python server over OSC (Open Sound Control).

**Author:** Anonymous

---

## For Users — Installing the Max Package

1. Copy the `multi_track/` folder (inside this repo) into your Max packages directory:
   ```
   Documents/Max 8/Packages/
   ```
2. Rename it to `multi_track` if it isn't already.
3. Restart Max. The external will be available as `multi_track`.

The `multi_track/` folder contains:
- `externals/multi_track.mxe64` — the compiled external (Windows 64-bit)
- `externals/multi_track.mxo` — the compiled external (macOS)
- `help/multi_track.maxhelp` — the interactive help patch
- `docs/refpages/multi_track.maxref.xml` — the reference documentation

### macOS — UDP packet size

macOS limits UDP datagrams to **9216 bytes** by default, which is too small for audio chunks. When you first load `multi_track` in Max, it will automatically detect this and show a system dialog asking for your admin password to raise the limit to 65535 bytes.

To avoid this prompt on every restart, make the change permanent by adding one line to `/etc/sysctl.conf`:

```bash
echo "net.inet.udp.maxdgram=65535" | sudo tee -a /etc/sysctl.conf
```

---

## Usage

Open `multi_track.maxhelp` in Max for a full interactive guide. The help patch demonstrates all messages, configuration options, and typical usage patterns.

The Python server component of this project is available here:
https://github.com/[anonymous]/musical-accompaniment-ldm

### OSC Communication

| Direction | Port |
|-----------|------|
| Max → Python (sender) | 7000 |
| Python → Max (listener) | 8000 |

### Expected messages from the server

| OSC address | Description |
|---|---|
| `/ready <bool>` | Server is initialized and ready |
| `/server_predicted <bool>` | Inference complete |
| `/bass <index> <float[]>` | Bass audio result |
| `/drums <index> <float[]>` | Drums audio result |
| `/guitar <index> <float[]>` | Guitar audio result |
| `/piano <index> <float[]>` | Piano audio result |
| `/packet_test_response <size> <float[]>` | Response to test_packet |

---

## For Developers — Building from Source

---

### Building on Windows

#### Requirements

- Windows 10, x64
- Visual Studio 2022
- Max SDK (`max-sdk-main`) — this source file must live inside it at:
  ```
  max-sdk-main/source/advanced/multi_track/
  ```
- oscpack 1.1.0 — download from https://github.com/RossBencina/oscpack and place the folder at `max-sdk-main/oscpack_1_1_0/`

#### Step 1 — Build oscpack

1. Download oscpack 1.1.0 from https://github.com/RossBencina/oscpack
2. Place it at `max-sdk-main/oscpack_1_1_0/`
3. Run CMake to generate the Visual Studio build files:
   ```
   cmake -S "path\to\max-sdk-main\oscpack_1_1_0" -B "path\to\max-sdk-main\oscpack_1_1_0\build" -G "Visual Studio 17 2022" -A x64
   ```
4. Open **Developer Command Prompt for VS 2022** and build both configurations:
   ```
   msbuild "path\to\max-sdk-main\oscpack_1_1_0\build\oscpack.vcxproj" /t:Rebuild /p:Configuration=Debug /p:Platform=x64
   msbuild "path\to\max-sdk-main\oscpack_1_1_0\build\oscpack.vcxproj" /t:Rebuild /p:Configuration=Release /p:Platform=x64
   ```

#### Step 2 — Build the external

1. Open `max-sdk-main/build/max-sdk-main.sln` in Visual Studio 2022
2. Select configuration **Release** and platform **x64**
3. **Build → Build Solution** (F7)

The compiled external will be placed at:
```
max-sdk-main/externals/multi_track.mxe64
```

Copy it into `multi_track/externals/` to update the package.

---

### Building on macOS

#### Requirements

- macOS 11 or later (Intel or Apple Silicon)
- Xcode (with Command Line Tools)
- CMake — install via Homebrew: `brew install cmake`
- Max SDK (`max-sdk-main`) with submodules initialized:
  ```
  git clone https://github.com/Cycling74/max-sdk.git max-sdk-main
  cd max-sdk-main
  git submodule update --init --recursive
  ```
- This source folder placed at `max-sdk-main/source/advanced/multi_track/`

#### Step 1 — Build oscpack (universal binary)

```bash
cd /path/to/max-sdk-main
git clone https://github.com/RossBencina/oscpack oscpack_1_1_0
mkdir oscpack_1_1_0/build
cd oscpack_1_1_0/build
cmake .. -DCMAKE_OSX_ARCHITECTURES="x86_64;arm64" -DCMAKE_BUILD_TYPE=Release -DCMAKE_POLICY_VERSION_MINIMUM=3.5
make
```

This produces `liboscpack.a` in `oscpack_1_1_0/build/`.

#### Step 2 — Generate the Xcode project

```bash
cd /path/to/max-sdk-main
mkdir -p build
cd build
cmake .. -G Xcode
```

#### Step 3 — Build the external in Xcode

1. Open `max-sdk-main/build/max-sdk-main.xcodeproj` in Xcode
2. Go to **Product → Scheme → Manage Schemes** and enable only `multi_track`
3. Select **Release** configuration
4. Press **Cmd+B** to build

The compiled external (`multi_track.mxo`) will appear in `max-sdk-main/externals/`.
Copy it into `multi_track/externals/` to update the package.

---

## License

See Max SDK license.
