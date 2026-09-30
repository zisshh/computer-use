// Records the microphone through Apple's voice processing (the VoiceProcessingIO audio
// unit, the echo canceller FaceTime uses) so scripts/aec_probe.py can measure whether it
// removes audio that OTHER processes play through the speakers.
//
//   aec-probe --in "USB Condenser" --out "MacBook Pro Speakers" --wav out.wav
//             [--bypass 1] [--seconds 20] [--play passage.f32 --play-delay 1.0]
//
// --out is the echo reference: the unit cancels what it believes that device plays.
// --play renders a mono 16 kHz float32 file through the unit's own output (the case the
// canceller certainly handles), as a positive control.
// Prints "ready" on stdout once the unit runs; stops at --seconds or when stdin closes;
// writes a 16 kHz mono 16-bit WAV. Any CoreAudio failure exits 1 with the step on stderr.
import AudioToolbox
import CoreAudio
import Foundation

let sampleRate = 16_000.0

struct Options {
    var input = ""
    var output = ""
    var wav = ""
    var seconds = 20.0
    var bypass = false
    var play = ""
    var playDelay = 1.0
}

func fail(_ what: String, _ status: OSStatus = noErr) -> Never {
    let code = status == noErr ? "" : " (OSStatus \(status))"
    FileHandle.standardError.write(Data("aec-probe: \(what)\(code)\n".utf8))
    exit(1)
}

func check(_ status: OSStatus, _ what: String) {
    if status != noErr { fail(what, status) }
}

func parseOptions() -> Options {
    var options = Options()
    var args = CommandLine.arguments.dropFirst().makeIterator()
    while let flag = args.next() {
        guard let value = args.next() else { fail("\(flag) needs a value") }
        switch flag {
        case "--in": options.input = value
        case "--out": options.output = value
        case "--wav": options.wav = value
        case "--seconds": options.seconds = Double(value) ?? options.seconds
        case "--bypass": options.bypass = value == "1"
        case "--play": options.play = value
        case "--play-delay": options.playDelay = Double(value) ?? options.playDelay
        default: fail("unknown flag \(flag)")
        }
    }
    if options.input.isEmpty || options.output.isEmpty || options.wav.isEmpty {
        fail("usage: aec-probe --in <mic> --out <speakers> --wav <path> [--bypass 1] "
            + "[--seconds N] [--play file.f32 --play-delay S]")
    }
    return options
}

// MARK: - devices

func address(_ selector: AudioObjectPropertySelector,
             _ scope: AudioObjectPropertyScope = kAudioObjectPropertyScopeGlobal)
    -> AudioObjectPropertyAddress {
    AudioObjectPropertyAddress(mSelector: selector, mScope: scope,
                               mElement: kAudioObjectPropertyElementMain)
}

func deviceName(_ id: AudioDeviceID) -> String {
    var addr = address(kAudioObjectPropertyName)
    var name: Unmanaged<CFString>?
    var size = UInt32(MemoryLayout<Unmanaged<CFString>?>.size)
    let status = withUnsafeMutablePointer(to: &name) {
        AudioObjectGetPropertyData(id, &addr, 0, nil, &size, $0)
    }
    guard status == noErr, let name else { return "device \(id)" }
    return name.takeRetainedValue() as String
}

func hasStreams(_ id: AudioDeviceID, input: Bool) -> Bool {
    var addr = address(kAudioDevicePropertyStreams,
                       input ? kAudioObjectPropertyScopeInput : kAudioObjectPropertyScopeOutput)
    var size: UInt32 = 0
    return AudioObjectGetPropertyDataSize(id, &addr, 0, nil, &size) == noErr && size > 0
}

func findDevice(_ fragment: String, input: Bool) -> AudioDeviceID {
    var addr = address(kAudioHardwarePropertyDevices)
    let system = AudioObjectID(kAudioObjectSystemObject)
    var size: UInt32 = 0
    check(AudioObjectGetPropertyDataSize(system, &addr, 0, nil, &size), "list devices")
    var ids = [AudioDeviceID](repeating: 0, count: Int(size) / MemoryLayout<AudioDeviceID>.size)
    check(AudioObjectGetPropertyData(system, &addr, 0, nil, &size, &ids), "list devices")
    let want = fragment.lowercased()
    for id in ids where hasStreams(id, input: input)
        && deviceName(id).lowercased().contains(want) {
        return id
    }
    fail("no \(input ? "input" : "output") device matching \"\(fragment)\"")
}

// MARK: - the session the callbacks see

final class Session {
    var unit: AudioUnit?
    let capacity: Int
    let samples: UnsafeMutablePointer<Float>
    var count = 0
    let scratchFrames = 16_384
    let scratch: UnsafeMutablePointer<Float>
    var renderError: OSStatus = noErr
    var callbacks = 0
    var largest = 0
    var dropped = 0
    let playback: [Float]
    let playStart: Int
    var rendered = 0

    init(seconds: Double, playback: [Float], playDelay: Double) {
        capacity = Int(seconds * sampleRate) + 1
        samples = .allocate(capacity: capacity)
        scratch = .allocate(capacity: scratchFrames)
        self.playback = playback
        playStart = Int(playDelay * sampleRate)
    }

    /// Input callback: pull the processed microphone out of the unit. No I/O here.
    func pull(_ flags: UnsafeMutablePointer<AudioUnitRenderActionFlags>,
              _ stamp: UnsafePointer<AudioTimeStamp>, _ bus: UInt32, _ frames: UInt32) -> OSStatus {
        let n = Int(frames)
        callbacks += 1
        largest = max(largest, n)
        guard let unit, n <= scratchFrames else {
            dropped += n
            return noErr
        }
        var list = AudioBufferList(
            mNumberBuffers: 1,
            mBuffers: AudioBuffer(mNumberChannels: 1, mDataByteSize: UInt32(n * 4),
                                  mData: UnsafeMutableRawPointer(scratch)))
        let status = AudioUnitRender(unit, flags, stamp, bus, frames, &list)
        if status != noErr {
            renderError = status
            return status
        }
        let keep = min(n, capacity - count)
        if keep > 0 {
            (samples + count).update(from: scratch, count: keep)
            count += keep
        }
        return noErr
    }

    /// Render callback: what the unit plays. Silence, or the positive-control passage.
    func fill(_ ioData: UnsafeMutablePointer<AudioBufferList>, _ frames: UInt32) {
        let n = Int(frames)
        for buffer in UnsafeMutableAudioBufferListPointer(ioData) {
            guard let data = buffer.mData else { continue }
            let out = data.assumingMemoryBound(to: Float.self)
            let channelFrames = min(n, Int(buffer.mDataByteSize) / 4)
            for i in 0..<channelFrames {
                let k = rendered + i - playStart
                out[i] = (k >= 0 && k < playback.count) ? playback[k] : 0
            }
        }
        rendered += n
    }
}

let inputCallback: AURenderCallback = { refCon, flags, stamp, bus, frames, _ in
    Unmanaged<Session>.fromOpaque(refCon).takeUnretainedValue().pull(flags, stamp, bus, frames)
}

let renderCallback: AURenderCallback = { refCon, _, _, _, frames, ioData in
    guard let ioData else { return noErr }
    Unmanaged<Session>.fromOpaque(refCon).takeUnretainedValue().fill(ioData, frames)
    return noErr
}

// MARK: - unit setup

func set<T>(_ unit: AudioUnit, _ property: AudioUnitPropertyID, _ scope: AudioUnitScope,
            _ element: AudioUnitElement, _ value: T, _ what: String) {
    let status = withUnsafePointer(to: value) {
        AudioUnitSetProperty(unit, property, scope, element, $0, UInt32(MemoryLayout<T>.size))
    }
    check(status, what)
}

func currentDevice(_ unit: AudioUnit, _ element: AudioUnitElement) -> AudioDeviceID {
    var id = AudioDeviceID(0)
    var size = UInt32(MemoryLayout<AudioDeviceID>.size)
    AudioUnitGetProperty(unit, kAudioOutputUnitProperty_CurrentDevice,
                         kAudioUnitScope_Global, element, &id, &size)
    return id
}

func makeUnit(_ options: Options, _ session: Session, input: AudioDeviceID,
              output: AudioDeviceID) -> AudioUnit {
    var desc = AudioComponentDescription(
        componentType: kAudioUnitType_Output,
        componentSubType: kAudioUnitSubType_VoiceProcessingIO,
        componentManufacturer: kAudioUnitManufacturer_Apple,
        componentFlags: 0, componentFlagsMask: 0)
    guard let component = AudioComponentFindNext(nil, &desc) else {
        fail("no VoiceProcessingIO unit on this system")
    }
    var made: AudioUnit?
    check(AudioComponentInstanceNew(component, &made), "create the unit")
    guard let unit = made else { fail("create the unit") }

    set(unit, kAudioOutputUnitProperty_EnableIO, kAudioUnitScope_Input, 1, UInt32(1), "enable input")
    set(unit, kAudioOutputUnitProperty_EnableIO, kAudioUnitScope_Output, 0, UInt32(1), "enable output")
    set(unit, kAudioOutputUnitProperty_CurrentDevice, kAudioUnitScope_Global, 1, input, "input device")
    set(unit, kAudioOutputUnitProperty_CurrentDevice, kAudioUnitScope_Global, 0, output, "output device")
    set(unit, kAUVoiceIOProperty_BypassVoiceProcessing, kAudioUnitScope_Global, 0,
        UInt32(options.bypass ? 1 : 0), "bypass")
    // Levels have to be comparable between runs; AGC would chase them.
    set(unit, kAUVoiceIOProperty_VoiceProcessingEnableAGC, kAudioUnitScope_Global, 0, UInt32(0), "AGC off")
    if #available(macOS 14.0, *) {
        let ducking = AUVoiceIOOtherAudioDuckingConfiguration(
            mEnableAdvancedDucking: false,
            mDuckingLevel: AUVoiceIOOtherAudioDuckingLevel(rawValue: 10)!)   // ...LevelMin
        set(unit, kAUVoiceIOProperty_OtherAudioDuckingConfiguration, kAudioUnitScope_Global, 0,
            ducking, "ducking")
    }

    let format = AudioStreamBasicDescription(
        mSampleRate: sampleRate, mFormatID: kAudioFormatLinearPCM,
        mFormatFlags: kAudioFormatFlagIsFloat | kAudioFormatFlagIsPacked
            | kAudioFormatFlagIsNonInterleaved,
        mBytesPerPacket: 4, mFramesPerPacket: 1, mBytesPerFrame: 4,
        mChannelsPerFrame: 1, mBitsPerChannel: 32, mReserved: 0)
    set(unit, kAudioUnitProperty_StreamFormat, kAudioUnitScope_Output, 1, format, "mic format")
    set(unit, kAudioUnitProperty_StreamFormat, kAudioUnitScope_Input, 0, format, "speaker format")

    let ref = Unmanaged.passUnretained(session).toOpaque()
    set(unit, kAudioOutputUnitProperty_SetInputCallback, kAudioUnitScope_Global, 1,
        AURenderCallbackStruct(inputProc: inputCallback, inputProcRefCon: ref), "input callback")
    set(unit, kAudioUnitProperty_SetRenderCallback, kAudioUnitScope_Input, 0,
        AURenderCallbackStruct(inputProc: renderCallback, inputProcRefCon: ref), "render callback")
    return unit
}

// MARK: - files

func readFloats(_ path: String) -> [Float] {
    guard let data = FileManager.default.contents(atPath: path) else { fail("cannot read \(path)") }
    return data.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
}

func writeWAV(_ path: String, _ samples: UnsafeBufferPointer<Float>) {
    var data = Data()
    func u32(_ v: UInt32) { withUnsafeBytes(of: v.littleEndian) { data.append(contentsOf: $0) } }
    func u16(_ v: UInt16) { withUnsafeBytes(of: v.littleEndian) { data.append(contentsOf: $0) } }
    let bytes = UInt32(samples.count * 2)
    data.append(Data("RIFF".utf8)); u32(36 + bytes); data.append(Data("WAVE".utf8))
    data.append(Data("fmt ".utf8)); u32(16); u16(1); u16(1)
    u32(UInt32(sampleRate)); u32(UInt32(sampleRate) * 2); u16(2); u16(16)
    data.append(Data("data".utf8)); u32(bytes)
    for s in samples {
        let v = Int16(max(-1, min(1, s)) * 32767)
        withUnsafeBytes(of: v.littleEndian) { data.append(contentsOf: $0) }
    }
    if !FileManager.default.createFile(atPath: path, contents: data) { fail("cannot write \(path)") }
}

// MARK: - main

let options = parseOptions()
let inputID = findDevice(options.input, input: true)
let outputID = findDevice(options.output, input: false)
let session = Session(seconds: options.seconds,
                      playback: options.play.isEmpty ? [] : readFloats(options.play),
                      playDelay: options.playDelay)
let unit = makeUnit(options, session, input: inputID, output: outputID)
session.unit = unit
check(AudioUnitInitialize(unit), "initialise the unit")
check(AudioOutputUnitStart(unit), "start the unit")
FileHandle.standardError.write(Data((
    "aec-probe: mic=\(deviceName(currentDevice(unit, 1))) "
    + "reference=\(deviceName(currentDevice(unit, 0))) bypass=\(options.bypass)\n").utf8))
// "ready" means audio is arriving, not that the unit started: the first input can lag.
let started = Date()
while session.count == 0 && Date().timeIntervalSince(started) < 5 {
    usleep(2_000)
}
if session.count == 0 { fail("no audio arrived within 5 s") }
let lagMs = Int(Date().timeIntervalSince(started) * 1000)
FileHandle.standardError.write(Data("aec-probe: first audio after \(lagMs) ms\n".utf8))
print("ready")
fflush(stdout)

let stop = DispatchSemaphore(value: 0)
DispatchQueue.global().async {
    _ = FileHandle.standardInput.readDataToEndOfFile()
    stop.signal()
}
_ = stop.wait(timeout: .now() + options.seconds)

AudioOutputUnitStop(unit)
AudioUnitUninitialize(unit)
AudioComponentInstanceDispose(unit)
writeWAV(options.wav, UnsafeBufferPointer(start: session.samples, count: session.count))
FileHandle.standardError.write(Data((
    "aec-probe: \(session.count) frames in \(session.callbacks) callbacks, largest "
    + "\(session.largest), dropped \(session.dropped)\n").utf8))
if session.renderError != noErr { fail("AudioUnitRender", session.renderError) }
