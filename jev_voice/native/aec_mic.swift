// The microphone through Apple's voice processing, for jev_voice/aec.py (AEC=apple).
//
//   aec-mic [--in <name fragment>]        (no --in: the system default input)
//           [--out <name fragment>]       (tests only; no --out: the default output)
//
// Streams 16 kHz mono float32 little-endian samples on stdout. The echo reference is the
// current default output device; when that changes the canceller would be listening for
// the wrong speakers, so it exits 3 and the Python side decides what to start next.
// Exits 0 when stdin closes, 4 when the microphone disappears, 1 on a CoreAudio failure
// (the step on stderr). The audio thread never blocks: it copies into a ring buffer and
// a writer thread drains that to stdout; if the reader stalls, samples are dropped.
import AudioToolbox
import CoreAudio
import Foundation
import Synchronization

let sampleRate = 16_000.0
let exitDeviceChanged: Int32 = 3
let exitMicGone: Int32 = 4

func fail(_ what: String, _ status: OSStatus = noErr) -> Never {
    let code = status == noErr ? "" : " (OSStatus \(status))"
    FileHandle.standardError.write(Data("aec-mic: \(what)\(code)\n".utf8))
    exit(1)
}

func check(_ status: OSStatus, _ what: String) {
    if status != noErr { fail(what, status) }
}

// MARK: - devices

func address(_ selector: AudioObjectPropertySelector,
             _ scope: AudioObjectPropertyScope = kAudioObjectPropertyScopeGlobal)
    -> AudioObjectPropertyAddress {
    AudioObjectPropertyAddress(mSelector: selector, mScope: scope,
                               mElement: kAudioObjectPropertyElementMain)
}

let system = AudioObjectID(kAudioObjectSystemObject)

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

func defaultDevice(_ selector: AudioObjectPropertySelector) -> AudioDeviceID {
    var addr = address(selector)
    var id = AudioDeviceID(0)
    var size = UInt32(MemoryLayout<AudioDeviceID>.size)
    check(AudioObjectGetPropertyData(system, &addr, 0, nil, &size, &id), "read the default device")
    return id
}

func findDevice(_ fragment: String, scope: AudioObjectPropertyScope) -> AudioDeviceID {
    var addr = address(kAudioHardwarePropertyDevices)
    var size: UInt32 = 0
    check(AudioObjectGetPropertyDataSize(system, &addr, 0, nil, &size), "list devices")
    var ids = [AudioDeviceID](repeating: 0, count: Int(size) / MemoryLayout<AudioDeviceID>.size)
    check(AudioObjectGetPropertyData(system, &addr, 0, nil, &size, &ids), "list devices")
    let want = fragment.lowercased()
    for id in ids where deviceName(id).lowercased().contains(want) {
        var streams = address(kAudioDevicePropertyStreams, scope)
        var bytes: UInt32 = 0
        if AudioObjectGetPropertyDataSize(id, &streams, 0, nil, &bytes) == noErr && bytes > 0 {
            return id
        }
    }
    fail("no device matching \"\(fragment)\"")
}

func findInput(_ fragment: String) -> AudioDeviceID {
    if fragment.isEmpty { return defaultDevice(kAudioHardwarePropertyDefaultInputDevice) }
    return findDevice(fragment, scope: kAudioObjectPropertyScopeInput)
}

// MARK: - audio thread -> writer thread

/// Single producer (the audio thread), single consumer (the writer thread).
final class Ring: @unchecked Sendable {
    let capacity = 1 << 16                     // 4 s at 16 kHz
    let buffer: UnsafeMutablePointer<Float>
    let head = Atomic<Int>(0)                  // samples written, ever
    let tail = Atomic<Int>(0)                  // samples read, ever
    let dropped = Atomic<Int>(0)

    init() { buffer = .allocate(capacity: capacity) }

    func push(_ source: UnsafePointer<Float>, _ count: Int) {
        let written = head.load(ordering: .relaxed)
        let read = tail.load(ordering: .acquiring)
        if written - read + count > capacity {
            dropped.wrappingAdd(count, ordering: .relaxed)
            return
        }
        for i in 0..<count { buffer[(written + i) & (capacity - 1)] = source[i] }
        head.store(written + count, ordering: .releasing)
    }

    func pop(into target: UnsafeMutablePointer<Float>, max: Int) -> Int {
        let read = tail.load(ordering: .relaxed)
        let count = min(head.load(ordering: .acquiring) - read, max)
        for i in 0..<count { target[i] = buffer[(read + i) & (capacity - 1)] }
        tail.store(read + count, ordering: .releasing)
        return count
    }
}

final class Session: @unchecked Sendable {
    var unit: AudioUnit?
    let ring = Ring()
    let scratchFrames = 16_384
    let scratch: UnsafeMutablePointer<Float>

    init() { scratch = .allocate(capacity: scratchFrames) }

    func pull(_ flags: UnsafeMutablePointer<AudioUnitRenderActionFlags>,
              _ stamp: UnsafePointer<AudioTimeStamp>, _ bus: UInt32, _ frames: UInt32) -> OSStatus {
        let count = Int(frames)
        guard let unit, count <= scratchFrames else { return noErr }
        var list = AudioBufferList(
            mNumberBuffers: 1,
            mBuffers: AudioBuffer(mNumberChannels: 1, mDataByteSize: UInt32(count * 4),
                                  mData: UnsafeMutableRawPointer(scratch)))
        let status = AudioUnitRender(unit, flags, stamp, bus, frames, &list)
        if status == noErr { ring.push(scratch, count) }
        return status
    }
}

let inputCallback: AURenderCallback = { refCon, flags, stamp, bus, frames, _ in
    Unmanaged<Session>.fromOpaque(refCon).takeUnretainedValue().pull(flags, stamp, bus, frames)
}

// The unit's own output stays silent: Jev plays nothing through it.
let silentOutput: AURenderCallback = { _, flags, _, _, _, ioData in
    guard let ioData else { return noErr }
    for buffer in UnsafeMutableAudioBufferListPointer(ioData) {
        if let data = buffer.mData { memset(data, 0, Int(buffer.mDataByteSize)) }
    }
    flags.pointee.insert(.unitRenderAction_OutputIsSilence)
    return noErr
}

// MARK: - the unit

func set<T>(_ unit: AudioUnit, _ property: AudioUnitPropertyID, _ scope: AudioUnitScope,
            _ element: AudioUnitElement, _ value: T, _ what: String) {
    let status = withUnsafePointer(to: value) {
        AudioUnitSetProperty(unit, property, scope, element, $0, UInt32(MemoryLayout<T>.size))
    }
    check(status, what)
}

func makeUnit(_ session: Session, input: AudioDeviceID, reference: AudioDeviceID) -> AudioUnit {
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
    set(unit, kAudioOutputUnitProperty_CurrentDevice, kAudioUnitScope_Global, 0, reference,
        "reference device")
    // The listener tracks its own noise floor; AGC would move it under its feet.
    set(unit, kAUVoiceIOProperty_VoiceProcessingEnableAGC, kAudioUnitScope_Global, 0, UInt32(0),
        "AGC off")
    // Other audio is ducked while the unit runs. Min is as low as macOS goes (3.2 dB here).
    let ducking = AUVoiceIOOtherAudioDuckingConfiguration(
        mEnableAdvancedDucking: false, mDuckingLevel: .min)
    set(unit, kAUVoiceIOProperty_OtherAudioDuckingConfiguration, kAudioUnitScope_Global, 0,
        ducking, "ducking")

    let format = AudioStreamBasicDescription(
        mSampleRate: sampleRate, mFormatID: kAudioFormatLinearPCM,
        mFormatFlags: kAudioFormatFlagIsFloat | kAudioFormatFlagIsPacked
            | kAudioFormatFlagIsNonInterleaved,
        mBytesPerPacket: 4, mFramesPerPacket: 1, mBytesPerFrame: 4,
        mChannelsPerFrame: 1, mBitsPerChannel: 32, mReserved: 0)
    set(unit, kAudioUnitProperty_StreamFormat, kAudioUnitScope_Output, 1, format, "mic format")
    set(unit, kAudioUnitProperty_StreamFormat, kAudioUnitScope_Input, 0, format, "output format")

    let ref = Unmanaged.passUnretained(session).toOpaque()
    set(unit, kAudioOutputUnitProperty_SetInputCallback, kAudioUnitScope_Global, 1,
        AURenderCallbackStruct(inputProc: inputCallback, inputProcRefCon: ref), "input callback")
    set(unit, kAudioUnitProperty_SetRenderCallback, kAudioUnitScope_Input, 0,
        AURenderCallbackStruct(inputProc: silentOutput, inputProcRefCon: nil), "output callback")
    return unit
}

func shutDown(_ code: Int32) -> Never {
    if let unit = session.unit {
        AudioOutputUnitStop(unit)
        AudioUnitUninitialize(unit)
        AudioComponentInstanceDispose(unit)
    }
    let dropped = session.ring.dropped.load(ordering: .relaxed)
    if dropped > 0 {
        FileHandle.standardError.write(Data("aec-mic: dropped \(dropped) samples\n".utf8))
    }
    exit(code)
}

// MARK: - main

var fragment = ""
var referenceFragment = ""      // tests only: the reference is otherwise the default output
var args = CommandLine.arguments.dropFirst().makeIterator()
while let flag = args.next() {
    guard let value = args.next(), flag == "--in" || flag == "--out" else {
        fail("usage: aec-mic [--in <mic name fragment>] [--out <reference name fragment>]")
    }
    if flag == "--in" { fragment = value } else { referenceFragment = value }
}
signal(SIGPIPE, SIG_IGN)

let session = Session()
let inputID = findInput(fragment)
let referenceID = referenceFragment.isEmpty
    ? defaultDevice(kAudioHardwarePropertyDefaultOutputDevice)
    : findDevice(referenceFragment, scope: kAudioObjectPropertyScopeOutput)
let unit = makeUnit(session, input: inputID, reference: referenceID)
session.unit = unit
check(AudioUnitInitialize(unit), "initialise the unit")
check(AudioOutputUnitStart(unit), "start the unit")
FileHandle.standardError.write(Data(
    "aec-mic: mic=\(deviceName(inputID)) reference=\(deviceName(referenceID))\n".utf8))

var outputAddress = address(kAudioHardwarePropertyDefaultOutputDevice)
AudioObjectAddPropertyListenerBlock(system, &outputAddress, DispatchQueue.main) { _, _ in
    shutDown(exitDeviceChanged)
}
var aliveAddress = address(kAudioDevicePropertyDeviceIsAlive)
AudioObjectAddPropertyListenerBlock(inputID, &aliveAddress, DispatchQueue.main) { _, _ in
    shutDown(exitMicGone)
}

Thread {
    let chunk = UnsafeMutablePointer<Float>.allocate(capacity: 4096)
    defer { chunk.deallocate() }
    while true {
        let count = session.ring.pop(into: chunk, max: 4096)
        if count == 0 {
            usleep(5_000)
            continue
        }
        var remaining = count * 4
        var cursor = UnsafeRawPointer(chunk)
        while remaining > 0 {
            let written = write(STDOUT_FILENO, cursor, remaining)
            if written < 0 {
                if errno == EINTR { continue }
                DispatchQueue.main.async { shutDown(0) }     // the reader went away
                return
            }
            remaining -= written
            cursor += written
        }
    }
}.start()

Thread {
    _ = FileHandle.standardInput.readDataToEndOfFile()
    DispatchQueue.main.async { shutDown(0) }
}.start()

dispatchMain()
