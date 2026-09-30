// alfred-computer-use's Apple speech sidecar: DictationTranscriber (SpeechAnalyzer, macOS 26+)
// behind a pipe, because PyObjC cannot reach these Swift-only async APIs.
//
//   apple-stt [locale=en-IN] [--install]
//
// On start, one line:   {"ready": true, "locale": "en-IN", "format": "1 ch, 16000 Hz, Int16"}
//                  or   {"ready": false, "error": "..."}  and a non-zero exit.
// Then, per utterance:
//   request   {"id": 7, "samples": N, "context": ["Karan Aujla", ...] | null}\n
//             followed by exactly N*4 bytes of little-endian float32 mono 16 kHz PCM.
//             "context": null keeps the previous phrases; a list replaces them.
//             An optional "bytes" field states the payload size, so a request whose
//             size does not match its sample count can be skipped instead of trusted.
//   response  {"id": 7, "text": "open Spotify", "ms": 71.4}\n   or   {"id": 7, "error": "..."}\n
//
// stdout carries protocol lines and nothing else; everything a human might want goes
// to stderr. stdin closing is the signal to leave.
//
// DictationTranscriber rather than the newer SpeechTranscriber: measured on this
// machine, SpeechTranscriber ignores contextualStrings, and the contextual phrases are
// the whole reason to try this engine on Indian names.

import AVFoundation
import Foundation
import Speech

// MARK: - Output

/// print() is block-buffered when stdout is a pipe: the research probe sat "hung" for
/// ninety seconds with every result in a buffer nobody flushed. So protocol lines go
/// out through write(2), whole, the moment they exist.
func emit(_ fields: [String: Any]) {
    guard var line = try? JSONSerialization.data(withJSONObject: fields,
                                                 options: [.sortedKeys, .withoutEscapingSlashes]) else {
        log("could not encode a response: \(fields)")
        return
    }
    line.append(0x0A)
    let written = line.withUnsafeBytes { raw -> Bool in
        var offset = 0
        while offset < raw.count {
            let n = write(STDOUT_FILENO, raw.baseAddress! + offset, raw.count - offset)
            if n < 0 && errno == EINTR { continue }
            if n <= 0 { return false }
            offset += n
        }
        return true
    }
    // Nobody is reading any more: the parent is gone, and so is the reason to be here.
    if !written { exit(0) }
}

func log(_ message: String) {
    FileHandle.standardError.write(Data(("apple-stt: " + message + "\n").utf8))
}

func elapsedMs(since start: DispatchTime) -> Double {
    let ns = DispatchTime.now().uptimeNanoseconds - start.uptimeNanoseconds
    return (Double(ns) / 1e5).rounded() / 10
}

// MARK: - Requests off stdin

/// Two minutes of audio. A command is a few seconds; anything near this is a framing
/// error, and reading it would mean waiting on bytes that are never coming.
let maxSamples = 16_000 * 120
let maxHeaderBytes = 1 << 20
let maxContext = 100

enum Incoming {
    case utterance(id: Int, pcm: Data, context: [String]?)
    case bad(id: Int?, why: String)
}

final class StdinReader {
    private var pending = Data()

    private func fill() -> Bool {
        var chunk = [UInt8](repeating: 0, count: 1 << 16)
        while true {
            let n = read(STDIN_FILENO, &chunk, chunk.count)
            if n < 0 && errno == EINTR { continue }
            if n <= 0 { return false }
            pending.append(contentsOf: chunk[0..<n])
            return true
        }
    }

    /// The next line without its newline; nil once stdin has closed.
    func line() -> Data? {
        while true {
            if let newline = pending.firstIndex(of: 0x0A) {
                let line = pending.subdata(in: pending.startIndex..<newline)
                pending.removeSubrange(pending.startIndex...newline)
                return line
            }
            if pending.count > maxHeaderBytes {
                // Not a header. Drop it so garbage cannot grow without bound; the caller
                // sees an unparseable line and answers with an error.
                pending.removeAll()
                return Data([0x3F])
            }
            if !fill() { return nil }
        }
    }

    func take(_ count: Int) -> Data? {
        while pending.count < count {
            if !fill() { return nil }
        }
        let end = pending.startIndex + count
        let out = pending.subdata(in: pending.startIndex..<end)
        pending.removeSubrange(pending.startIndex..<end)
        return out
    }

    /// Read and throw away, without ever holding more than one chunk of it.
    func skip(_ count: Int) -> Bool {
        var left = count
        while left > 0 {
            if pending.isEmpty && !fill() { return false }
            let drop = min(left, pending.count)
            pending.removeSubrange(pending.startIndex..<(pending.startIndex + drop))
            left -= drop
        }
        return true
    }
}

/// One request off the wire, or nil when stdin is closed and it is time to go.
func nextRequest(_ reader: StdinReader) -> Incoming? {
    guard let raw = reader.line() else { return nil }
    if raw.allSatisfy({ $0 == 0x20 || $0 == 0x0D || $0 == 0x09 }) {
        return .bad(id: nil, why: "empty request line")
    }
    guard let header = (try? JSONSerialization.jsonObject(with: raw)) as? [String: Any] else {
        return .bad(id: nil, why: "request line is not a JSON object")
    }
    let id = header["id"] as? Int
    let samples = header["samples"] as? Int ?? -1
    let stated = header["bytes"] as? Int

    let inRange = samples > 0 && samples <= maxSamples
    if !inRange || (stated != nil && stated != samples * 4) {
        // A stated size that is sane gets consumed whatever else is wrong, which is what
        // keeps the stream in step after a bad request instead of parsing PCM as JSON.
        if let stated, stated > 0, stated <= maxSamples * 4 {
            guard reader.skip(stated) else { return nil }
        }
        let why = inRange
            ? "wrong byte count: \(samples) samples need \(samples * 4) bytes, request carried \(stated ?? -1)"
            : "samples=\(samples) is out of range (1...\(maxSamples))"
        return .bad(id: id, why: why)
    }

    var context: [String]?
    if let listed = header["context"], !(listed is NSNull) {
        guard let strings = listed as? [String] else {
            guard reader.skip(samples * 4) else { return nil }
            return .bad(id: id, why: "context must be a list of strings or null")
        }
        context = Array(strings.prefix(maxContext))
    }
    guard let pcm = reader.take(samples * 4) else {
        emit(["id": id ?? NSNull(), "error": "stdin closed before \(samples * 4) bytes of audio arrived"])
        return nil
    }
    return .utterance(id: id ?? -1, pcm: pcm, context: context)
}

/// Blocking reads get a thread of their own: a read(2) parked on a Swift concurrency
/// thread is one fewer thread for the recogniser to finish on.
func requests() -> AsyncStream<Incoming> {
    let (stream, feed) = AsyncStream<Incoming>.makeStream()
    let thread = Thread {
        let reader = StdinReader()
        while let incoming = nextRequest(reader) {
            feed.yield(incoming)
        }
        feed.finish()
    }
    thread.name = "apple-stt-stdin"
    thread.start()
    return stream
}

// MARK: - The recogniser

struct SetupError: Error {
    let message: String
    let exitCode: Int32
}

@available(macOS 26.0, *)
struct Session {
    let analyzer: SpeechAnalyzer
    let texts: Task<[String], Error>
}

/// Why every utterance gets an analyzer of its own, built ahead of time:
///
/// The obvious warm pattern -- one analyzer on an endless AsyncStream, yield a buffer,
/// `finalize(through: nil)` -- never returned from the first finalize on this machine
/// (macOS 27.0), with or without an explicit timeline. One analyzer fed a sequence per
/// utterance does work, but it hears the whole session as a single dictation: the
/// second command came back as " tight" and the third as " def me", lower-case and
/// continuing the previous sentence, and "AP Dhillon" fused into "APDhillon", where a
/// fresh analyzer gave "Tight", "Defend me" and "AP Dhillon". Commands are independent,
/// so each gets a clean one.
///
/// A fresh analyzer costs 130-160 ms to set up with a hundred contextual phrases (6-40 ms
/// without), so that is paid between sentences: the next session is prepared the moment
/// the current one is handed out, and the model itself stays loaded for the life of
/// the process. Measured here: 55-130 ms per 1-3 s utterance, the same as cold setup
/// minus the setup.
@available(macOS 26.0, *)
final class Engine {
    let localeID: String
    let format: AVAudioFormat
    private let locale: Locale
    private let wire = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: 16_000,
                                     channels: 1, interleaved: false)!
    private var context: [String] = []
    private var next: Task<Session, Error>

    private init(locale: Locale, format: AVAudioFormat) {
        self.locale = locale
        self.localeID = locale.identifier(.bcp47)
        self.format = format
        self.next = Engine.prepare(locale: locale, format: format, context: [])
    }

    static func transcriber(_ locale: Locale) -> DictationTranscriber {
        // shortForm: these are commands, not paragraphs. No punctuation option, because
        // "Open Spotify." would only have to be stripped again on the other side.
        DictationTranscriber(locale: locale, contentHints: [.shortForm], transcriptionOptions: [],
                             reportingOptions: [], attributeOptions: [])
    }

    /// installedLocales, not AssetInventory.status: on this machine the inventory calls
    /// en-IN merely "supported" while the recogniser lists it as installed and
    /// transcribes with it, so trusting the inventory refused an engine that works.
    static func installed(_ locale: Locale) async -> Bool {
        let wanted = locale.identifier(.bcp47)
        return await DictationTranscriber.installedLocales.contains { $0.identifier(.bcp47) == wanted }
    }

    static func make(localeID: String, install: Bool) async throws -> Engine {
        let wanted = Locale(identifier: localeID)
        guard let locale = await DictationTranscriber.supportedLocale(equivalentTo: wanted) else {
            let known = await DictationTranscriber.supportedLocales.map { $0.identifier(.bcp47) }.sorted()
            throw SetupError(message: "locale \(localeID) is not supported by DictationTranscriber "
                             + "(supported: \(known.joined(separator: ", ")))", exitCode: 3)
        }
        let module = transcriber(locale)
        if !(await installed(locale)) && install {
            log("downloading the \(localeID) speech assets...")
            if let request = try await AssetInventory.assetInstallationRequest(supporting: [module]) {
                try await request.downloadAndInstall()
            }
        }
        guard await installed(locale) else {
            // Never downloaded behind the user's back: it is hundreds of megabytes.
            throw SetupError(message: "speech assets for \(localeID) are not installed; "
                             + "run the sidecar once with --install", exitCode: 4)
        }
        guard let format = await SpeechAnalyzer.bestAvailableAudioFormat(compatibleWith: [module]) else {
            throw SetupError(message: "no audio format available for \(localeID)", exitCode: 5)
        }
        let engine = Engine(locale: locale, format: format)
        try await engine.warmUp()
        return engine
    }

    private static func prepare(locale: Locale, format: AVAudioFormat,
                                context: [String]) -> Task<Session, Error> {
        Task {
            let module = transcriber(locale)
            let analyzer = SpeechAnalyzer(
                modules: [module],
                options: .init(priority: .userInitiated, modelRetention: .processLifetime))
            if !context.isEmpty {
                let phrases = AnalysisContext()
                phrases.contextualStrings[.general] = context
                try await analyzer.setContext(phrases)
            }
            try await analyzer.prepareToAnalyze(in: format)
            // The results sequence ends when the analyzer finishes, so awaiting this task
            // is a complete answer -- no sleeping and hoping the last result has landed.
            let texts = Task { () -> [String] in
                var pieces: [(start: Double, text: String)] = []
                for try await result in module.results {
                    let start = result.range.start.seconds
                    pieces.removeAll { $0.start == start }      // a later result replaces an earlier guess
                    pieces.append((start, String(result.text.characters)))
                }
                return pieces.sorted { $0.start < $1.start }.map(\.text)
            }
            return Session(analyzer: analyzer, texts: texts)
        }
    }

    /// The first inference loads the model (the first clip of every probe run was the
    /// slow one); pay that before "ready", not under the user's first sentence.
    private func warmUp() async throws {
        let started = DispatchTime.now()
        let count = 8_000
        var tone = Data(count: count * 4)
        tone.withUnsafeMutableBytes { raw in
            let samples = raw.bindMemory(to: Float32.self)
            for i in 0..<count { samples[i] = 0.05 * sin(Float(i) * 2 * .pi * 220 / 16_000) }
        }
        _ = try await transcribe(tone)
        log("warm in \(elapsedMs(since: started)) ms")
    }

    /// Replace the contextual phrases. The session already waiting was built with the
    /// old ones, so it is dropped and rebuilt; that costs this one utterance ~140 ms,
    /// and the phrases change about once in ten minutes.
    func setContext(_ phrases: [String]) async {
        guard phrases != context else { return }
        context = phrases
        let stale = next
        next = Engine.prepare(locale: locale, format: format, context: phrases)
        if let session = try? await stale.value {
            await session.analyzer.cancelAndFinishNow()
        }
    }

    func transcribe(_ pcm: Data) async throws -> String {
        let waiting = next
        // Queued before anything can throw: whatever happens to this utterance, the
        // next one still finds an analyzer being built for it.
        next = Engine.prepare(locale: locale, format: format, context: context)
        let session = try await waiting.value
        let (stream, feed) = AsyncStream<AnalyzerInput>.makeStream()
        feed.yield(AnalyzerInput(buffer: try buffer(from: pcm)))
        feed.finish()
        if let end = try await session.analyzer.analyzeSequence(stream) {
            try await session.analyzer.finalizeAndFinish(through: end)
        } else {
            await session.analyzer.cancelAndFinishNow()
        }
        let pieces = try await session.texts.value
        return pieces.map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { !$0.isEmpty }.joined(separator: " ")
    }

    private func buffer(from pcm: Data) throws -> AVAudioPCMBuffer {
        let frames = AVAudioFrameCount(pcm.count / 4)
        guard let source = AVAudioPCMBuffer(pcmFormat: wire, frameCapacity: frames),
              let channel = source.floatChannelData?[0] else {
            throw SetupError(message: "could not allocate \(frames) frames", exitCode: 0)
        }
        source.frameLength = frames
        // Every Mac this runs on is little-endian, which is what the wire is.
        pcm.withUnsafeBytes { raw in _ = memcpy(channel, raw.baseAddress!, Int(frames) * 4) }
        return try convert(source, to: format)
    }
}

final class FedOnce: @unchecked Sendable {
    var done = false
}

/// The model asks for Int16 here, and may ask for something else on another OS, so the
/// conversion goes through whatever bestAvailableAudioFormat answered.
func convert(_ source: AVAudioPCMBuffer, to format: AVAudioFormat) throws -> AVAudioPCMBuffer {
    if source.format == format { return source }
    guard let converter = AVAudioConverter(from: source.format, to: format) else {
        throw SetupError(message: "no converter from \(source.format) to \(format)", exitCode: 0)
    }
    converter.primeMethod = .none
    let ratio = format.sampleRate / source.format.sampleRate
    let capacity = AVAudioFrameCount(Double(source.frameLength) * ratio) + 2048
    guard let out = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: capacity) else {
        throw SetupError(message: "could not allocate the converted buffer", exitCode: 0)
    }
    let fed = FedOnce()
    var failure: NSError?
    _ = converter.convert(to: out, error: &failure) { _, status in
        if fed.done {
            status.pointee = .endOfStream
            return nil
        }
        fed.done = true
        status.pointee = .haveData
        return source
    }
    if let failure { throw failure }
    return out
}

func describe(_ format: AVAudioFormat) -> String {
    let sample: String
    switch format.commonFormat {
    case .pcmFormatInt16: sample = "Int16"
    case .pcmFormatInt32: sample = "Int32"
    case .pcmFormatFloat32: sample = "Float32"
    case .pcmFormatFloat64: sample = "Float64"
    default: sample = "other"
    }
    return "\(format.channelCount) ch, \(Int(format.sampleRate)) Hz, \(sample)"
}

func explain(_ error: Error) -> String {
    (error as? SetupError)?.message ?? String(describing: error)
}

// MARK: - Main

@available(macOS 26.0, *)
func serve(localeID: String, install: Bool) async -> Int32 {
    let engine: Engine
    do {
        engine = try await Engine.make(localeID: localeID, install: install)
    } catch {
        emit(["ready": false, "error": explain(error)])
        return (error as? SetupError)?.exitCode ?? 5
    }
    emit(["ready": true, "locale": engine.localeID, "format": describe(engine.format)])

    for await incoming in requests() {
        switch incoming {
        case .bad(let id, let why):
            log("bad request: \(why)")
            emit(["id": id ?? NSNull(), "error": why])
        case .utterance(let id, let pcm, let context):
            let started = DispatchTime.now()
            if let context { await engine.setContext(context) }
            do {
                let text = try await engine.transcribe(pcm)
                emit(["id": id, "text": text, "ms": elapsedMs(since: started)])
            } catch {
                // This utterance is lost to this engine; the next already has a fresh
                // analyzer on the way, and Python falls back to whisper for this one.
                log("utterance \(id) failed: \(explain(error))")
                emit(["id": id, "error": explain(error)])
            }
        }
    }
    return 0
}

@main
struct AppleSTT {
    static func main() async {
        // A parent that died mid-write must end us through emit(), not through a signal
        // that leaves a core report behind.
        signal(SIGPIPE, SIG_IGN)
        let arguments = Array(CommandLine.arguments.dropFirst())
        let localeID = arguments.first { !$0.hasPrefix("--") } ?? "en-IN"
        guard #available(macOS 26.0, *) else {
            emit(["ready": false, "error": "macOS 26 or later is needed (SpeechAnalyzer)"])
            exit(2)
        }
        exit(await serve(localeID: localeID, install: arguments.contains("--install")))
    }
}
