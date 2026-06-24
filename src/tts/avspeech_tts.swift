// AVSpeechSynthesizer-based TTS helper. Unlike `say -o`, this code path can
// use the neural Siri voices.
//
// Build:
//   swiftc -O -o avspeech_tts avspeech_tts.swift
//
// Usage:
//   avspeech_tts --list
//     Print one line per installed voice: identifier|name|language|quality
//   avspeech_tts <text> [voice-identifier] [sample-rate]
//     Synthesize <text> and write raw int16 LE PCM mono to stdout at the
//     requested sample rate (default 22050). Empty voice-identifier = system
//     default voice.

import AVFoundation
import Foundation

let args = CommandLine.arguments

if args.count >= 2 && args[1] == "--list" {
    for v in AVSpeechSynthesisVoice.speechVoices() {
        let quality: String
        switch v.quality {
        case .default: quality = "default"
        case .enhanced: quality = "enhanced"
        case .premium: quality = "premium"
        @unknown default: quality = "unknown"
        }
        print("\(v.identifier)|\(v.name)|\(v.language)|\(quality)")
    }
    exit(0)
}

guard args.count >= 2 else {
    FileHandle.standardError.write(
        "usage: avspeech_tts <text> [voice-id] [sample-rate]\n".data(using: .utf8)!
    )
    exit(2)
}

let text = args[1]
let voiceID: String = args.count >= 3 ? args[2] : ""
let targetSR: Double = args.count >= 4 ? (Double(args[3]) ?? 22050) : 22050

let synth = AVSpeechSynthesizer()
let utterance = AVSpeechUtterance(string: text)
if !voiceID.isEmpty, let v = AVSpeechSynthesisVoice(identifier: voiceID) {
    utterance.voice = v
}

let outFormat = AVAudioFormat(
    commonFormat: .pcmFormatInt16,
    sampleRate: targetSR,
    channels: 1,
    interleaved: true
)!
var converter: AVAudioConverter?
let stdout = FileHandle.standardOutput
let done = DispatchSemaphore(value: 0)

synth.write(utterance) { buffer in
    guard let pcm = buffer as? AVAudioPCMBuffer else {
        done.signal()
        return
    }
    if pcm.frameLength == 0 {
        // End-of-stream sentinel from AVSpeechSynthesizer.
        done.signal()
        return
    }
    if converter == nil {
        converter = AVAudioConverter(from: pcm.format, to: outFormat)
    }
    let ratio = outFormat.sampleRate / pcm.format.sampleRate
    let cap = AVAudioFrameCount(Double(pcm.frameLength) * ratio + 1024)
    guard let out = AVAudioPCMBuffer(pcmFormat: outFormat, frameCapacity: cap) else { return }
    var err: NSError?
    var consumed = false
    _ = converter!.convert(to: out, error: &err) { _, status in
        if consumed {
            status.pointee = .endOfStream
            return nil
        }
        consumed = true
        status.pointee = .haveData
        return pcm
    }
    if let ch = out.int16ChannelData, out.frameLength > 0 {
        let n = Int(out.frameLength)
        let data = Data(bytes: ch[0], count: n * MemoryLayout<Int16>.size)
        stdout.write(data)
    }
}

done.wait()
