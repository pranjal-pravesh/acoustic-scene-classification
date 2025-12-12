//
//  AudioPipeline.swift
//  Acoustic-scene
//
//  Handles audio capture, preprocessing, and YAMNet ONNX inference.
//

import Foundation
import AVFoundation
import Accelerate
import onnxruntime_objc

struct ClassificationResult {
    let index: Int
    let label: String
    let probability: Float
}

struct YamnetResult {
    let topClasses: [ClassificationResult]
}

enum YamnetError: Error, LocalizedError {
    case modelNotFound
    case failedToCreateSession(String)
    case missingOutput
    case emptyAudio

    var errorDescription: String? {
        switch self {
        case .modelNotFound:
            return "Unable to find ONNX model in bundle."
        case .failedToCreateSession(let message):
            return "Could not create ONNX session: \(message)"
        case .missingOutput:
            return "Model did not return scores."
        case .emptyAudio:
            return "No audio samples were captured."
        }
    }
}

/// Records mono 16 kHz PCM audio to a temporary WAV file.
final class AudioRecorder {
    private var recorder: AVAudioRecorder?

    private lazy var recordingURL: URL = {
        let dir = FileManager.default.temporaryDirectory
        return dir.appendingPathComponent("capture.wav")
    }()

    private let settings: [String: Any] = [
        AVFormatIDKey: kAudioFormatLinearPCM,
        AVSampleRateKey: 16_000,
        AVNumberOfChannelsKey: 1,
        AVLinearPCMBitDepthKey: 16,
        AVLinearPCMIsFloatKey: true,
        AVEncoderAudioQualityKey: AVAudioQuality.high.rawValue
    ]

    func startRecording() throws {
        let session = AVAudioSession.sharedInstance()
        try session.setCategory(.playAndRecord, mode: .default, options: .defaultToSpeaker)
        try session.setActive(true)

        recorder = try AVAudioRecorder(url: recordingURL, settings: settings)
        guard recorder?.record() == true else {
            throw AVError(.unknown)
        }
    }

    func stopRecording() -> URL? {
        recorder?.stop()
        let url = recorder?.url
        recorder = nil
        return url
    }
}

/// Generates log-mel spectrogram patches matching YAMNet's expected shape.
final class YamnetPreprocessor {
    private let sampleRate: Float = 16_000
    private let frameLength = 400
    private let frameStep = 160
    private let fftLength = 512
    private let numMels = 64
    private let numFrames = 96
    private let minFrequency: Float = 125.0
    private let maxFrequency: Float = 7500.0

    private lazy var window: [Float] = {
        var w = [Float](repeating: 0, count: frameLength)
        vDSP_hann_window(&w, vDSP_Length(frameLength), Int32(vDSP_HANN_NORM))
        return w
    }()

    private lazy var melFilter: [[Float]] = {
        let fftBins = fftLength / 2
        let lowMel = hzToMel(minFrequency)
        let highMel = hzToMel(maxFrequency)
        let melPoints = (0..<(numMels + 2)).map { i -> Float in
            let fraction = Float(i) / Float(numMels + 1)
            return lowMel + fraction * (highMel - lowMel)
        }
        let hzPoints = melPoints.map { melToHz($0) }
        let binPoints = hzPoints.map { floor((Float(fftLength) + 1) * $0 / sampleRate) }

        var filters: [[Float]] = Array(repeating: Array(repeating: 0, count: fftBins), count: numMels)
        for m in 0..<numMels {
            let left = Int(binPoints[m])
            let center = Int(binPoints[m + 1])
            let right = Int(binPoints[m + 2])
            if center == left || right == center { continue }

            for k in left..<center where k < fftBins {
                filters[m][k] = (Float(k) - Float(left)) / (Float(center) - Float(left))
            }
            for k in center..<right where k < fftBins {
                filters[m][k] = (Float(right) - Float(k)) / (Float(right) - Float(center))
            }
        }
        return filters
    }()

    /// Convert raw audio samples (one patch worth) to log-mel spectrogram
    func prepareInput(from samples: [Float]) throws -> [Float] {
        guard !samples.isEmpty else { throw YamnetError.emptyAudio }

        // Ensure we have exactly the samples needed: 95 * hop + window = 15_600
        let requiredLength = (numFrames - 1) * frameStep + frameLength  // 15,200 samples = 0.95s
        var padded = samples
        if padded.count < requiredLength {
            padded.append(contentsOf: repeatElement(0, count: requiredLength - padded.count))
        } else if padded.count > requiredLength {
            padded = Array(padded.prefix(requiredLength))
        }

        let fftBins = fftLength / 2
        var output = [Float](repeating: 0, count: numFrames * numMels)

        // Setup FFT
        let log2n = vDSP_Length(log2(Double(fftLength)))
        guard let fftSetup = vDSP_create_fftsetup(log2n, FFTRadix(kFFTRadix2)) else {
            throw YamnetError.emptyAudio
        }
        defer { vDSP_destroy_fftsetup(fftSetup) }

        for frame in 0..<numFrames {
            let start = frame * frameStep
            let frameSlice = padded[start..<start + frameLength]
            var frameBuffer = Array(frameSlice)
            
            // Apply window
            vDSP_vmul(frameBuffer, 1, window, 1, &frameBuffer, 1, vDSP_Length(frameLength))

            // Zero-pad to fftLength
            frameBuffer.append(contentsOf: repeatElement(0, count: fftLength - frameBuffer.count))

            // Prepare split complex for FFT (real data packed)
            // For real FFT: input is fftLength real values, output is fftBins+1 complex values
            var realp = frameBuffer
            var imagp = [Float](repeating: 0, count: fftLength)
            var splitComplex = DSPSplitComplex(realp: &realp, imagp: &imagp)
            
            // Perform real-to-complex FFT
            vDSP_fft_zrip(fftSetup, &splitComplex, 1, log2n, FFTDirection(FFT_FORWARD))

            // Calculate power spectrum (magnitude squared)
            var magnitudes = [Float](repeating: 0, count: fftBins)
            vDSP_zvmags(&splitComplex, 1, &magnitudes, 1, vDSP_Length(fftBins))

            for mel in 0..<numMels {
                let filter = melFilter[mel]
                var energy: Float = 0
                vDSP_dotpr(magnitudes, 1, filter, 1, &energy, vDSP_Length(fftBins))
                let logEnergy = log(max(energy, 1e-6))
                output[frame * numMels + mel] = logEnergy
            }
        }

        return output
    }
    
    /// Process multiple overlapping patches from longer audio
    /// Returns array of raw sample patches (each patch is ~0.96s with 50% overlap)
    func preparePatches(from samples: [Float]) throws -> [[Float]] {
        guard !samples.isEmpty else { throw YamnetError.emptyAudio }
        
        let patchLength = (numFrames - 1) * frameStep + frameLength  // 15,200 samples = 0.95s
        let patchHop = patchLength / 2  // 50% overlap (0.48s hop as per YAMNet guide)
        
        var patches: [[Float]] = []
        var start = 0
        
        // Extract overlapping patches
        while start + patchLength <= samples.count {
            let patch = Array(samples[start..<start + patchLength])
            patches.append(patch)
            start += patchHop
        }
        
        // If we have at least one patch, return them
        if !patches.isEmpty {
            return patches
        }
        
        // If audio is too short, pad to one patch
        var padded = samples
        if padded.count < patchLength {
            padded.append(contentsOf: repeatElement(0, count: patchLength - padded.count))
        }
        return [padded]
    }

    private func hzToMel(_ hz: Float) -> Float {
        return 2595 * log10(1 + hz / 700)
    }

    private func melToHz(_ mel: Float) -> Float {
        return 700 * (pow(10, mel / 2595) - 1)
    }
}

/// Loads YAMNet ONNX and produces top-1 prediction.
final class YamnetClassifier {
    static let shared = YamnetClassifier()

    private let env: ORTEnv
    private let session: ORTSession
    private let preprocessor = YamnetPreprocessor()
    private let labels = YamnetLabels.shared

    private init?() {
        do {
            let env = try ORTEnv(loggingLevel: ORTLoggingLevel.warning)
            self.env = env

            guard let modelPath = Bundle.main.path(forResource: "model", ofType: "onnx") else {
                print("YAMNet ONNX file not found in bundle.")
                return nil
            }

            let options = try ORTSessionOptions()
            let session = try ORTSession(env: env, modelPath: modelPath, sessionOptions: options)
            self.session = session
        } catch {
            print("ORT init error: \(error.localizedDescription)")
            return nil
        }
    }

    func predict(audioURL: URL) throws -> YamnetResult {
        let file = try AVAudioFile(forReading: audioURL)
        let frameCount = AVAudioFrameCount(file.length)
        guard let buffer = AVAudioPCMBuffer(pcmFormat: file.processingFormat, frameCapacity: frameCount) else {
            throw YamnetError.emptyAudio
        }
        try file.read(into: buffer)
        guard let channel = buffer.floatChannelData?.pointee else {
            throw YamnetError.emptyAudio
        }
        let samples = Array(UnsafeBufferPointer(start: channel, count: Int(buffer.frameLength)))

        // Process all patches from the audio (handles both short and long recordings)
        let patches = try preprocessor.preparePatches(from: samples)
        
        // Process each patch and collect scores
        var allScores: [[Float]] = []
        
        for patch in patches {
            // Convert patch to log-mel spectrogram
            let inputFloats = try preprocessor.prepareInput(from: patch)
            
            let inputData = NSMutableData(length: inputFloats.count * MemoryLayout<Float>.size)!
            inputFloats.withUnsafeBytes { rawBuffer in
                inputData.replaceBytes(in: NSRange(location: 0, length: inputData.length), withBytes: rawBuffer.baseAddress!)
            }

            let inputTensor = try ORTValue(tensorData: inputData,
                                           elementType: ORTTensorElementDataType.float,
                                           shape: [1, 1, 96, 64] as [NSNumber])

            let outputMap = try session.run(withInputs: ["audio": inputTensor],
                                            outputNames: ["class_scores"],
                                            runOptions: nil)
            
            guard let scoresValue = outputMap["class_scores"] else {
                throw YamnetError.missingOutput
            }

            let scoresData = try scoresValue.tensorData()
            let count = scoresData.length / MemoryLayout<Float>.size
            let pointer = scoresData.mutableBytes.bindMemory(to: Float.self, capacity: count)
            let scores = Array(UnsafeBufferPointer(start: pointer, count: count))
            
            allScores.append(scores)
        }
        
        // Aggregate scores across patches (mean aggregation as per YAMNet guide)
        let aggregatedScores = aggregateScores(allScores, method: .mean)
        
        // Apply sigmoid to convert logits to independent probabilities (multi-label classification)
        // Note: YAMNet ONNX typically outputs raw logits. If your model already outputs probabilities,
        // remove sigmoid and use scores directly. Verify by checking if values are already in [0,1].
        let probabilities = sigmoid(aggregatedScores)
        
        // Get top 10 classes
        let indexedProbabilities = probabilities.enumerated().map { (index: $0.offset, probability: $0.element) }
        let top10 = indexedProbabilities
            .sorted { $0.probability > $1.probability }
            .prefix(10)
            .map { item in
                ClassificationResult(
                    index: item.index,
                    label: labels.label(for: item.index),
                    probability: item.probability
                )
            }

        return YamnetResult(topClasses: top10)
    }
    
    enum AggregationMethod {
        case mean    // Average across patches (good for clip-level classification)
        case max     // Maximum across patches (good for detecting short events)
    }
    
    /// Aggregate scores across multiple patches
    private func aggregateScores(_ patchScores: [[Float]], method: AggregationMethod) -> [Float] {
        guard !patchScores.isEmpty, !patchScores[0].isEmpty else { return [] }
        
        let numClasses = patchScores[0].count
        var aggregated = [Float](repeating: 0, count: numClasses)
        
        switch method {
        case .mean:
            // Average across all patches
            for patch in patchScores {
                for (i, score) in patch.enumerated() {
                    aggregated[i] += score
                }
            }
            let count = Float(patchScores.count)
            aggregated = aggregated.map { $0 / count }
            
        case .max:
            // Maximum across all patches (for detecting short events)
            for patch in patchScores {
                for (i, score) in patch.enumerated() {
                    aggregated[i] = max(aggregated[i], score)
                }
            }
        }
        
        return aggregated
    }
    
    /// Apply sigmoid to convert logits to independent probabilities for multi-label classification
    /// Each probability is independent and represents the likelihood of that class being present
    private func sigmoid(_ input: [Float]) -> [Float] {
        return input.map { 1.0 / (1.0 + exp(-$0)) }
    }
}

/// Label helper that maps YAMNet class indices to AudioSet display names.
/// Loads from yamnet_class_map.csv with format: index,mid,display_name
final class YamnetLabels {
    static let shared = YamnetLabels()
    private let labels: [String]

    private init() {
        if let url = Bundle.main.url(forResource: "yamnet_class_map", withExtension: "csv"),
           let csv = try? String(contentsOf: url) {
            let lines = csv.components(separatedBy: .newlines)
                .filter { !$0.trimmingCharacters(in: .whitespaces).isEmpty }
            
            // Skip header line and parse
            var parsedLabels: [String] = []
            for (lineIndex, line) in lines.enumerated() {
                // Skip header (first line)
                if lineIndex == 0 {
                    continue
                }
                
                // Parse CSV line: index,mid,display_name
                // Handle quoted display names that may contain commas
                let parts = Self.parseCSVLine(line)
                
                if parts.count >= 3 {
                    // Verify index matches position (safety check)
                    if let csvIndex = Int(parts[0]), csvIndex == parsedLabels.count {
                        // parts[2] is the display_name (3rd column)
                        parsedLabels.append(parts[2])
                    } else {
                        // If index doesn't match, still add but log warning
                        parsedLabels.append(parts[2])
                    }
                }
            }
            
            labels = parsedLabels
            print("✅ Loaded \(labels.count) class labels from CSV")
        } else {
            labels = []
            print("⚠️ yamnet_class_map.csv not found - using index-based labels")
        }
    }

    func label(for index: Int) -> String {
        guard index >= 0 else { return "Class \(index)" }
        guard index < labels.count else { return "Class \(index)" }
        return labels[index]
    }
    
    /// Parse CSV line handling quoted fields that may contain commas
    private static func parseCSVLine(_ line: String) -> [String] {
        var result: [String] = []
        var current = ""
        var inQuotes = false
        
        for char in line {
            if char == "\"" {
                inQuotes.toggle()
            } else if char == "," && !inQuotes {
                result.append(current.trimmingCharacters(in: .whitespaces))
                current = ""
            } else {
                current.append(char)
            }
        }
        result.append(current.trimmingCharacters(in: .whitespaces))
        return result
    }
}
