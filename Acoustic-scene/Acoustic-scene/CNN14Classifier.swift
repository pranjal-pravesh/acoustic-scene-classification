//
//  CNN14Classifier.swift
//  Acoustic-scene
//
//  PANNs CNN14 ONNX classifier for audio classification.
//  Model expects raw waveform input (16kHz mono).
//

import Foundation
import AVFoundation
import Accelerate
import onnxruntime_objc

/// PANNs CNN14 ONNX classifier
/// Input: raw waveform [batch, time_samples] at 16kHz
/// Output: clip_scores [1, 527] - probabilities for 527 AudioSet classes
final class CNN14Classifier {
    static let shared = CNN14Classifier()
    
    private let env: ORTEnv
    private let session: ORTSession
    private let labels = AudioSetLabels.shared
    
    private init?() {
        do {
            let env = try ORTEnv(loggingLevel: ORTLoggingLevel.warning)
            self.env = env
            
            // Try multiple possible paths for the model
            var modelPath: String?
            
            // Try in subdirectory first
            if let path = Bundle.main.path(forResource: "Cnn14_16k", ofType: "onnx", inDirectory: "PANNs_cnn14") {
                modelPath = path
            } else if let path = Bundle.main.path(forResource: "Cnn14_16k", ofType: "onnx") {
                modelPath = path
            } else if let dir = Bundle.main.resourcePath {
                let fullPath = (dir as NSString).appendingPathComponent("PANNs_cnn14/Cnn14_16k.onnx")
                if FileManager.default.fileExists(atPath: fullPath) {
                    modelPath = fullPath
                }
            }
            
            guard let path = modelPath else {
                print("CNN14 ONNX file not found in bundle. Expected: PANNs_cnn14/Cnn14_16k.onnx")
                return nil
            }
            
            let options = try ORTSessionOptions()
            let session = try ORTSession(env: env, modelPath: path, sessionOptions: options)
            self.session = session
            print("✅ CNN14 ONNX model loaded")
        } catch {
            print("ORT init error: \(error.localizedDescription)")
            return nil
        }
    }
    
    func predict(audioURL: URL) throws -> YamnetResult {
        // Load audio and convert to 16kHz mono float32
        let waveform = try loadAudio(url: audioURL)
        
        // CNN14 expects raw waveform input
        // Input shape: [1, num_samples] or [num_samples] depending on model
        let inputData = NSMutableData(length: waveform.count * MemoryLayout<Float>.size)!
        waveform.withUnsafeBytes { rawBuffer in
            inputData.replaceBytes(in: NSRange(location: 0, length: inputData.length), withBytes: rawBuffer.baseAddress!)
        }
        
        // Create input tensor - shape [1, num_samples]
        let inputTensor = try ORTValue(tensorData: inputData,
                                       elementType: ORTTensorElementDataType.float,
                                       shape: [1, waveform.count] as [NSNumber])
        
        // Run inference
        let outputMap = try session.run(withInputs: ["input_audio": inputTensor],
                                        outputNames: ["clip_scores"],
                                        runOptions: nil)
        
        guard let scoresValue = outputMap["clip_scores"] else {
            throw YamnetError.missingOutput
        }
        
        let scoresData = try scoresValue.tensorData()
        let count = scoresData.length / MemoryLayout<Float>.size
        let pointer = scoresData.mutableBytes.bindMemory(to: Float.self, capacity: count)
        let scores = Array(UnsafeBufferPointer(start: pointer, count: count))
        
        // CNN14 outputs probabilities directly (527 classes)
        // No sigmoid needed - outputs are already in [0, 1] range
        
        // Get top 10 classes
        let indexedProbabilities = scores.enumerated().map { (index: $0.offset, probability: $0.element) }
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
    
    /// Load audio file and convert to 16kHz mono float32
    private func loadAudio(url: URL) throws -> [Float] {
        let file = try AVAudioFile(forReading: url)
        let format = file.processingFormat
        let frameCount = AVAudioFrameCount(file.length)
        
        guard let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: frameCount) else {
            throw YamnetError.emptyAudio
        }
        try file.read(into: buffer)
        
        guard let channelData = buffer.floatChannelData else {
            throw YamnetError.emptyAudio
        }
        
        var floatData: [Float]
        
        // Convert to mono if stereo
        if format.channelCount == 2 {
            let left = channelData[0]
            let right = channelData[1]
            floatData = (0..<Int(buffer.frameLength)).map { i in
                (left[i] + right[i]) / 2.0
            }
        } else {
            floatData = Array(UnsafeBufferPointer(start: channelData[0], count: Int(buffer.frameLength)))
        }
        
        // Resample to 16kHz if needed
        let sampleRate = Int(format.sampleRate)
        if sampleRate != 16000 {
            floatData = resampleTo16k(data: floatData, originalRate: sampleRate)
        }
        
        return floatData
    }
    
    /// Simple linear resampler (for demo - use high-quality DSP in production)
    private func resampleTo16k(data: [Float], originalRate: Int) -> [Float] {
        let targetRate = 16000
        let ratio = Float(originalRate) / Float(targetRate)
        let newLength = Int(Float(data.count) / ratio)
        
        var out = [Float](repeating: 0, count: newLength)
        for i in 0..<newLength {
            let idx = Int(Float(i) * ratio)
            out[i] = idx < data.count ? data[idx] : 0
        }
        return out
    }
}

/// Label helper for AudioSet 527 classes (used by CNN14)
/// Loads labels from PANNs_class_labels_indices.csv
final class AudioSetLabels {
    static let shared = AudioSetLabels()
    private let labels: [String]
    
    private init() {
        // Try to load PANNs class map (527 classes)
        if let url = Bundle.main.url(forResource: "PANNs_class_labels_indices", withExtension: "csv"),
           let csv = try? String(contentsOf: url) {
            labels = Self.parseCSV(csv)
            print("✅ Loaded \(labels.count) AudioSet class labels from CSV")
        } else {
            // Fallback: use index-based labels
            labels = []
            print("⚠️ PANNs_class_labels_indices.csv not found - using index-based labels")
        }
    }
    
    func label(for index: Int) -> String {
        guard index >= 0 else { return "Class \(index)" }
        guard index < labels.count else { return "Class \(index)" }
        return labels[index]
    }
    
    private static func parseCSV(_ csv: String) -> [String] {
        let lines = csv.components(separatedBy: .newlines)
            .filter { !$0.trimmingCharacters(in: .whitespaces).isEmpty }
        
        var parsedLabels: [String] = []
        for (lineIndex, line) in lines.enumerated() {
            if lineIndex == 0 { continue } // Skip header
            
            let parts = parseCSVLine(line)
            if parts.count >= 3 {
                // Verify index matches (safety check)
                if let csvIndex = Int(parts[0]), csvIndex == parsedLabels.count {
                    parsedLabels.append(parts[2]) // display_name is 3rd column
                } else {
                    // If index doesn't match, still append (handles missing indices)
                    parsedLabels.append(parts[2])
                }
            }
        }
        return parsedLabels
    }
    
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

