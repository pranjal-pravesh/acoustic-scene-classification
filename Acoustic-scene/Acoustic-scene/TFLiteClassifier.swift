//
//  TFLiteClassifier.swift
//  Acoustic-scene
//
//  YAMNet inference using TensorFlow Lite.
//  TFLite YAMNet does preprocessing internally - just feed raw waveform.
//

import Foundation
import AVFoundation
import TensorFlowLite

/// YAMNet TFLite classifier
/// Key difference from ONNX: TFLite model does mel-spectrogram preprocessing internally
final class YamnetTFLiteClassifier {
    static let shared = YamnetTFLiteClassifier()
    
    private var interpreter: Interpreter?
    private let labels = YamnetLabels.shared
    
    private init() {
        guard let modelPath = Bundle.main.path(forResource: "yamnet", ofType: "tflite") else {
            print("❌ YAMNet TFLite model not found in bundle")
            return
        }
        
        do {
            interpreter = try Interpreter(modelPath: modelPath)
            try interpreter?.allocateTensors()
            print("✅ TFLite interpreter initialized")
        } catch {
            print("❌ TFLite interpreter init error: \(error)")
        }
    }
    
    /// Run inference on audio file
    func predict(audioURL: URL) throws -> YamnetResult {
        guard let interpreter = interpreter else {
            throw YamnetError.modelNotFound
        }
        
        // Load and preprocess audio to 16kHz mono float32
        let waveform = try loadAudio(url: audioURL)
        
        // TFLite YAMNet expects raw waveform, does mel-spectrogram internally
        // Input shape: [num_samples] float32
        let inputData = Data(bytes: waveform, count: waveform.count * MemoryLayout<Float>.size)
        
        // Resize input tensor to match waveform length
        try interpreter.resizeInput(at: 0, to: [waveform.count])
        try interpreter.allocateTensors()
        
        // Copy input data
        try interpreter.copy(inputData, toInputAt: 0)
        
        // Run inference
        try interpreter.invoke()
        
        // Get output tensors
        // Output 0: scores [num_patches, 521]
        // Output 1: embeddings [num_patches, 1024]  
        // Output 2: spectrogram [num_frames, 64]
        let scoresTensor = try interpreter.output(at: 0)
        let scoresData = scoresTensor.data
        
        // Convert to float array
        let scores = scoresData.toFloatArray()
        
        // Reshape scores → [[patch][521]]
        let classCount = 521
        let numPatches = scores.count / classCount
        
        guard numPatches > 0 else {
            throw YamnetError.missingOutput
        }
        
        // Get scores for each patch
        var patchScores: [[Float]] = []
        for i in 0..<numPatches {
            let start = i * classCount
            let end = start + classCount
            patchScores.append(Array(scores[start..<end]))
        }
        
        // Aggregate scores across patches (mean)
        var aggregated = [Float](repeating: 0, count: classCount)
        for patch in patchScores {
            for (i, score) in patch.enumerated() {
                aggregated[i] += score
            }
        }
        aggregated = aggregated.map { $0 / Float(numPatches) }
        
        // TFLite YAMNet outputs are already sigmoid-activated (probabilities)
        // No need to apply sigmoid again
        
        // Get top 10 classes
        let indexedProbabilities = aggregated.enumerated().map { (index: $0.offset, probability: $0.element) }
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

// MARK: - Data Extension for TFLite

extension Data {
    /// Convert Data to Float array
    func toFloatArray() -> [Float] {
        var floatArray = [Float](repeating: 0, count: self.count / MemoryLayout<Float>.size)
        _ = floatArray.withUnsafeMutableBytes { self.copyBytes(to: $0) }
        return floatArray
    }
}


