//
//  ContentView.swift
//  Acoustic-scene
//
//  Created by Pranjal on 11/12/25.
//

import SwiftUI
import AVFoundation
import Combine

enum InferenceBackend: String, CaseIterable {
    case onnx = "ONNX Runtime"
    case tflite = "TensorFlow Lite"
}

final class AudioClassificationViewModel: ObservableObject {
    @Published var status: String = "Ready"
    @Published var topClasses: [ClassificationResult] = []
    @Published var isRecording = false
    @Published var isProcessing = false
    @Published var inferenceTime: TimeInterval = 0
    @Published var backend: InferenceBackend = .onnx

    private let recorder = AudioRecorder()
    private let onnxClassifier = CNN14Classifier.shared
    private let tfliteClassifier = YamnetTFLiteClassifier.shared

    func start() {
        guard !isRecording else { return }
        AVAudioSession.sharedInstance().requestRecordPermission { [weak self] allowed in
            guard let self else { return }
            DispatchQueue.main.async {
                if allowed {
                    do {
                        try self.recorder.startRecording()
                        self.status = "Recording..."
                        self.topClasses = []
                        self.inferenceTime = 0
                        self.isRecording = true
                    } catch {
                        self.status = "Record failed: \(error.localizedDescription)"
                    }
                } else {
                    self.status = "Microphone permission denied"
                }
            }
        }
    }

    func stop() {
        guard isRecording else { return }
        isRecording = false
        status = "Processing with \(backend.rawValue)..."
        isProcessing = true

        guard let url = recorder.stopRecording() else {
            status = "No recording found"
            isProcessing = false
            return
        }

        let currentBackend = backend
        
        Task.detached { [weak self] in
            guard let self else { return }
            
            let startTime = CFAbsoluteTimeGetCurrent()
            
            do {
                let output: YamnetResult
                
                switch currentBackend {
                case .onnx:
                    guard let classifier = self.onnxClassifier else {
                        throw YamnetError.modelNotFound
                    }
                    output = try classifier.predict(audioURL: url)
                    
                case .tflite:
                    output = try self.tfliteClassifier.predict(audioURL: url)
                }
                
                let elapsed = CFAbsoluteTimeGetCurrent() - startTime
                
                await MainActor.run {
                    self.topClasses = output.topClasses
                    self.inferenceTime = elapsed
                    self.status = "Done (\(String(format: "%.2f", elapsed))s) - \(output.topClasses.first?.label ?? "Unknown")"
                    self.isProcessing = false
                }
            } catch {
                await MainActor.run {
                    self.status = "Error: \(error.localizedDescription)"
                    self.isProcessing = false
                }
            }
        }
    }
}

struct ContentView: View {
    @StateObject private var viewModel = AudioClassificationViewModel()

    var body: some View {
        VStack(spacing: 12) {
            // Header
            Text("YAMNet Audio Classification")
                .font(.title2)
                .bold()
                .padding(.top)
            
            // Backend Picker
            Picker("Backend", selection: $viewModel.backend) {
                ForEach(InferenceBackend.allCases, id: \.self) { backend in
                    Text(backend.rawValue).tag(backend)
                }
            }
            .pickerStyle(.segmented)
            .padding(.horizontal)
            .disabled(viewModel.isRecording || viewModel.isProcessing)

            // Status
            Text(viewModel.status)
                .font(.footnote)
                .foregroundStyle(.secondary)
                .multilineTextAlignment(.center)
                .padding(.horizontal)

            // Record Controls
            HStack(spacing: 16) {
                Button(action: viewModel.start) {
                    Label("Start", systemImage: "circle.fill")
                        .frame(maxWidth: .infinity)
                }
                .buttonStyle(.borderedProminent)
                .tint(.green)
                .disabled(viewModel.isRecording || viewModel.isProcessing)

                Button(action: viewModel.stop) {
                    Label("Stop", systemImage: "stop.fill")
                        .frame(maxWidth: .infinity)
                }
                .buttonStyle(.bordered)
                .tint(.red)
                .disabled(!viewModel.isRecording || viewModel.isProcessing)
            }
            .padding(.horizontal)

            if viewModel.isProcessing {
                ProgressView("Running \(viewModel.backend.rawValue)...")
                    .progressViewStyle(.circular)
                    .padding()
            }

            // Results
            if !viewModel.topClasses.isEmpty {
                VStack(alignment: .leading, spacing: 8) {
                    HStack {
                        Text("Top 10 Classes")
                            .font(.headline)
                        Spacer()
                        if viewModel.inferenceTime > 0 {
                            Text(String(format: "%.2fs", viewModel.inferenceTime))
                                .font(.caption)
                                .foregroundStyle(.secondary)
                        }
                    }
                    .padding(.horizontal)
                    
                    ScrollView {
                        VStack(spacing: 4) {
                            ForEach(Array(viewModel.topClasses.enumerated()), id: \.offset) { index, result in
                                ResultRow(index: index, result: result, backend: viewModel.backend)
                            }
                        }
                    }
                }
            } else {
                Spacer()
                VStack(spacing: 8) {
                    Image(systemName: "waveform.circle")
                        .font(.system(size: 48))
                        .foregroundStyle(.secondary)
                    Text("Record audio to classify")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
                Spacer()
            }

            // Footer
            Group {
                if viewModel.backend == .onnx {
                    Text("Using ONNX Runtime • CNN14 (527 classes)")
                } else {
                    Text("Using TensorFlow Lite • YAMNet (521 classes)")
                }
            }
            .font(.caption2)
            .foregroundStyle(.secondary)
            .padding(.bottom, 8)
        }
    }
}

struct ResultRow: View {
    let index: Int
    let result: ClassificationResult
    let backend: InferenceBackend
    
    var barColor: Color {
        backend == .onnx ? .blue : .orange
    }
    
    var body: some View {
        HStack {
            Text("\(index + 1).")
                .font(.caption)
                .foregroundStyle(.secondary)
                .frame(width: 25, alignment: .trailing)
            
            Text(result.label)
                .font(.body)
                .lineLimit(1)
            
            Spacer()
            
            Text(String(format: "%.1f%%", result.probability * 100))
                .font(.caption)
                .foregroundStyle(.secondary)
                .monospacedDigit()
            
            // Progress bar
            GeometryReader { geo in
                ZStack(alignment: .leading) {
                    Rectangle()
                        .fill(Color.gray.opacity(0.2))
                        .frame(height: 4)
                    
                    Rectangle()
                        .fill(barColor)
                        .frame(width: geo.size.width * CGFloat(min(result.probability, 1.0)), height: 4)
                }
            }
            .frame(width: 60, height: 4)
        }
        .padding(.horizontal)
        .padding(.vertical, 4)
    }
}

#Preview {
    ContentView()
}
