# Acoustic Scene Classification

An iOS application for real-time acoustic scene classification using deep learning models. The app supports two inference backends: ONNX Runtime (PANNs CNN14) and TensorFlow Lite (YAMNet).

## Setup Instructions

### Prerequisites

- Xcode 14.0 or later
- iOS 13.0 or later
- CocoaPods installed

### Building the Project

1. **Install Dependencies**
   ```bash
   pod install
   ```

2. **Download ONNX Model Files**

   **⚠️ IMPORTANT:** The ONNX model files are not included in the repository due to their large size (328 MB). You must download them manually:

   - Download from: [https://huggingface.co/pranjal-pravesh/PANNs_CNN14_ONNX/tree/main](https://huggingface.co/pranjal-pravesh/PANNs_CNN14_ONNX/tree/main)
   - Required files:
     - `Cnn14_16k.onnx` (86.6 kB)
     - `Cnn14_16k.onnx.data` (327 MB)
   - Save both files to: `Acoustic-scene/Acoustic-scene/PANNs_cnn14/`

3. **Add Files to Xcode Project**

   - Open `Acoustic-scene.xcworkspace` (not `.xcodeproj`)
   - Right-click on the project → "Add Files to Acoustic-scene..."
   - Add the `PANNs_cnn14` folder containing both ONNX files
   - Ensure "Copy items if needed" is checked
   - Verify Target Membership includes "Acoustic-scene"

4. **Add CSV Label File**

   - The `PANNs_class_labels_indices.csv` file should already be in the project
   - If missing, add it to the project with Target Membership enabled

5. **Build and Run**

   - Select your target device/simulator
   - Build and run the project (⌘R)

### Note on Model Files

- **TFLite models** (YAMNet): Already included in the bundle
- **ONNX models** (CNN14): Must be downloaded manually (gitignored due to size)
- The `.gitignore` file excludes large model files to keep the repository size manageable

## What the App Does

This application performs **acoustic scene classification** - it identifies and classifies different types of sounds and audio scenes in real-time. The app:

1. **Records Audio**: Captures audio from the device's microphone
2. **Processes Audio**: Converts the audio to the format required by the selected model
3. **Classifies Sounds**: Uses deep learning models to identify audio events and scenes
4. **Displays Results**: Shows the top 10 most likely sound classes with confidence scores

## Models Supported

### 1. PANNs CNN14 (ONNX Runtime)
- **Model**: CNN14 from PANNs (Pretrained Audio Neural Networks)
- **Classes**: 527 AudioSet classes
- **Input**: Raw waveform at 16kHz (mono)
- **Preprocessing**: Handled internally by the model
- **Output**: Probabilities for 527 audio event classes

### 2. YAMNet (TensorFlow Lite)
- **Model**: YAMNet (Yet Another Mobile Network)
- **Classes**: 521 AudioSet classes
- **Input**: Raw waveform at 16kHz (mono)
- **Preprocessing**: Handled internally by the model
- **Output**: Probabilities for 521 audio event classes

## Features

- **Dual Backend Support**: Switch between ONNX and TFLite inference engines
- **Real-time Classification**: Record audio and get instant classification results
- **Top 10 Results**: View the most likely sound classes with confidence percentages
- **Visual Feedback**: Progress bars and color-coded results for each backend
- **Performance Metrics**: Inference time displayed for each classification

## Project Structure

```
Acoustic-scene/
├── Acoustic-scene/
│   ├── ContentView.swift              # Main UI
│   ├── AudioPipeline.swift            # YAMNet ONNX classifier (legacy)
│   ├── CNN14Classifier.swift          # PANNs CNN14 ONNX classifier
│   ├── TFLiteClassifier.swift         # YAMNet TFLite classifier
│   ├── AudioRecorder.swift            # Audio recording functionality
│   ├── PANNs_cnn14/                   # ONNX model files (download required)
│   │   ├── Cnn14_16k.onnx
│   │   └── Cnn14_16k.onnx.data
│   ├── PANNs_class_labels_indices.csv # Class label mapping (527 classes)
│   └── yamnet_class_map.csv           # YAMNet class labels (521 classes)
├── Podfile                            # CocoaPods dependencies
└── README.md                          # This file
```

## Dependencies

- **onnxruntime-objc**: ONNX Runtime for iOS
- **TensorFlowLiteSwift**: TensorFlow Lite Swift API
- **AVFoundation**: Audio recording and processing

## Usage

1. **Select Backend**: Choose between "ONNX (CNN14)" or "TFLite (YAMNet)" using the segmented control
2. **Start Recording**: Tap the "Start" button to begin recording audio
3. **Stop Recording**: Tap the "Stop" button to process the audio
4. **View Results**: The top 10 classified sound classes will appear with confidence scores

## Permissions

The app requires microphone access. The permission prompt will appear on first launch.

## License

MIT License

## References

- [PANNs CNN14 Model](https://huggingface.co/pranjal-pravesh/PANNs_CNN14_ONNX)
- [YAMNet Model](https://github.com/tensorflow/models/tree/master/research/audioset/yamnet)
- [AudioSet Dataset](https://research.google.com/audioset/)

