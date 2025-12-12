# Uncomment the next line to define a global platform for your project
#platform :ios, '11.0'    
target 'Acoustic-scene' do
  use_frameworks!

  # --- ONNX Runtime ---
  pod 'onnxruntime-objc'

  # --- TensorFlow Lite ---
  pod 'TensorFlowLiteSwift'        # Swift API for TFLite
  # If you prefer the C API instead of Swift, use this (not both):
  # pod 'TensorFlowLiteC'

  # Optional: TFLite GPU delegate (only for some models)
  # pod 'TensorFlowLiteSwift/Metal'

  target 'Acoustic-sceneTests' do
    inherit! :search_paths
  end

  target 'Acoustic-sceneUITests' do
  end
end
