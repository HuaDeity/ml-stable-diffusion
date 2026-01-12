import CoreML
import Foundation

@available(iOS 16.2, macOS 13.1, *)
public struct MaskProcessor: ResourceManaging {

    var model: ManagedMLModel

    public init(modelAt url: URL, configuration: MLModelConfiguration) {
        self.model = ManagedMLModel(modelAt: url, configuration: configuration)
    }

    public func loadResources() throws {
        try model.loadResources()
    }

    public func unloadResources() {
        model.unloadResources()
    }

    public func prewarmResources() throws {
        try model.loadResources()
        model.unloadResources()
    }

    public func maskFeatures(_ mask: MLShapedArray<Float32>) throws -> MLShapedArray<Float32> {
        let inputName = inputDescription.name
        let input = try MLDictionaryFeatureProvider(dictionary: [inputName: MLMultiArray(mask)])
        let result = try model.perform { model in
            try model.prediction(from: input)
        }
        guard let output = result.featureValue(for: "mask_features")?.multiArrayValue else {
            throw MaskProcessorError.missingOutput
        }
        return MLShapedArray<Float32>(converting: output)
    }

    var inputDescription: MLFeatureDescription {
        try! model.perform { model in
            model.modelDescription.inputDescriptionsByName.first!.value
        }
    }

    enum MaskProcessorError: Error {
        case missingOutput
    }
}
