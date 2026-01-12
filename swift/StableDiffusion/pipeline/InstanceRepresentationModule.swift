import CoreML
import Foundation

@available(iOS 16.2, macOS 13.1, *)
public struct InstanceRepresentationModule: ResourceManaging {

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

    public func instanceRepresentation(_ features: MLShapedArray<Float32>) throws -> MLShapedArray<Float32> {
        let inputName = inputDescription.name
        let input = try MLDictionaryFeatureProvider(dictionary: [inputName: MLMultiArray(features)])
        let result = try model.perform { model in
            try model.prediction(from: input)
        }
        guard let output = result.featureValue(for: "instance_representation")?.multiArrayValue else {
            throw InstanceRepresentationError.missingOutput
        }
        return MLShapedArray<Float32>(converting: output)
    }

    var inputDescription: MLFeatureDescription {
        try! model.perform { model in
            model.modelDescription.inputDescriptionsByName.first!.value
        }
    }

    enum InstanceRepresentationError: Error {
        case missingOutput
    }
}
