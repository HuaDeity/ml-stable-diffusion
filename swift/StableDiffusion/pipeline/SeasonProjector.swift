// For licensing see accompanying LICENSE.md file.
// Copyright (C) 2022 Apple Inc. All Rights Reserved.

import Foundation
import CoreML

/// A model for projecting season values to embeddings
@available(iOS 16.2, macOS 13.1, *)
public struct SeasonProjector: ResourceManaging {

    /// Season projector model
    var model: ManagedMLModel

    /// Create season projector from Core ML model
    ///
    /// - Parameters:
    ///     - url: Location of compiled SeasonProjector Core ML model
    ///     - configuration: configuration to be used when the model is loaded
    /// - Returns: A season projector that will lazily load its required resources when needed or requested
    public init(modelAt url: URL, configuration: MLModelConfiguration) {
        self.model = ManagedMLModel(modelAt: url, configuration: configuration)
    }

    /// Ensure the model has been loaded into memory
    public func loadResources() throws {
        try model.loadResources()
    }

    /// Unload the underlying model to free up memory
    public func unloadResources() {
        model.unloadResources()
    }

    /// Project season value to embedding
    ///
    /// - Parameters:
    ///   - seasonValue: Float value in [0, 1]
    /// - Returns: Season embedding as MLShapedArray
    public func project(seasonValue: Float) throws -> MLShapedArray<Float32> {
        let inputName = "season_value"
        let inputValue = MLShapedArray<Float32>(scalars: [seasonValue], shape: [1, 1])
        
        let dict = [inputName: MLMultiArray(inputValue)]
        
        // Add contains_snow if required (default 0)
        // We can check model description if needed, but for now let's try just season_value
        // If the model was converted with optional inputs, this might be tricky.
        // But our torch2coreml logic added contains_snow to the trace if enabled.
        // Let's assume we might need it.
        
        // Check input description
        var needsSnow = false
        try? model.perform { model in
            if model.modelDescription.inputDescriptionsByName["contains_snow"] != nil {
                needsSnow = true
            }
        }
        
        var inputs = dict
        if needsSnow {
            inputs["contains_snow"] = MLMultiArray(MLShapedArray<Float32>(scalars: [0.0], shape: [1, 1]))
        }
        
        let input = try MLDictionaryFeatureProvider(dictionary: inputs)

        let result = try model.perform { model in
            try model.prediction(from: input)
        }

        let outputName = "season_embeddings"
        guard let outputValue = result.featureValue(for: outputName)?.multiArrayValue else {
            throw SeasonProjectorError.missingOutput
        }

        return MLShapedArray<Float32>(converting: outputValue)
    }
    
    enum SeasonProjectorError: Error {
        case missingOutput
    }
}
