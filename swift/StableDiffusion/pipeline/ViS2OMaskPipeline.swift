import Accelerate
import CoreGraphics
import CoreML
import Foundation

@available(iOS 16.2, macOS 13.1, *)
public struct ViS2OMaskPipeline: ResourceManaging {

    public struct Configuration {
        public var prompt: String
        public var startingImage: CGImage
        public var instanceMasks: [CGImage]
        public var instanceTexts: [String]
        public var stepCount: Int = 50
        public var seed: UInt32 = 0
        public var guidanceScale: Float = 1.5
        public var imageGuidanceScale: Float = 1.5
        public var schedulerType: StableDiffusionScheduler = .pndmScheduler
        public var rngType: StableDiffusionRNG = .numpyRNG
        public var encoderScaleFactor: Float32 = 0.18215
        public var decoderScaleFactor: Float32 = 0.18215
        public var imageCount: Int = 1
        public var useDenoisedIntermediates: Bool = false

        public init(
            prompt: String,
            startingImage: CGImage,
            instanceMasks: [CGImage],
            instanceTexts: [String]
        ) {
            self.prompt = prompt
            self.startingImage = startingImage
            self.instanceMasks = instanceMasks
            self.instanceTexts = instanceTexts
        }
    }

    public struct Progress {
        public let step: Int
        public let stepCount: Int
    }

    var textEncoder: TextEncoder
    var unet: Unet
    var decoder: Decoder
    var encoder: Encoder
    var maskProcessor: MaskProcessor
    var instanceRepresentationModule: InstanceRepresentationModule
    var reduceMemory: Bool = false

    public init(
        textEncoder: TextEncoder,
        unet: Unet,
        decoder: Decoder,
        encoder: Encoder,
        maskProcessor: MaskProcessor,
        instanceRepresentationModule: InstanceRepresentationModule,
        reduceMemory: Bool = false
    ) {
        self.textEncoder = textEncoder
        self.unet = unet
        self.decoder = decoder
        self.encoder = encoder
        self.maskProcessor = maskProcessor
        self.instanceRepresentationModule = instanceRepresentationModule
        self.reduceMemory = reduceMemory
    }

    public func loadResources() throws {
        if reduceMemory {
            try prewarmResources()
        } else {
            try textEncoder.loadResources()
            try unet.loadResources()
            try decoder.loadResources()
            try encoder.loadResources()
            try maskProcessor.loadResources()
            try instanceRepresentationModule.loadResources()
        }
    }

    public func unloadResources() {
        textEncoder.unloadResources()
        unet.unloadResources()
        decoder.unloadResources()
        encoder.unloadResources()
        maskProcessor.unloadResources()
        instanceRepresentationModule.unloadResources()
    }

    public func prewarmResources() throws {
        try textEncoder.prewarmResources()
        try unet.prewarmResources()
        try decoder.prewarmResources()
        try encoder.prewarmResources()
        try maskProcessor.prewarmResources()
        try instanceRepresentationModule.prewarmResources()
    }

    public func generateImages(
        configuration config: Configuration,
        progressHandler: (Progress) -> Bool = { _ in true }
    ) throws -> [CGImage?] {
        if config.instanceMasks.count != config.instanceTexts.count {
            throw ViS2OMaskError.mismatchedInstanceInputs
        }
        if config.instanceMasks.isEmpty {
            throw ViS2OMaskError.missingInstanceMasks
        }

        if let expected = expectedInstanceCount(), expected != config.instanceMasks.count {
            throw ViS2OMaskError.unexpectedInstanceCount(expected: expected, actual: config.instanceMasks.count)
        }

        let promptEmbedding = try textEncoder.encode(config.prompt)
        let hiddenStates = toHiddenStates(promptEmbedding)
        let batchHiddenStates = MLShapedArray<Float32>(
            concatenating: [hiddenStates, hiddenStates, hiddenStates],
            alongAxis: 0
        )

        let (instanceRep, nullInstanceRep, maskBatch) = try buildInstanceInputs(
            masks: config.instanceMasks,
            texts: config.instanceTexts
        )
        let batchInstanceRep = MLShapedArray<Float32>(
            concatenating: [instanceRep, nullInstanceRep, nullInstanceRep],
            alongAxis: 0
        )

        // Build batch masks: [mask, zeros, zeros]
        let zeroMasks = MLShapedArray<Float32>(repeating: 0.0, shape: maskBatch.shape)
        let batchMasks = MLShapedArray<Float32>(
            concatenating: [maskBatch, zeroMasks, zeroMasks],
            alongAxis: 0
        )

        let scheduler: [Scheduler] = (0..<config.imageCount).map { _ in
            switch config.schedulerType {
            case .pndmScheduler:
                return PNDMScheduler(stepCount: config.stepCount)
            case .dpmSolverMultistepScheduler:
                return DPMSolverMultistepScheduler(stepCount: config.stepCount, timeStepSpacing: .linspace)
            case .discreteFlowScheduler:
                return DiscreteFlowScheduler(stepCount: config.stepCount, timeStepShift: 3.0)
            case .eulerAncestralDiscreteScheduler:
                return EulerAncestralDiscreteScheduler(stepCount: config.stepCount, timestepSpacing: .linspace)
            }
        }

        var latents = try generateLatentSamples(
            configuration: config,
            scheduler: scheduler[0]
        )
        var denoisedLatents: [MLShapedArray<Float32>] = latents.map { MLShapedArray(converting: $0) }

        var random = randomSource(from: config.rngType, seed: config.seed)
        let encodedLatent = try encoder.encode(
            config.startingImage,
            scaleFactor: config.encoderScaleFactor,
            random: &random,
            useMeanOnly: true
        )
        let imageLatents = (0..<config.imageCount).map { _ in encodedLatent }

        let timeSteps = scheduler[0].calculateTimesteps(strength: nil)
        for (step, t) in timeSteps.enumerated() {
            let scaledLatents = zip(latents, scheduler).map { latent, scheduler in
                scheduler.scaleModelInput(sample: latent, timestep: t)
            }

            var latentUnetInput = scaledLatents.map {
                MLShapedArray<Float32>(concatenating: [$0, $0, $0], alongAxis: 0)
            }

            latentUnetInput = zip(latentUnetInput, imageLatents).map { noiseLatent, imgLatent in
                let zeros = MLShapedArray<Float32>(repeating: 0.0, shape: imgLatent.shape)
                let expandedImageLatent = MLShapedArray<Float32>(
                    concatenating: [imgLatent, imgLatent, zeros],
                    alongAxis: 0
                )
                return MLShapedArray<Float32>(
                    concatenating: [noiseLatent, expandedImageLatent],
                    alongAxis: 1
                )
            }

            let noise = try unet.predictNoise(
                latents: latentUnetInput,
                timeStep: t,
                hiddenStates: batchHiddenStates,
                instanceRepresentation: batchInstanceRep,
                instanceMasks: batchMasks
            )

            let guided = applyTripleGuidance(
                noise: noise,
                guidanceScale: config.guidanceScale,
                imageGuidanceScale: config.imageGuidanceScale
            )

            for i in 0..<config.imageCount {
                latents[i] = scheduler[i].step(
                    output: guided[i],
                    timeStep: t,
                    sample: latents[i]
                )
                denoisedLatents[i] = scheduler[i].modelOutputs.last ?? latents[i]
            }

            let progress = Progress(step: step, stepCount: timeSteps.count)
            if !progressHandler(progress) {
                return []
            }
        }

        let finalLatents = config.useDenoisedIntermediates ? denoisedLatents : latents
        return try decodeToImages(finalLatents, configuration: config)
    }

    func decodeToImages(
        _ latents: [MLShapedArray<Float32>],
        configuration config: Configuration
    ) throws -> [CGImage?] {
        return try decoder.decode(latents, scaleFactor: config.decoderScaleFactor)
    }

    func generateLatentSamples(configuration config: Configuration, scheduler: Scheduler) throws
        -> [MLShapedArray<Float32>]
    {
        var sampleShape = unet.latentSampleShape
        sampleShape[0] = 1
        if sampleShape[1] == 8 {
            sampleShape[1] = 4
        }

        let stdev = scheduler.initNoiseSigma
        var random = randomSource(from: config.rngType, seed: config.seed)
        let samples = (0..<config.imageCount).map { _ in
            MLShapedArray<Float32>(
                converting: random.normalShapedArray(sampleShape, mean: 0.0, stdev: Double(stdev)))
        }
        return samples
    }

    func applyTripleGuidance(
        noise: [MLShapedArray<Float32>],
        guidanceScale: Float,
        imageGuidanceScale: Float
    ) -> [MLShapedArray<Float32>] {
        noise.map { noisePred in
            var shape = noisePred.shape
            shape[0] = 1
            return MLShapedArray<Float32>(unsafeUninitializedShape: shape) { result, _ in
                noisePred.withUnsafeShapedBufferPointer { scalars, _, strides in
                    for i in 0..<result.count {
                        let fullPred = scalars[i]
                        let imgPred = scalars[strides[0] + i]
                        let uncondPred = scalars[2 * strides[0] + i]
                        let value = uncondPred
                            + imageGuidanceScale * (imgPred - uncondPred)
                            + guidanceScale * (fullPred - imgPred)
                        result.initializeElement(at: i, to: value)
                    }
                }
            }
        }
    }

    func buildInstanceInputs(
        masks: [CGImage],
        texts: [String]
    ) throws -> (MLShapedArray<Float32>, MLShapedArray<Float32>, MLShapedArray<Float32>) {
        let maskCount = masks.count
        let height = masks.first!.height
        let width = masks.first!.width

        var maskFeatures = [Float32]()
        maskFeatures.reserveCapacity(maskCount * 256)

        var maskBatchScalars = [Float32]()
        maskBatchScalars.reserveCapacity(maskCount * height * width)

        for mask in masks {
            let maskScalars = try maskScalars(from: mask, width: width, height: height)
            maskBatchScalars.append(contentsOf: maskScalars)

            let maskTensor = MLShapedArray<Float32>(
                scalars: maskScalars,
                shape: [1, 1, height, width]
            )
            let features = try maskProcessor.maskFeatures(maskTensor)
            let flatFeatures = flatten(features, expected: 256)
            maskFeatures.append(contentsOf: flatFeatures)
        }

        let textFeatures = try encodeInstanceTextFeatures(texts: texts)
        let normText = l2Normalize(textFeatures, rows: maskCount, cols: 768)
        let normMask = l2Normalize(maskFeatures, rows: maskCount, cols: 256)

        var concat = [Float32]()
        concat.reserveCapacity(maskCount * 1024)
        for i in 0..<maskCount {
            let tStart = i * 768
            let mStart = i * 256
            concat.append(contentsOf: normText[tStart..<(tStart + 768)])
            concat.append(contentsOf: normMask[mStart..<(mStart + 256)])
        }

        let instanceInput = MLShapedArray<Float32>(
            scalars: concat,
            shape: [1, maskCount, 1024]
        )
        let instanceRep = try instanceRepresentationModule.instanceRepresentation(instanceInput)

        let nullTexts = Array(repeating: "", count: maskCount)
        let nullMask = MLShapedArray<Float32>(
            repeating: 0.0,
            shape: [1, 1, height, width]
        )
        var nullMaskFeatures = [Float32]()
        nullMaskFeatures.reserveCapacity(maskCount * 256)
        for _ in 0..<maskCount {
            let features = try maskProcessor.maskFeatures(nullMask)
            nullMaskFeatures.append(contentsOf: flatten(features, expected: 256))
        }

        let nullTextFeatures = try encodeInstanceTextFeatures(texts: nullTexts)
        let normNullText = l2Normalize(nullTextFeatures, rows: maskCount, cols: 768)
        let normNullMask = l2Normalize(nullMaskFeatures, rows: maskCount, cols: 256)

        var nullConcat = [Float32]()
        nullConcat.reserveCapacity(maskCount * 1024)
        for i in 0..<maskCount {
            let tStart = i * 768
            let mStart = i * 256
            nullConcat.append(contentsOf: normNullText[tStart..<(tStart + 768)])
            nullConcat.append(contentsOf: normNullMask[mStart..<(mStart + 256)])
        }

        let nullInput = MLShapedArray<Float32>(
            scalars: nullConcat,
            shape: [1, maskCount, 1024]
        )
        let nullInstanceRep = try instanceRepresentationModule.instanceRepresentation(nullInput)

        let maskBatch = MLShapedArray<Float32>(
            scalars: maskBatchScalars,
            shape: [1, maskCount, height, width]
        )

        return (instanceRep, nullInstanceRep, maskBatch)
    }

    func encodeInstanceTextFeatures(texts: [String]) throws -> [Float32] {
        var features = [Float32]()
        features.reserveCapacity(texts.count * 768)
        for text in texts {
            let outputs = try textEncoder.encodeWithOutputs(text)
            let pooled = flatten(outputs.pooledOutput, expected: 768)
            features.append(contentsOf: pooled)
        }
        return features
    }

    func flatten(_ array: MLShapedArray<Float32>, expected: Int) -> [Float32] {
        if array.scalarCount == expected {
            return Array(array.scalars)
        }
        if array.scalarCount >= expected {
            return Array(array.scalars.prefix(expected))
        }
        var padded = Array(array.scalars)
        padded.append(contentsOf: Array(repeating: 0.0, count: expected - padded.count))
        return padded
    }

    func l2Normalize(_ scalars: [Float32], rows: Int, cols: Int) -> [Float32] {
        var result = [Float32](repeating: 0.0, count: rows * cols)
        for r in 0..<rows {
            let start = r * cols
            let end = start + cols
            var sum: Float32 = 0.0
            for i in start..<end {
                sum += scalars[i] * scalars[i]
            }
            let norm = sqrt(sum)
            if norm == 0 {
                continue
            }
            for i in start..<end {
                result[i] = scalars[i] / norm
            }
        }
        return result
    }

    func maskScalars(from image: CGImage, width: Int, height: Int) throws -> [Float32] {
        let bytes = try renderRGBABytes(from: image, width: width, height: height)
        var scalars = [Float32](repeating: 0.0, count: width * height)
        for i in 0..<(width * height) {
            let value = bytes[i * 4]
            scalars[i] = Float32(value) / 255.0
        }
        return scalars
    }

    func renderRGBABytes(from image: CGImage, width: Int, height: Int) throws -> [UInt8] {
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        let bytesPerRow = width * 4
        let bitmapInfo = CGImageAlphaInfo.premultipliedLast.rawValue

        guard let context = CGContext(
            data: nil,
            width: width,
            height: height,
            bitsPerComponent: 8,
            bytesPerRow: bytesPerRow,
            space: colorSpace,
            bitmapInfo: bitmapInfo
        ) else {
            throw ViS2OMaskError.invalidContext
        }

        context.interpolationQuality = .high
        context.draw(image, in: CGRect(x: 0, y: 0, width: width, height: height))

        guard let data = context.data else {
            throw ViS2OMaskError.missingContextData
        }

        let buffer = data.bindMemory(to: UInt8.self, capacity: width * height * 4)
        return Array(UnsafeBufferPointer(start: buffer, count: width * height * 4))
    }

    func expectedInstanceCount() -> Int? {
        guard let shape = unet.inputShape(named: "instance_representation"),
              shape.count >= 2 else {
            return nil
        }
        return shape[1]
    }
}

@available(iOS 16.2, macOS 13.1, *)
extension ViS2OMaskPipeline {
    func randomSource(from rng: StableDiffusionRNG, seed: UInt32) -> RandomSource {
        switch rng {
        case .numpyRNG:
            return NumPyRandomSource(seed: seed)
        case .torchRNG:
            return TorchRandomSource(seed: seed)
        case .nvidiaRNG:
            return NvRandomSource(seed: seed)
        }
    }

    func toHiddenStates(_ embedding: MLShapedArray<Float32>) -> MLShapedArray<Float32> {
        let fromShape = embedding.shape
        let stateShape = [fromShape[0], fromShape[2], 1, fromShape[1]]
        var states = MLShapedArray<Float32>(repeating: 0.0, shape: stateShape)
        for i0 in 0..<fromShape[0] {
            for i1 in 0..<fromShape[1] {
                for i2 in 0..<fromShape[2] {
                    states[scalarAt: i0, i2, 0, i1] = embedding[scalarAt: i0, i1, i2]
                }
            }
        }
        return states
    }
}

@available(iOS 16.2, macOS 13.1, *)
enum ViS2OMaskError: Error {
    case mismatchedInstanceInputs
    case unexpectedInstanceCount(expected: Int, actual: Int)
    case invalidContext
    case missingContextData
    case missingInstanceMasks
}
