import Foundation
import CoreML

@available(iOS 16.2, macOS 13.1, *)
public extension ViS2OMaskPipeline {

    struct ResourceURLs {
        public let textEncoderURL: URL
        public let unetURL: URL
        public let decoderURL: URL
        public let encoderURL: URL
        public let maskProcessorURL: URL
        public let instanceRepresentationURL: URL
        public let vocabURL: URL
        public let mergesURL: URL

        public init(resourcesAt baseURL: URL) throws {
            textEncoderURL = baseURL.appending(path: "TextEncoder.mlmodelc")
            unetURL = baseURL.appending(path: "Unet.mlmodelc")
            decoderURL = baseURL.appending(path: "VAEDecoder.mlmodelc")
            encoderURL = baseURL.appending(path: "VAEEncoder.mlmodelc")
            vocabURL = baseURL.appending(path: "vocab.json")
            mergesURL = baseURL.appending(path: "merges.txt")

            if let resolved = resolveModelURL(baseURL: baseURL, name: "MaskProcessor") {
                maskProcessorURL = resolved
            } else {
                throw ViS2OMaskResourceError.missingModel("MaskProcessor.mlmodelc")
            }

            if let resolved = resolveModelURL(baseURL: baseURL, name: "InstanceRepresentationModule") {
                instanceRepresentationURL = resolved
            } else {
                throw ViS2OMaskResourceError.missingModel("InstanceRepresentationModule.mlmodelc")
            }
        }
    }

    init(
        resourcesAt baseURL: URL,
        configuration config: MLModelConfiguration = .init(),
        reduceMemory: Bool = false
    ) throws {
        let urls = try ResourceURLs(resourcesAt: baseURL)
        let tokenizer = try BPETokenizer(mergesAt: urls.mergesURL, vocabularyAt: urls.vocabURL)
        let textEncoder = TextEncoder(tokenizer: tokenizer, modelAt: urls.textEncoderURL, configuration: config)
        let unet = Unet(modelAt: urls.unetURL, configuration: config)
        let decoder = Decoder(modelAt: urls.decoderURL, configuration: config)
        let encoder = Encoder(modelAt: urls.encoderURL, configuration: config)
        let maskProcessor = MaskProcessor(modelAt: urls.maskProcessorURL, configuration: config)
        let instanceRepresentationModule = InstanceRepresentationModule(
            modelAt: urls.instanceRepresentationURL,
            configuration: config
        )

        self.init(
            textEncoder: textEncoder,
            unet: unet,
            decoder: decoder,
            encoder: encoder,
            maskProcessor: maskProcessor,
            instanceRepresentationModule: instanceRepresentationModule,
            reduceMemory: reduceMemory
        )
    }
}

@available(iOS 16.2, macOS 13.1, *)
private func resolveModelURL(baseURL: URL, name: String) -> URL? {
    let mlmodelc = baseURL.appending(path: "\(name).mlmodelc")
    if FileManager.default.fileExists(atPath: mlmodelc.path) {
        return mlmodelc
    }
    let mlpackage = baseURL.appending(path: "\(name).mlpackage")
    if FileManager.default.fileExists(atPath: mlpackage.path) {
        return mlpackage
    }
    return nil
}

@available(iOS 16.2, macOS 13.1, *)
enum ViS2OMaskResourceError: Error {
    case missingModel(String)
}
