// For licensing see accompanying LICENSE.md file.
// Copyright (C) 2025 Apple Inc. and The HuggingFace Team. All Rights Reserved.

import Accelerate
import CoreML
import Foundation

/// Prediction type of the scheduler function
public enum PredictionType {
    /// Predicts the noise of the diffusion process
    case epsilon
    /// Directly predicts the noisy sample
    case sample
    /// v-prediction (see Imagen Video paper)
    case vPrediction
}

/// A scheduler that uses ancestral sampling with Euler method steps
///
/// This implementation matches:
/// [Hugging Face Diffusers EulerAncestralDiscreteScheduler](https://github.com/huggingface/diffusers/blob/main/src/diffusers/schedulers/scheduling_euler_ancestral_discrete.py)
///
/// Uses ancestral sampling with Euler method steps, which adds stochastic noise at each step.
@available(iOS 16.2, macOS 13.1, *)
public final class EulerAncestralDiscreteScheduler: Scheduler {
    public let trainStepCount: Int
    public let inferenceStepCount: Int
    public let betas: [Float]
    public let alphas: [Float]
    public let alphasCumProd: [Float]
    public var timeSteps: [Int]

    public let predictionType: PredictionType
    public let timestepSpacing: TimeStepSpacing

    public private(set) var modelOutputs: [MLShapedArray<Float32>] = []

    // Sigmas for Euler method
    private var sigmas: [Float]

    // Step tracking
    private var stepIndex: Int?
    private var isScaleInputCalled = false

    /// Standard deviation of the initial noise distribution
    public var initNoiseSigma: Float {
        if timestepSpacing == .linspace {
            return sigmas.max() ?? 1.0
        }
        // For leading/trailing spacing
        let maxSigma = sigmas.max() ?? 1.0
        return sqrt(maxSigma * maxSigma + 1)
    }

    /// Create a scheduler that uses ancestral sampling with Euler method steps
    ///
    /// - Parameters:
    ///   - stepCount: Number of inference steps to schedule
    ///   - trainStepCount: Number of training diffusion steps
    ///   - betaSchedule: Method to schedule betas from betaStart to betaEnd
    ///   - betaStart: The starting value of beta for inference
    ///   - betaEnd: The end value for beta for inference
    ///   - predictionType: Type of prediction (epsilon, sample, or v_prediction)
    ///   - timestepSpacing: How to space timesteps (linspace, leading, or trailing)
    ///   - rescaleBetasZeroSnr: Whether to rescale betas to have zero terminal SNR
    /// - Returns: A scheduler ready for its first step
    public init(
        stepCount: Int = 50,
        trainStepCount: Int = 1000,
        betaSchedule: BetaSchedule = .scaledLinear,
        betaStart: Float = 0.00085,
        betaEnd: Float = 0.012,
        predictionType: PredictionType = .epsilon,
        timestepSpacing: TimeStepSpacing = .linspace,
        rescaleBetasZeroSnr: Bool = false
    ) {
        self.trainStepCount = trainStepCount
        self.inferenceStepCount = stepCount
        self.predictionType = predictionType
        self.timestepSpacing = timestepSpacing

        // Initialize betas based on schedule
        var betas: [Float]
        switch betaSchedule {
        case .linear:
            betas = linspace(betaStart, betaEnd, trainStepCount)
        case .scaledLinear:
            betas = linspace(pow(betaStart, 0.5), pow(betaEnd, 0.5), trainStepCount).map { $0 * $0 }
        }

        // Rescale betas for zero terminal SNR if requested
        if rescaleBetasZeroSnr {
            betas = Self.rescaleZeroTerminalSnr(betas: betas)
        }

        self.betas = betas

        // Compute alphas and cumulative product
        self.alphas = betas.map { 1.0 - $0 }
        var alphasCumProd = self.alphas
        for i in 1..<alphasCumProd.count {
            alphasCumProd[i] *= alphasCumProd[i - 1]
        }

        // Handle zero terminal SNR edge case
        if rescaleBetasZeroSnr {
            // Close to 0 without being 0 so first sigma is not inf
            // FP16 smallest positive subnormal works well here
            alphasCumProd[alphasCumProd.count - 1] = pow(2.0, -24.0)
        }

        self.alphasCumProd = alphasCumProd

        // Compute sigmas from alphas_cumprod: sigma = sqrt((1 - alpha_cumprod) / alpha_cumprod)
        var sigmas = alphasCumProd.map { alpha -> Float in
            sqrt((1 - alpha) / alpha)
        }
        // Reverse sigmas and append 0
        sigmas = sigmas.reversed() + [0.0]
        self.sigmas = sigmas

        // Initialize timesteps for full schedule
        let timesteps = linspace(0, Float(trainStepCount - 1), trainStepCount).reversed().map { Int($0) }
        self.timeSteps = timesteps

        self.stepIndex = nil
    }

    /// Rescale betas to have zero terminal SNR
    /// Based on https://arxiv.org/abs/2305.08891 (Algorithm 1)
    private static func rescaleZeroTerminalSnr(betas: [Float]) -> [Float] {
        // Convert betas to alphas_bar_sqrt
        let alphas = betas.map { 1.0 - $0 }
        var alphasCumprod = alphas
        for i in 1..<alphasCumprod.count {
            alphasCumprod[i] *= alphasCumprod[i - 1]
        }
        var alphasBarSqrt = alphasCumprod.map { sqrt($0) }

        // Store old values
        let alphasBarSqrt0 = alphasBarSqrt[0]
        let alphasBarSqrtT = alphasBarSqrt[alphasBarSqrt.count - 1]

        // Shift so the last timestep is zero
        alphasBarSqrt = alphasBarSqrt.map { $0 - alphasBarSqrtT }

        // Scale so the first timestep is back to the old value
        alphasBarSqrt = alphasBarSqrt.map { $0 * alphasBarSqrt0 / (alphasBarSqrt0 - alphasBarSqrtT) }

        // Convert alphas_bar_sqrt back to betas
        let alphasBar = alphasBarSqrt.map { $0 * $0 }
        var newAlphas = [alphasBar[0]]
        for i in 1..<alphasBar.count {
            newAlphas.append(alphasBar[i] / alphasBar[i - 1])
        }
        let newBetas = newAlphas.map { 1.0 - $0 }

        return newBetas
    }

    /// Set the discrete timesteps used for the diffusion chain
    ///
    /// - Parameter numInferenceSteps: Number of diffusion steps for generation
    public func setTimesteps(numInferenceSteps: Int) {
        let timesteps: [Float]

        switch timestepSpacing {
        case .linspace:
            timesteps = linspace(0, Float(trainStepCount - 1), numInferenceSteps).reversed()
        case .leading:
            let stepRatio = trainStepCount / numInferenceSteps
            // Creates integer timesteps by multiplying by ratio
            timesteps = (0..<numInferenceSteps).map { Float($0 * stepRatio) }.reversed()
        case .trailing:
            let stepRatio = Float(trainStepCount) / Float(numInferenceSteps)
            // Creates integer timesteps by multiplying by ratio
            var trailingTimesteps: [Float] = []
            var t = Float(trainStepCount)
            while t > 0 {
                trailingTimesteps.append(round(t) - 1)
                t -= stepRatio
            }
            timesteps = trailingTimesteps
        case .karras:
            // Karras spacing not yet implemented for EulerAncestral, default to linspace
            timesteps = linspace(0, Float(trainStepCount - 1), numInferenceSteps).reversed()
        }

        self.timeSteps = timesteps.map { Int(round($0)) }

        // Recompute sigmas for the new timestep schedule
        let sigmasAll = alphasCumProd.map { alpha -> Float in
            sqrt((1 - alpha) / alpha)
        }

        // Interpolate sigmas for the selected timesteps
        var newSigmas: [Float] = []
        for timestep in timesteps {
            let idx = Int(timestep)
            if idx < sigmasAll.count {
                newSigmas.append(sigmasAll[sigmasAll.count - 1 - idx])
            }
        }
        newSigmas.append(0.0)
        self.sigmas = newSigmas

        self.stepIndex = nil
    }

    /// Find the index of a given timestep in the timestep schedule
    private func indexForTimestep(_ timestep: Int) -> Int {
        let indices = timeSteps.enumerated().filter { $0.element == timestep }.map { $0.offset }
        if indices.isEmpty {
            return 0
        }
        // For the very first step, use the second index if multiple matches exist
        // to avoid skipping a sigma when starting mid-schedule
        let pos = indices.count > 1 ? 1 : 0
        return indices[pos]
    }

    /// Initialize the step index based on the given timestep
    private func initStepIndex(timestep: Int) {
        self.stepIndex = indexForTimestep(timestep)
    }

    /// Scale the model input by (sigma^2 + 1)^0.5 to match the Euler algorithm
    ///
    /// - Parameters:
    ///   - sample: The input sample
    ///   - timestep: The current timestep
    /// - Returns: A scaled input sample
    public func scaleModelInput(
        sample: MLShapedArray<Float32>,
        timestep: Int
    ) -> MLShapedArray<Float32> {
        if stepIndex == nil {
            initStepIndex(timestep: timestep)
        }

        let sigma = sigmas[stepIndex!]
        let scale = 1.0 / sqrt(sigma * sigma + 1)

        isScaleInputCalled = true

        return MLShapedArray(unsafeUninitializedShape: sample.shape) { scalars, _ in
            sample.withUnsafeShapedBufferPointer { buffer, _, _ in
                for i in 0..<scalars.count {
                    scalars.initializeElement(at: i, to: buffer[i] * scale)
                }
            }
        }
    }

    /// Perform one step of the Euler ancestral sampling
    ///
    /// - Parameters:
    ///   - output: The predicted noise or sample from the model
    ///   - timeStep: The current timestep
    ///   - sample: The current sample
    /// - Returns: The denoised sample at the previous timestep
    public func step(
        output: MLShapedArray<Float32>,
        timeStep t: Int,
        sample s: MLShapedArray<Float32>
    ) -> MLShapedArray<Float32> {
        if stepIndex == nil {
            initStepIndex(timestep: t)
        }

        let sigma = sigmas[stepIndex!]

        // Upcast to avoid precision issues
        let sample = s

        // 1. Compute predicted original sample (x_0) from sigma-scaled predicted noise
        let predOriginalSample: MLShapedArray<Float32>
        switch predictionType {
        case .epsilon:
            // x_0 = x_t - sigma * epsilon
            predOriginalSample = MLShapedArray(unsafeUninitializedShape: sample.shape) { scalars, _ in
                sample.withUnsafeShapedBufferPointer { sampleBuf, _, _ in
                    output.withUnsafeShapedBufferPointer { outputBuf, _, _ in
                        for i in 0..<scalars.count {
                            scalars.initializeElement(at: i, to: sampleBuf[i] - sigma * outputBuf[i])
                        }
                    }
                }
            }
        case .vPrediction:
            // x_0 = model_output * (-sigma / sqrt(sigma^2 + 1)) + (sample / (sigma^2 + 1))
            let denomSqrt = sqrt(sigma * sigma + 1)
            predOriginalSample = MLShapedArray(unsafeUninitializedShape: sample.shape) { scalars, _ in
                sample.withUnsafeShapedBufferPointer { sampleBuf, _, _ in
                    output.withUnsafeShapedBufferPointer { outputBuf, _, _ in
                        for i in 0..<scalars.count {
                            let term1 = outputBuf[i] * (-sigma / denomSqrt)
                            let term2 = sampleBuf[i] / (sigma * sigma + 1)
                            scalars.initializeElement(at: i, to: term1 + term2)
                        }
                    }
                }
            }
        case .sample:
            predOriginalSample = output
        }

        // Store the predicted original sample
        modelOutputs.append(predOriginalSample)

        // 2. Compute sigma_from, sigma_to, sigma_up, sigma_down
        let sigmaFrom = sigmas[stepIndex!]
        let sigmaTo = stepIndex! + 1 < sigmas.count ? sigmas[stepIndex! + 1] : 0

        // sigma_up = sqrt(sigma_to^2 * (sigma_from^2 - sigma_to^2) / sigma_from^2)
        let sigmaUp = sqrt(sigmaTo * sigmaTo * (sigmaFrom * sigmaFrom - sigmaTo * sigmaTo) / (sigmaFrom * sigmaFrom))

        // sigma_down = sqrt(sigma_to^2 - sigma_up^2)
        let sigmaDown = sqrt(sigmaTo * sigmaTo - sigmaUp * sigmaUp)

        // 3. Convert to an ODE derivative
        let dt = sigmaDown - sigma

        // derivative = (sample - pred_original_sample) / sigma
        // prev_sample = sample + derivative * dt
        var prevSample = MLShapedArray<Float32>(unsafeUninitializedShape: sample.shape) { scalars, _ in
            sample.withUnsafeShapedBufferPointer { sampleBuf, _, _ in
                predOriginalSample.withUnsafeShapedBufferPointer { predBuf, _, _ in
                    for i in 0..<scalars.count {
                        let derivative = (sampleBuf[i] - predBuf[i]) / sigma
                        scalars.initializeElement(at: i, to: sampleBuf[i] + derivative * dt)
                    }
                }
            }
        }

        // 4. Add ancestral noise
        if sigmaUp > 0 {
            let noise = generateNoise(shape: output.shape)
            prevSample = MLShapedArray(unsafeUninitializedShape: prevSample.shape) { scalars, _ in
                prevSample.withUnsafeShapedBufferPointer { prevBuf, _, _ in
                    noise.withUnsafeShapedBufferPointer { noiseBuf, _, _ in
                        for i in 0..<scalars.count {
                            scalars.initializeElement(at: i, to: prevBuf[i] + noiseBuf[i] * sigmaUp)
                        }
                    }
                }
            }
        }

        // Increment step index
        stepIndex! += 1
        isScaleInputCalled = false

        return prevSample
    }

    /// Generate random noise with the given shape
    private func generateNoise(shape: [Int]) -> MLShapedArray<Float32> {
        let scalarCount = shape.reduce(1, *)
        return MLShapedArray(unsafeUninitializedShape: shape) { scalars, _ in
            for i in 0..<scalarCount {
                // Generate standard normal random number using Box-Muller transform
                let u1 = Float.random(in: 0..<1)
                let u2 = Float.random(in: 0..<1)
                let z = sqrt(-2 * log(u1)) * cos(2 * Float.pi * u2)
                scalars.initializeElement(at: i, to: z)
            }
        }
    }
}
