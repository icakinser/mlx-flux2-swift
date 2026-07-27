import AppKit
import Flux2Kit
import SwiftUI
import UniformTypeIdentifiers

private final class ImageBox: @unchecked Sendable {
    let image: CGImage
    init(_ image: CGImage) { self.image = image }
}

@MainActor
final class GenerationViewModel: ObservableObject {
    @Published var repository = ProcessInfo.processInfo.environment["FLUX2_REPO"] ?? ""
    @Published var prompt = "a red bicycle leaning against a stone wall, golden hour"
    @Published var status = "Load a model to begin"
    @Published var progress = 0.0
    @Published var output: NSImage?
    @Published var referenceName: String?
    @Published var isLoaded = false
    @Published var isGenerating = false

    private var pipeline: Flux2Pipeline?
    private var cancellation: GenerationCancellation?
    private var reference: ImageBox?

    func loadModel() {
        let path = repository
        guard !path.isEmpty else {
            status = "Choose a local FLUX.2 snapshot"
            return
        }
        status = "Loading model…"
        Task {
            do {
                let loaded = try await Task.detached {
                    try await Flux2Pipeline(
                        configuration: PipelineConfiguration(
                            repoPath: URL(fileURLWithPath: path),
                            residency: .keepResident))
                }.value
                pipeline = loaded
                isLoaded = true
                status = "Ready"
            } catch {
                status = error.localizedDescription
            }
        }
    }

    func importReference(_ url: URL) {
        Task {
            do {
                let loaded = try await Task.detached {
                    guard let image = try loadImages([url]).first else {
                        throw Flux2Error.loadFailed("No image at \(url.path)")
                    }
                    return ImageBox(image)
                }.value
                reference = loaded
                referenceName = url.lastPathComponent
            } catch {
                status = error.localizedDescription
            }
        }
    }

    func generate() {
        guard let pipeline else { return }
        let prompt = prompt
        let reference = reference
        let token = GenerationCancellation()
        cancellation = token
        isGenerating = true
        progress = 0
        status = "Generating…"
        let progressHandler: @Sendable (GenerationProgress) -> Void = { update in
            Task { @MainActor in
                self.progress =
                    Double(update.step + 1) / Double(update.totalSteps)
                self.status = "Step \(update.step + 1) of \(update.totalSteps)"
            }
        }

        Task {
            do {
                let result = try await Task.detached {
                    var options = GenerationOptions(
                        prompt: prompt,
                        width: 512,
                        height: 512,
                        numSteps: 4,
                        guidance: 1.0,
                        seed: UInt64.random(in: 0 ... UInt64.max))
                    options.cancellation = token
                    options.progress = progressHandler
                    return ImageBox(
                        try pipeline.generate(
                            options,
                            inputImages: reference.map { [$0.image] }))
                }.value
                output = NSImage(
                    cgImage: result.image,
                    size: NSSize(width: result.image.width, height: result.image.height))
                status = "Complete"
            } catch Flux2Error.cancelled {
                status = "Cancelled"
            } catch {
                status = error.localizedDescription
            }
            isGenerating = false
            cancellation = nil
        }
    }

    func cancel() {
        cancellation?.cancel()
    }
}

struct ContentView: View {
    @StateObject private var model = GenerationViewModel()
    @State private var importing = false

    var body: some View {
        HSplitView {
            Form {
                TextField("Model snapshot", text: $model.repository)
                Button(model.isLoaded ? "Model loaded" : "Load model") {
                    model.loadModel()
                }
                .disabled(model.isLoaded || model.isGenerating)

                TextField("Prompt", text: $model.prompt, axis: .vertical)
                    .lineLimit(3 ... 6)

                HStack {
                    Button("Import reference") { importing = true }
                    if let name = model.referenceName {
                        Text(name).foregroundStyle(.secondary)
                    }
                }

                ProgressView(value: model.progress)
                Text(model.status).foregroundStyle(.secondary)

                HStack {
                    Button("Generate") { model.generate() }
                        .disabled(!model.isLoaded || model.isGenerating)
                    Button("Cancel") { model.cancel() }
                        .disabled(!model.isGenerating)
                }
            }
            .formStyle(.grouped)
            .frame(minWidth: 360)

            Group {
                if let output = model.output {
                    Image(nsImage: output)
                        .resizable()
                        .scaledToFit()
                } else {
                    ContentUnavailableView(
                        "No output",
                        systemImage: "photo",
                        description: Text("Generated images appear here."))
                }
            }
            .frame(minWidth: 520, minHeight: 560)
            .padding()
        }
        .fileImporter(
            isPresented: $importing,
            allowedContentTypes: [.image],
            allowsMultipleSelection: false
        ) { result in
            if case .success(let urls) = result, let url = urls.first {
                model.importReference(url)
            }
        }
    }
}

@main
struct Flux2KitSwiftUIExampleApp: App {
    var body: some Scene {
        WindowGroup {
            ContentView()
        }
        .defaultSize(width: 960, height: 640)
    }
}
