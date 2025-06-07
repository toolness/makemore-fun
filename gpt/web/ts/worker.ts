import init, { WasmLanguageModel } from "../pkg/web.js"
import { GenerateMessage, GptMessage, ModelInfo } from "./worker-types.js"

async function generate(options: GenerateMessage) {
    const modelInfo = options.model

    await init()

    const safetensors = await fetch(modelInfo.url)
    const safetensorsU8 = new Uint8Array(await safetensors.arrayBuffer())
    const model = createModel(safetensorsU8, modelInfo)
    let text = options.initialContext
    const generator = model.create_generator(BigInt(Date.now()), 1.0, text)

    for (let i = 0; i < options.chars; i++) {
        text += generator.next_token()
        postGptMessage({
            type: "output",
            text,
        })
    }

    postGptMessage({
        type: "done",
    })
}

function createModel(
    safetensors: Uint8Array,
    modelInfo: ModelInfo
): WasmLanguageModel {
    switch (modelInfo.type) {
        case "bigram":
            return WasmLanguageModel.bigram(safetensors)

        case "transformer":
            return WasmLanguageModel.transformer(
                modelInfo.n_embed,
                modelInfo.block_size,
                modelInfo.num_layers,
                modelInfo.num_heads,
                0.0,
                safetensors
            )
    }
}

onmessage = async (e: MessageEvent<GptMessage>) => {
    if (e.data.type === "generate") {
        generate(e.data)
    }
}

function postGptMessage(message: GptMessage) {
    postMessage(message)
}
