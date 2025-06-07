interface BaseModelInfo {
    url: string
}

interface BigramModelInfo extends BaseModelInfo {
    type: "bigram"
}

interface TransformerModelInfo extends BaseModelInfo {
    type: "transformer",
    n_embed: number,
    block_size: number,
    num_layers: number,
    num_heads: number,
}

export type ModelInfo = BigramModelInfo | TransformerModelInfo

export interface GenerateMessage {
    type: "generate",
    model: ModelInfo,
    chars: number
}

export interface OutputMessage {
    type: "output",
    text: string
}

export type GptMessage = GenerateMessage | OutputMessage
