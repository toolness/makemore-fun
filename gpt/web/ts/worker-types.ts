export type GptMessage = {
    type: "generate"
} | {
    type: "output",
    text: string
}
