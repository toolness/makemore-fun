// NumberSlider.tsx
import React from "react"

interface NumberSliderProps {
    min: number
    max: number
    step: number
    value: number
    onChange: (value: number) => void
    label: string
}

/**
 * Vibe-coded by GitHub copilot.
 */
export const NumberSlider: React.FC<NumberSliderProps> = ({
    min,
    max,
    step,
    value,
    onChange,
    label,
}) => (
    <label style={{ display: "flex", alignItems: "center", gap: "0.5em" }}>
        {label && <span>{label}</span>}
        <input
            type="range"
            min={min}
            max={max}
            step={step}
            value={value}
            onChange={(e) => onChange(Number(e.target.value))}
        />
        <span style={{ minWidth: 32, textAlign: "right" }}>{value}</span>
    </label>
)
