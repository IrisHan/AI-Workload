export function fmtParams(n: number): string {
  const abs = Math.abs(n);
  if (abs >= 1e12) return `${(n / 1e12).toFixed(3)}T`;
  if (abs >= 1e9) return `${(n / 1e9).toFixed(3)}B`;
  if (abs >= 1e6) return `${(n / 1e6).toFixed(2)}M`;
  if (abs >= 1e3) return `${(n / 1e3).toFixed(1)}K`;
  return String(n);
}

export function fmtInt(n: number): string {
  return new Intl.NumberFormat("en-US").format(Math.round(n));
}

export function fmtFlop(n: number): string {
  if (n >= 1e12) return `${(n / 1e12).toFixed(2)} TFLOP`;
  if (n >= 1e9) return `${(n / 1e9).toFixed(2)} GFLOP`;
  return `${n.toExponential(3)} FLOP`;
}

export function attnColor(kind: string): string {
  switch (kind) {
    case "MLA":
      return "var(--attn-mla)";
    case "KDA":
      return "var(--attn-kda, var(--attn-mla))";
    case "GQA":
      return "var(--attn-gqa)";
    case "MQA":
      return "var(--attn-mqa)";
    case "MHA":
      return "var(--attn-mha)";
    default:
      return "var(--ink-3)";
  }
}

export function ffnColor(kind: string): string {
  return kind === "MOE" ? "var(--ffn-moe)" : "var(--ffn-dense)";
}
