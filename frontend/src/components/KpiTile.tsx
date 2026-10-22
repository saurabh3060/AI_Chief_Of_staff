import { Sparkline } from "./Sparkline";
import { QUARTER_LABELS } from "@/data/mockData";

type Tone = "emerald" | "amber" | "crimson" | "info";
const toneMap: Record<Tone, { bg: string; text: string; line: string }> = {
  emerald: { bg: "#DCFCE7", text: "#166534", line: "#16A34A" },
  amber:   { bg: "#FEF3C7", text: "#92400E", line: "#D97706" },
  crimson: { bg: "#FEE2E2", text: "#991B1B", line: "#DC2626" },
  info:    { bg: "#DBEAFE", text: "#1E40AF", line: "#1E40AF" },
};

interface Props { name: string; value: string; delta: string; tone: Tone; series: number[]; labels?: string[]; }

export function KpiTile({ name, value, delta, tone, series, labels }: Props) {
  const t = toneMap[tone];
  const ls = labels ?? QUARTER_LABELS;
  return (
    <div className="bg-white rounded-[14px] border border-[#E2E8F0] p-4 shadow-[0_1px_3px_rgba(10,37,64,0.06)]">
      <div className="flex items-center justify-between">
        <div className="text-[11px] uppercase tracking-[0.08em] font-semibold text-[#475569]">{name}</div>
        <span className="text-[10px] font-semibold px-2 py-[2px] rounded-[6px]" style={{ background: t.bg, color: t.text }}>●</span>
      </div>
      <div className="serif text-[30px] leading-[1.1] text-[#0A2540] mt-1.5">{value}</div>
      <div className="flex items-end justify-between mt-1.5">
        <span className="text-[11px] font-semibold px-2 py-[3px] rounded-[6px]" style={{ background: t.bg, color: t.text }}>{delta}</span>
        <Sparkline data={series} color={t.line} width={120} height={36} />
      </div>
      <div className="flex justify-between mt-1.5 text-[10px] text-[#94A3B8]">
        {ls.map((q) => <span key={q}>{q.replace("FY", "'")}</span>)}
      </div>
    </div>
  );
}