interface Props { title: string; outcomes: string[]; openItems: string[]; tags?: { label: string; tone: "emerald" | "amber" | "crimson" }[]; }
const tones = {
  emerald: { bg: "#DCFCE7", text: "#166534" },
  amber: { bg: "#FEF3C7", text: "#92400E" },
  crimson: { bg: "#FEE2E2", text: "#991B1B" },
};
export function MeetingCard({ title, outcomes, openItems, tags = [] }: Props) {
  return (
    <div className="bg-white rounded-[14px] border border-[#E2E8F0] p-4 shadow-[0_1px_3px_rgba(10,37,64,0.06)]">
      <div className="flex items-start justify-between gap-2">
        <div className="text-[13px] font-semibold text-[#0A2540]">{title}</div>
        <div className="flex gap-1 flex-wrap justify-end">
          {tags.map((t, i) => (
            <span key={i} className="text-[10px] font-semibold px-2 py-[2px] rounded-[6px]" style={{ background: tones[t.tone].bg, color: tones[t.tone].text }}>{t.label}</span>
          ))}
        </div>
      </div>
      <div className="mt-3">
        <div className="text-[10px] uppercase tracking-[0.08em] font-semibold text-[#475569] mb-1">Outcomes</div>
        <ul className="space-y-1">
          {outcomes.map((o, i) => <li key={i} className="text-[12px] text-[#0A2540] flex gap-2"><span className="text-[#16A34A] mt-0.5">●</span><span>{o}</span></li>)}
        </ul>
      </div>
      <div className="mt-3">
        <div className="text-[10px] uppercase tracking-[0.08em] font-semibold text-[#475569] mb-1">Open Items</div>
        <ul className="space-y-1">
          {openItems.map((o, i) => <li key={i} className="text-[12px] text-[#0A2540] flex gap-2"><span className="text-[#DC2626] mt-0.5">●</span><span>{o}</span></li>)}
        </ul>
      </div>
    </div>
  );
}