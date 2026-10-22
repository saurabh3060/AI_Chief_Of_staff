interface Props { active: string; onChange: (p: string) => void; }
const PHASES = ["Pre-Meeting Briefing", "Live In-Meeting Assistant", "Post-Meeting & Actions"];
const KEYS = ["Pre-Meeting", "Live", "Post-Meeting"];

export function PhaseTabs({ active, onChange }: Props) {
  return (
    <div className="flex gap-6 border-b border-[#E2E8F0]">
      {PHASES.map((p, i) => {
        const k = KEYS[i];
        const isActive = active === k;
        return (
          <button key={k} onClick={() => onChange(k)} className={`pb-3 pt-2 text-[13px] transition-colors ${isActive ? "text-[#0A2540] font-semibold border-b-2 border-[#E8192C]" : "text-[#475569] hover:text-[#0A2540]"}`}>
            {p}
          </button>
        );
      })}
    </div>
  );
}