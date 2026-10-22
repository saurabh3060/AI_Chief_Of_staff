import { Link, useRouterState } from "@tanstack/react-router";
import { LayoutDashboard, ShieldAlert, Building2, Briefcase, CreditCard, ListChecks } from "lucide-react";
import rblLogo from "@/assets/rbl-logo.png";
import { useAppState } from "@/context/AppState";

const NAV: { path: string; label: string; icon: typeof LayoutDashboard }[] = [
  { path: "/",                label: "Executive Dashboard",                          icon: LayoutDashboard },
  { path: "/bbrl",            label: "BBRL",                                         icon: Building2 },
  { path: "/retail-advances", label: "Retail Advances - Excluding Credit Cards",     icon: CreditCard },
  { path: "/credit",          label: "Credit Cards",                                 icon: ShieldAlert },
  { path: "/wholesale",       label: "Wholesale Banking",                            icon: Briefcase },
  { path: "/ledger",          label: "Action & Accountability",                      icon: ListChecks },
];

export function Sidebar() {
  const { location } = useRouterState();
  const currentPath = location.pathname;
  const { clearPremeetingBrief } = useAppState();

  return (
    <aside className="w-[248px] shrink-0 bg-[#0A2540] text-white h-screen sticky top-0 flex flex-col">
      <div className="px-5 pt-5 pb-6 border-b border-white/10">
        <img src={rblLogo} alt="RBL Bank" width={140} height={56} className="h-14 w-auto object-contain" />
        <div className="text-[#94A3B8] text-[11px] uppercase tracking-[0.08em] mt-2">CEO Digital Office</div>
      </div>
      <nav className="flex-1 py-3">
        {NAV.map((n) => {
          const Icon = n.icon;
          // "/" is active only on exact match; all others use startsWith
          const active =
            n.path === "/" ? currentPath === "/" : currentPath.startsWith(n.path);
          return (
            <Link
              key={n.path}
              to={n.path}
              // Direct sidebar navigation always means "show the whole
              // department view" - clear any meeting-scoped selection left
              // over from an "Open Pre Meeting Brief" click so stale
              // filtered state doesn't leak into this navigation.
              onClick={clearPremeetingBrief}
              className={`w-full text-left flex items-start gap-3 pl-5 pr-4 py-3 text-[12.5px] relative transition ${
                active
                  ? "bg-white/[0.06] text-white"
                  : "text-[#94A3B8] hover:bg-white/[0.03] hover:text-white"
              }`}
            >
              {active && <span className="absolute left-0 top-0 bottom-0 w-[3px] bg-[#E8192C]" />}
              <Icon size={16} className="mt-0.5 shrink-0" />
              <span>{n.label}</span>
            </Link>
          );
        })}
      </nav>
      <div className="px-5 py-3 border-t border-white/10 text-[10px] text-[#475569]"></div>
    </aside>
  );
}
