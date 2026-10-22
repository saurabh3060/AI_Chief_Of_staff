import { useAppState } from "@/context/AppState";

export function Toast() {
  const { toast } = useAppState();
  if (!toast) return null;
  return (
    <div className="fixed bottom-6 right-6 z-[100] slide-in-right">
      <div className="bg-[#0A2540] text-white text-[13px] px-4 py-3 rounded-[10px] shadow-lg max-w-sm">
        {toast}
      </div>
    </div>
  );
}