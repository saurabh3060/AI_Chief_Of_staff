/**
 * EditActionModal.tsx
 * ===================
 * Popup modal for editing an action item's description, team, due date,
 * and owner.  Used by both the Launchpad Open Tracker card and the
 * Action & Accountability Ledger.
 *
 * For tracker-sourced items the caller is expected to call the backend
 * PATCH endpoint on save; this component only manages local form state
 * and calls `onSave` with the edited values.
 */

import { useState, useEffect, useRef } from "react";
import { X } from "lucide-react";

const C = { navy: "#0A2540", body: "#334155" };

const TEAM_OPTIONS = [
  "BBRL",
  "Credit Cards",
  "Retail Advances - Excluding Credit Cards",
  "Wholesale Banking",
] as const;

export interface EditActionData {
  text: string;
  owner: string;
  business_unit: string;
  due_date: string;        // YYYY-MM-DD or ""
}

interface EditActionModalProps {
  /** Whether the modal is visible. */
  open: boolean;
  /** Current values to pre-fill the form. */
  initial: EditActionData;
  /** Called when the user clicks "Save changes". */
  onSave: (data: EditActionData) => void;
  /** Called when the user cancels / closes. */
  onClose: () => void;
  /** Whether save is in progress (disables the button). */
  saving?: boolean;
}

export function EditActionModal({ open, initial, onSave, onClose, saving }: EditActionModalProps) {
  const [text, setText] = useState(initial.text);
  const [owner, setOwner] = useState(initial.owner);
  const [bu, setBu] = useState(initial.business_unit);
  const [dueDate, setDueDate] = useState(initial.due_date);
  const backdropRef = useRef<HTMLDivElement>(null);

  // Sync form when modal opens with new initial values.
  useEffect(() => {
    if (open) {
      setText(initial.text);
      setOwner(initial.owner);
      setBu(initial.business_unit);
      setDueDate(initial.due_date);
    }
  }, [open, initial.text, initial.owner, initial.business_unit, initial.due_date]);

  if (!open) return null;

  function handleBackdropClick(e: React.MouseEvent) {
    if (e.target === backdropRef.current) onClose();
  }

  function handleSave() {
    onSave({ text: text.trim(), owner: owner.trim(), business_unit: bu, due_date: dueDate });
  }

  return (
    <div
      ref={backdropRef}
      onClick={handleBackdropClick}
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/30"
    >
      <div
        className="bg-white rounded-[14px] shadow-xl w-full max-w-[480px] mx-4"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Header */}
        <div className="flex items-center justify-between px-6 pt-5 pb-3">
          <h2 className="text-[16px] font-bold uppercase tracking-[0.04em]" style={{ color: C.navy }}>
            Edit Action Item
          </h2>
          <button onClick={onClose} className="text-[#94A3B8] hover:text-[#0A2540] transition" title="Close">
            <X size={18} />
          </button>
        </div>

        {/* Form */}
        <div className="px-6 pb-5 space-y-4">
          {/* Description — read-only (view only, not editable) */}
          <div>
            <label className="block text-[11px] font-semibold uppercase tracking-[0.06em] text-[#94A3B8] mb-1">
              Description
            </label>
            <textarea
              value={text}
              disabled
              rows={3}
              className="w-full text-[13px] px-3 py-2 rounded-[8px] border border-[#E2E8F0] bg-[#F1F5F9] resize-none cursor-not-allowed"
              style={{ color: "#94A3B8" }}
            />
          </div>

          {/* Team — read-only (view only, not editable) */}
          <div>
            <label className="block text-[11px] font-semibold uppercase tracking-[0.06em] text-[#94A3B8] mb-1">
              Team
            </label>
            <select
              value={bu}
              disabled
              className="w-full text-[13px] px-3 py-2 rounded-[8px] border border-[#E2E8F0] bg-[#F1F5F9] cursor-not-allowed"
              style={{ color: "#94A3B8" }}
            >
              {TEAM_OPTIONS.map((t) => (
                <option key={t} value={t}>{t}</option>
              ))}
            </select>
          </div>

          {/* Due date + Owner side by side */}
          <div className="flex gap-4">
            <div className="flex-1">
              <label className="block text-[11px] font-semibold uppercase tracking-[0.06em] text-[#94A3B8] mb-1">
                Due date
              </label>
              <input
                type="date"
                value={dueDate}
                onChange={(e) => setDueDate(e.target.value)}
                className="w-full text-[13px] px-3 py-2 rounded-[8px] border border-[#E2E8F0] focus:outline-none focus:border-[#0A2540]"
                style={{ color: C.body }}
              />
            </div>
            <div className="flex-1">
              <label className="block text-[11px] font-semibold uppercase tracking-[0.06em] text-[#94A3B8] mb-1">
                Owner
              </label>
              <input
                type="text"
                value={owner}
                onChange={(e) => setOwner(e.target.value)}
                className="w-full text-[13px] px-3 py-2 rounded-[8px] border border-[#E2E8F0] focus:outline-none focus:border-[#0A2540]"
                style={{ color: C.body }}
              />
            </div>
          </div>
        </div>

        {/* Footer */}
        <div className="flex items-center justify-end gap-3 px-6 pb-5">
          <button
            onClick={onClose}
            className="text-[12px] font-semibold px-4 py-2 rounded-[8px] text-[#475569] hover:bg-[#F1F5F9] transition"
          >
            Cancel
          </button>
          <button
            onClick={handleSave}
            disabled={saving || !text.trim()}
            className="text-[12px] font-semibold px-5 py-2 rounded-[8px] text-white transition disabled:opacity-50"
            style={{ background: C.navy }}
          >
            {saving ? "Saving..." : "Save changes"}
          </button>
        </div>
      </div>
    </div>
  );
}
