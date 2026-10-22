import React, { useState, useMemo } from "react";
import { ChevronDown, Plus } from "lucide-react";

/* ── Strip trailing due-date suffixes for display only ─────────── */
function stripDueSuffix(text: string): string {
  let out = text.trim();
  // Strip "(by ...)" patterns e.g. "(by 2026-02-10)", "(by 20-Feb-2026)"
  const parenByRe = /\s*\(\s*by\s+[^)]+\)\s*$/i;
  let prev: string;
  do { prev = out; out = out.replace(parenByRe, "").trim(); } while (out !== prev && out.length > 0);
  // Strip "- due ..." and bare ISO/date tails e.g. "- due 30 Jun - 2026-06-30", "- 2026-09-30"
  const tailRe = /\s*[-–—]\s*(?:due\s+.*|\d{4}-\d{1,2}-\d{1,2}|\d{1,2}[-/\s][A-Za-z]{3,9}[-/\s]\d{2,4}|[A-Za-z]{3,9}[-/\s]+\d{1,2},?[-/\s]+\d{2,4}|\d{1,2}[-/]\d{1,2}[-/]\d{4})\s*$/i;
  do { prev = out; out = out.replace(tailRe, "").trim(); } while (out !== prev && out.length > 0);
  return out.length > 0 ? out : text.trim();
}

/* ── Types for Action Items ────────────────────────────────────── */
export interface ActionItem {
  id: string;
  text: string;
  dueDate: string;
  status: "Overdue" | "Due" | "In Progress";
  team?: string;
  category?: string;
  tags?: string[];
  description?: string;
  context?: string;
  priority?: "High" | "Medium" | "Low";
}

/* ── Sample data structure (this would come from props/API) ──────── */
const SAMPLE_ACTION_ITEMS: ActionItem[] = [
  {
    id: "1",
    text: "LAGO ATS reconciliation — mix vs. price decomposition not yet delivered",
    dueDate: "21 Jun 2026",
    status: "Overdue",
    team: "BBRL",
    category: "Retail Advances",
    description: "Joint BBRL / Retail Advances work · Needed before the LAGO policy revision goes to Credit Committee.",
    priority: "High"
  },
  {
    id: "2", 
    text: "Confirm ticket-size band definitions for Retail Deposits <₹3 Cr",
    dueDate: "10 Sep 2026",
    status: "Due",
    team: "BBRL",
    description: "Raised by Deepak Kumar (Risk) · Band definitions must be frozen before the <₹3 Cr book is reported externally.",
    priority: "Medium"
  },
  {
    id: "3",
    text: "Provide LAGO ATS decomposition (mix vs. price)",
    dueDate: "8 Sep 2026", 
    status: "Due",
    team: "Retail Advances",
    category: "BBRL",
    description: "Raised by Deepak Kumar (Risk) · LAGO book ₹4,320 Cr, +3.35% MoM.",
    priority: "Medium"
  }
];

/* ── Color scheme matching the design ──────────────────────────── */
const COLORS = {
  navy: "#0A2540",
  red: "#DC2626", 
  lightRed: "#FEE2E2",
  darkRed: "#991B1B",
  lightGray: "#F8F9FA",
  borderGray: "#E2E8F0",
  mutedText: "#64748B",
  lightText: "#94A3B8"
};

/* ── Status styling ────────────────────────────────────────────── */
const STATUS_STYLES: Record<ActionItem["status"], { bg: string; text: string; badge: string }> = {
  "Overdue": { 
    bg: "#FEE2E2", 
    text: "#991B1B", 
    badge: "Overdue by 4 days" 
  },
  "Due": { 
    bg: "#FEF3C7", 
    text: "#92400E", 
    badge: "Due 10 Sep" 
  },
  "In Progress": { 
    bg: "#DCFCE7", 
    text: "#166534", 
    badge: "Due 8 Sep" 
  }
};

/* ── Tab component ─────────────────────────────────────────────── */
interface TabProps {
  label: string;
  count: number;
  isActive: boolean;
  onClick: () => void;
  variant?: "default" | "danger";
}

function Tab({ label, count, isActive, onClick, variant = "default" }: TabProps) {
  const borderColor = isActive ? (variant === "danger" ? COLORS.red : COLORS.navy) : "transparent";
  
  return (
    <button
      onClick={onClick}
      className="flex items-center gap-2 pb-2 px-1"
      style={{ 
        borderBottom: `2px solid ${borderColor}`,
        color: isActive ? COLORS.navy : COLORS.mutedText
      }}
    >
      <span className="text-xs uppercase tracking-wider font-semibold">
        {label}
      </span>
      <span 
        className="inline-flex items-center justify-center text-xs font-bold text-white rounded-full px-2 py-0.5 min-w-[20px] h-5"
        style={{ 
          backgroundColor: variant === "danger" ? COLORS.red : COLORS.navy 
        }}
      >
        {count}
      </span>
    </button>
  );
}

/* ── Filter dropdown component ─────────────────────────────────── */
interface FilterDropdownProps {
  label: string;
  value: string;
  options: string[];
  onChange: (value: string) => void;
}

function FilterDropdown({ label, value, options, onChange }: FilterDropdownProps) {
  return (
    <div className="relative">
      <select
        value={value}
        onChange={(e) => onChange(e.target.value)}
        className="appearance-none bg-white border border-gray-300 rounded px-3 py-1.5 pr-8 text-sm focus:outline-none focus:border-blue-500"
        style={{ color: COLORS.navy }}
      >
        <option value="All">{label}: All</option>
        {options.filter(opt => opt !== "All").map(option => (
          <option key={option} value={option}>{label}: {option}</option>
        ))}
      </select>
      <ChevronDown 
        size={16} 
        className="absolute right-2 top-1/2 transform -translate-y-1/2 pointer-events-none"
        style={{ color: COLORS.mutedText }}
      />
    </div>
  );
}

/* ── Action item row component ─────────────────────────────────── */
interface ActionItemRowProps {
  item: ActionItem;
  isSelected: boolean;
  onToggle: () => void;
}

function ActionItemRow({ item, isSelected, onToggle }: ActionItemRowProps) {
  const statusStyle = STATUS_STYLES[item.status];
  
  return (
    <div className="flex items-start gap-3 py-3 border-b border-gray-100 last:border-b-0">
      {/* Checkbox */}
      <input
        type="checkbox"
        checked={isSelected}
        onChange={onToggle}
        className="mt-1 w-4 h-4 rounded border-gray-300 focus:ring-2 focus:ring-blue-500"
      />
      
      {/* Status badge */}
      <div 
        className="px-2 py-1 rounded text-xs font-medium whitespace-nowrap mt-0.5"
        style={{ 
          backgroundColor: statusStyle.bg, 
          color: statusStyle.text 
        }}
      >
        {item.status}
      </div>
      
      {/* Team badges */}
      <div className="flex gap-1 mt-0.5">
        {item.team && (
          <span 
            className="px-2 py-1 rounded text-xs font-medium"
            style={{ 
              backgroundColor: COLORS.navy, 
              color: "white" 
            }}
          >
            {item.team}
          </span>
        )}
        {item.category && (
          <span 
            className="px-2 py-1 rounded text-xs font-medium"
            style={{ 
              backgroundColor: COLORS.red, 
              color: "white" 
            }}
          >
            {item.category}
          </span>
        )}
      </div>
      
      {/* Content */}
      <div className="flex-1 min-w-0">
        <div 
          className="text-sm font-medium leading-snug"
          style={{ color: COLORS.navy }}
        >
          {stripDueSuffix(item.text)}
        </div>
        {item.description && (
          <div 
            className="text-xs mt-1 leading-relaxed"
            style={{ color: COLORS.mutedText }}
          >
            {item.description}
          </div>
        )}
      </div>
      
      {/* Due date */}
      <div 
        className="text-xs font-medium whitespace-nowrap mt-0.5"
        style={{ color: COLORS.mutedText }}
      >
        {item.dueDate}
      </div>
    </div>
  );
}

/* ── Main ActionItemsView component ────────────────────────────── */
interface ActionItemsViewProps {
  actionItems?: ActionItem[];
  onClose?: () => void;
}

export function ActionItemsView({ actionItems = SAMPLE_ACTION_ITEMS, onClose }: ActionItemsViewProps) {
  const [activeTab, setActiveTab] = useState<"past" | "open">("open");
  const [statusFilter, setStatusFilter] = useState("All");
  const [teamFilter, setTeamFilter] = useState("All");
  const [selectedItems, setSelectedItems] = useState<Set<string>>(new Set());

  // Get unique values for filters
  const { statuses, teams } = useMemo(() => {
    const statuses = Array.from(new Set(actionItems.map(item => item.status)));
    const teams = Array.from(new Set(actionItems.map(item => item.team).filter(Boolean) as string[]));
    return { statuses, teams };
  }, [actionItems]);

  // Filter items based on current filters
  const filteredItems = useMemo(() => {
    return actionItems.filter(item => {
      if (statusFilter !== "All" && item.status !== statusFilter) return false;
      if (teamFilter !== "All" && item.team !== teamFilter) return false;
      return true;
    });
  }, [actionItems, statusFilter, teamFilter]);

  const handleToggleItem = (itemId: string) => {
    setSelectedItems(prev => {
      const newSet = new Set(prev);
      if (newSet.has(itemId)) {
        newSet.delete(itemId);
      } else {
        newSet.add(itemId);
      }
      return newSet;
    });
  };

  const handleAddItem = () => {
    // This would open a modal or navigate to an add item form
    console.log("Add new item");
  };

  return (
    <div className="bg-white rounded-lg border border-gray-200 overflow-hidden">
      {/* Header with tabs */}
      <div className="px-5 py-4 border-b border-gray-200">
        <div className="flex items-center gap-6 mb-4">
          <Tab
            label="Past meeting summaries"
            count={3}
            isActive={activeTab === "past"}
            onClick={() => setActiveTab("past")}
          />
          <Tab
            label="Open action items"
            count={filteredItems.length}
            isActive={activeTab === "open"}
            onClick={() => setActiveTab("open")}
            variant="danger"
          />
        </div>
        
        {activeTab === "open" && (
          <>
            {/* Description */}
            <div 
              className="text-sm mb-4"
              style={{ color: COLORS.lightText }}
            >
              Extracted from MoM's and manually added — status is derived from the due date.
            </div>
            
            {/* Filters and item count */}
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-4">
                <FilterDropdown
                  label="Status"
                  value={statusFilter}
                  options={["All", ...statuses]}
                  onChange={setStatusFilter}
                />
                <FilterDropdown
                  label="Team"
                  value={teamFilter}
                  options={["All", ...teams]}
                  onChange={setTeamFilter}
                />
              </div>
              <div 
                className="text-sm"
                style={{ color: COLORS.mutedText }}
              >
                {filteredItems.length} items
              </div>
            </div>
          </>
        )}
      </div>
      
      {/* Content */}
      <div className="px-5">
        {activeTab === "past" ? (
          <div className="py-8 text-center" style={{ color: COLORS.lightText }}>
            Past meeting summaries content would go here
          </div>
        ) : (
          <div>
            {filteredItems.map(item => (
              <ActionItemRow
                key={item.id}
                item={item}
                isSelected={selectedItems.has(item.id)}
                onToggle={() => handleToggleItem(item.id)}
              />
            ))}
          </div>
        )}
      </div>
    </div>
  );
}