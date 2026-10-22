import { useEffect, useRef, useState, useCallback } from "react";
import { Send, Square, RotateCcw, Lightbulb, Mail, FileText } from "lucide-react";
import type { ChatResponse } from "@/data/mockData";
import { clearChatHistory, fetchChatHistory, streamChatQuery } from "@/lib/api";
import { useAppState } from "@/context/AppState";
import type { ChatSource, ChatStreamEvent } from "@/lib/api";

export interface ChatMessage {
  role: "user" | "ai";
  text?: string;
  table?: ChatResponse["table"];
  footer?: string;
  sources?: ChatSource[];
}

interface Props {
  title?: string;
  preloaded?: ChatMessage[];
  chips?: string[];
  placeholder?: string;
  /** Local/fallback responder (sync). Ignored when backend is reachable. */
  getResponse: (q: string) => ChatResponse;
  pulledRef?: React.MutableRefObject<((label: string) => void) | null>;
  compact?: boolean;
  loadHistory?: boolean;
}

/**
 * Simple Markdown parser for bold text (**text** -> <strong>text</strong>)
 */
function parseMarkdown(text: string): React.ReactNode {
  const parts: React.ReactNode[] = [];
  let lastIndex = 0;

  const boldRegex = /\*\*(.+?)\*\*/g;
  let match;

  while ((match = boldRegex.exec(text)) !== null) {
    if (match.index > lastIndex) {
      parts.push(text.substring(lastIndex, match.index));
    }
    parts.push(<strong key={match.index}>{match[1]}</strong>);
    lastIndex = match.index + match[0].length;
  }

  if (lastIndex < text.length) {
    parts.push(text.substring(lastIndex));
  }

  return parts.length > 0 ? parts : text;
}

export function ChatPanel({ title = "Ask your Chief of Staff", preloaded = [], chips = [], placeholder = "Ask anything…", getResponse, pulledRef, compact = false, loadHistory = false }: Props) {
  const { showToast } = useAppState();
  const [messages, setMessages] = useState<ChatMessage[]>(preloaded);
  const [input, setInput] = useState("");
  const [typing, setTyping] = useState(false);
  const [streamingIdx, setStreamingIdx] = useState<number | null>(null);
  const [streamedText, setStreamedText] = useState("");
  const [stepMessage, setStepMessage] = useState<string | null>(null);
  const [isGenerating, setIsGenerating] = useState(false);
  const [chatMode, setChatMode] = useState<"email" | "document">("email");
  const [showModeMenu, setShowModeMenu] = useState(false);
  const endRef = useRef<HTMLDivElement>(null);
  const abortRef = useRef<AbortController | null>(null);
  const modeMenuRef = useRef<HTMLDivElement>(null);

  // One-time load of persisted history on mount (opt-in via `loadHistory`).
  useEffect(() => {
    if (!loadHistory) return;
    let cancelled = false;
    (async () => {
      const entries = await fetchChatHistory(50);
      if (cancelled || entries.length === 0) return;
      const historyMessages: ChatMessage[] = entries.flatMap((e) => [
        { role: "user" as const, text: e.question },
        { role: "ai" as const, text: e.answer },
      ]);
      setMessages((m) => [...historyMessages, ...m]);
    })();
    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Close mode menu when clicking outside
  useEffect(() => {
    function handleClickOutside(e: MouseEvent) {
      if (modeMenuRef.current && !modeMenuRef.current.contains(e.target as Node)) {
        setShowModeMenu(false);
      }
    }
    if (showModeMenu) {
      document.addEventListener("mousedown", handleClickOutside);
      return () => document.removeEventListener("mousedown", handleClickOutside);
    }
  }, [showModeMenu]);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, typing, streamedText, stepMessage]);

  // ── Refresh: clear all messages and reset ──────────────────
  const handleRefresh = useCallback(async () => {
    if (!window.confirm("Clear this chat? The saved conversation history will be deleted.")) return;
    // Clear Redis first. If it fails, keep the chat as-is and tell the user.
    try {
      await clearChatHistory();
    } catch {
      showToast("Couldn't clear chat history. Please try again.");
      return;
    }
    // Abort any in-flight stream
    if (abortRef.current) {
      abortRef.current.abort();
      abortRef.current = null;
    }
    setMessages(preloaded);
    setInput("");
    setTyping(false);
    setStreamingIdx(null);
    setStreamedText("");
    setStepMessage(null);
    setIsGenerating(false);
  }, [preloaded, showToast]);

  // ── Stop: abort the current SSE stream ─────────────────────
  const handleStop = useCallback(() => {
    if (abortRef.current) {
      abortRef.current.abort();
      abortRef.current = null;
    }
    setIsGenerating(false);
    setTyping(false);
    setStepMessage(null);
    // Finalize whatever text we have so far
    if (streamingIdx !== null) {
      setMessages((m) =>
        m.map((mm, idx) =>
          idx === streamingIdx
            ? { ...mm, text: (mm.text || "") + streamedText.slice((mm.text || "").length) }
            : mm,
        ),
      );
      setStreamingIdx(null);
      setStreamedText("");
    }
  }, [streamingIdx, streamedText]);

  function streamResponse(full: string, table?: ChatResponse["table"], footer?: string, sources?: ChatSource[]) {
    setMessages((m) => {
      const next = [...m, { role: "ai" as const, text: "", table, footer, sources }];
      setStreamingIdx(next.length - 1);
      return next;
    });
    let i = 0;
    setStreamedText("");
    if (!full) {
      setStreamingIdx(null);
      return;
    }
    const iv = setInterval(() => {
      i += 1;
      setStreamedText(full.slice(0, i));
      if (i >= full.length) {
        clearInterval(iv);
        setMessages((m) => m.map((mm, idx) => (idx === m.length - 1 ? { ...mm, text: full } : mm)));
        setStreamingIdx(null);
      }
    }, 12);
  }

  async function send(q: string) {
    if (!q.trim() || isGenerating) return;
    const userMsg: ChatMessage = { role: "user", text: q };
    setMessages((m) => [...m, userMsg]);
    setInput("");
    setTyping(true);
    setStepMessage(null);
    setIsGenerating(true);

    // Create abort controller for this request
    const controller = new AbortController();
    abortRef.current = controller;

    try {
      let fullText = "";
      let sources: ChatSource[] = [];
      let shellCreated = false;
      let shellIndex = -1;

      await streamChatQuery(q, (event: ChatStreamEvent) => {
        switch (event.type) {
          case "step":
            setStepMessage(event.message);
            break;

          case "sources":
            sources = event.sources;
            break;

          case "token":
            if (!shellCreated) {
              shellCreated = true;
              setTyping(false);
              setStepMessage(null);
              // Append AI message shell. We compute the index from the
              // messages state snapshot inside the updater (where it's
              // accurate), then store it in a local variable for later use.
              setMessages((m) => {
                shellIndex = m.length;
                const next = [...m, { role: "ai" as const, text: "" }];
                return next;
              });
              // setStreamingIdx must use a functional updater so it reads
              // the shellIndex AFTER the setMessages updater above has run
              // (React processes queued functional updaters in order).
              setStreamingIdx(() => shellIndex);
            }
            fullText += event.content;
            setStreamedText(fullText);
            break;

          case "greeting":
            setTyping(false);
            setStepMessage(null);
            setIsGenerating(false);
            streamResponse(event.content);
            break;

          case "done": {
            const finalText = fullText;
            const finalSources = sources;
            if (shellCreated) {
              // Use functional updater to find the AI message by matching
              // the last AI message in the array (robust regardless of
              // what shellIndex captured).
              setMessages((m) => {
                // Find the last AI message — that's always ours
                const lastAiIdx = m.reduce((acc, mm, i) =>
                  mm.role === "ai" ? i : acc, -1);
                if (lastAiIdx === -1) return m;
                return m.map((mm, i) =>
                  i === lastAiIdx
                    ? { ...mm, text: finalText, sources: finalSources.length > 0 ? finalSources : undefined }
                    : mm,
                );
              });
              setStreamingIdx(null);
            } else if (finalText) {
              setMessages((m) => [
                ...m,
                { role: "ai" as const, text: finalText, sources: finalSources.length > 0 ? finalSources : undefined },
              ]);
              setTyping(false);
            }
            setStepMessage(null);
            setIsGenerating(false);
            break;
          }

          case "error":
            setTyping(false);
            setStepMessage(null);
            setStreamingIdx(null);
            setIsGenerating(false);
            setMessages((m) => [...m, { role: "ai" as const, text: `Error: ${event.content}` }]);
            break;

          case "replace":
            // Backend detected a hallucinated response — replace all
            // streamed text with the corrected content.
            fullText = event.content;
            setStreamedText(fullText);
            break;
        }
      }, controller.signal, chatMode);
    } catch (err) {
      // Check if this was an intentional abort (stop button)
      if (err instanceof DOMException && err.name === "AbortError") {
        // User clicked stop — the handleStop callback already finalized state
        return;
      }
      // SSE streaming failed — show a user-friendly error so they can retry.
      // We intentionally do NOT fall back to the non-streaming endpoint here
      // because firing a second heavy LLM call while the first may still be
      // running on the server competes for CPU and can produce wrong answers.
      setTyping(false);
      setStepMessage(null);
      setIsGenerating(false);
      setStreamingIdx(null);
      setMessages((m) => [
        ...m,
        { role: "ai" as const, text: "Connection interrupted. Please try again." },
      ]);
    } finally {
      abortRef.current = null;
    }
  }

  // Allow parent to push a "Pulled:" system chip into chat
  useEffect(() => {
    if (pulledRef) {
      pulledRef.current = (label: string) => {
        setMessages((m) => [...m, { role: "ai", text: `Pulled into context: ${label}\nNoted - I'll factor this into the live Q&A.` }]);
      };
    }
  }, [pulledRef]);

  return (
    <div className={`bg-white rounded-[10px] border border-[#E2E8F0] flex flex-col ${compact ? "" : "shadow-[0_1px_3px_rgba(10,37,64,0.06)] h-[640px]"}`}>
      {!compact && (
        <div className="px-4 py-3 border-b border-[#EEF2F7] flex items-center justify-between">
          <div className="text-[12px] uppercase tracking-[0.08em] font-semibold text-[#475569]">
            {title}
            <span className={`ml-2 inline-flex items-center gap-1 px-1.5 py-0.5 rounded text-[9px] font-bold tracking-wide ${
              chatMode === "document"
                ? "bg-[#FEF3F2] text-[#E8192C]"
                : "bg-[#F0F9FF] text-[#0369A1]"
            }`}>
              {chatMode === "document" ? <><FileText size={9} /> DOC</> : <><Mail size={9} /> EMAIL</>}
            </span>
          </div>
          <div className="flex items-center gap-2">
            <button
              onClick={handleRefresh}
              title="Clear chat"
              className="p-1 rounded hover:bg-[#F1F5F9] text-[#94A3B8] hover:text-[#475569] transition"
            >
              <RotateCcw size={14} />
            </button>
            <div className="flex items-center gap-1.5">
              <span className="w-1.5 h-1.5 rounded-full bg-[#16A34A] pulse-dot" />
              <span className="text-[9px] text-[#475569]">Chief of Staff online</span>
            </div>
          </div>
        </div>
      )}
      <div className={`overflow-y-auto p-3 space-y-2 ${compact ? "h-[200px]" : "flex-1 p-4 space-y-3"}`}>
        {messages.length === 0 && (
          <div className="text-[12px] text-[#94A3B8] text-center py-6">{placeholder}</div>
        )}
        {messages.map((m, idx) => {
          const displayText = streamingIdx === idx ? streamedText : m.text;
          if (m.role === "user") {
            return (
              <div key={idx} className="flex justify-end fade-in">
                <div className="bg-[#0A2540] text-white text-[13px] px-3.5 py-2 rounded-[10px] max-w-[80%] whitespace-pre-line">{m.text}</div>
              </div>
            );
          }
          return (
            <div key={idx} className="flex justify-start fade-in">
              <div className="bg-white border border-[#E2E8F0] rounded-[10px] p-3 max-w-[88%]">
                <div className="text-[10px] uppercase tracking-[0.1em] font-bold text-[#E8192C] mb-1">Chief of Staff</div>
                {displayText && (
                  <div className="text-[13px] text-[#0A2540] whitespace-pre-line leading-relaxed">
                    {parseMarkdown(displayText)}
                    {streamingIdx === idx && <span className="text-[#94A3B8]">▍</span>}
                  </div>
                )}
                {m.table && (
                  <table className="mt-2 w-full text-[11.5px] border border-[#EEF2F7] rounded">
                    <thead><tr className="bg-[#F1F5F9] text-[10px] uppercase text-[#475569]">{m.table.headers.map((h, i) => <th key={i} className="text-left px-2 py-1.5 font-semibold">{h}</th>)}</tr></thead>
                    <tbody>{m.table.rows.map((r, i) => <tr key={i} className="border-t border-[#EEF2F7]">{r.map((c, j) => <td key={j} className="px-2 py-1.5 text-[#0A2540]">{c}</td>)}</tr>)}</tbody>
                  </table>
                )}
                {/* Source citations — shown after streaming completes */}
                {m.sources && m.sources.length > 0 && streamingIdx !== idx && (
                  <div className="mt-2 pt-2 border-t border-[#EEF2F7]">
                    <div className="text-[10px] uppercase tracking-[0.08em] font-semibold text-[#94A3B8] mb-1.5">Sources</div>
                    <div className="space-y-1.5">
                      {m.sources.map((s, i) => (
                        <div key={i} className="flex items-start gap-2 px-2.5 py-1.5 rounded-[6px] bg-[#F8FAFC] border border-[#E2E8F0]">
                          <div className="flex-1 min-w-0">
                            <div className="text-[11.5px] text-[#0A2540] font-medium truncate">{s.subject}</div>
                            {s.filename && s.filename !== "email" && s.filename !== "meeting" && (
                              <div className="text-[10px] text-[#64748B] truncate">From: {s.filename}</div>
                            )}
                          </div>
                          <div className="text-[10px] text-[#94A3B8] whitespace-nowrap font-medium">{Math.round(s.score * 100)}%</div>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
                {m.footer && streamingIdx !== idx && <div className="text-[12px] text-[#475569] mt-2 italic">{m.footer}</div>}
              </div>
            </div>
          );
        })}
        {/* Loading indicator: step message or bouncing dots */}
        {typing && (
          <div className="flex justify-start fade-in">
            <div className="bg-white border border-[#E2E8F0] rounded-[10px] px-3 py-2.5">
              {stepMessage ? (
                <div className="flex items-center gap-2">
                  <span className="w-1.5 h-1.5 bg-[#E8192C] rounded-full pulse-dot" />
                  <span className="text-[12px] text-[#475569]">{stepMessage}</span>
                </div>
              ) : (
                <div className="flex gap-1">
                  <span className="w-1.5 h-1.5 bg-[#94A3B8] rounded-full pulse-dot" />
                  <span className="w-1.5 h-1.5 bg-[#94A3B8] rounded-full pulse-dot" style={{ animationDelay: "0.2s" }} />
                  <span className="w-1.5 h-1.5 bg-[#94A3B8] rounded-full pulse-dot" style={{ animationDelay: "0.4s" }} />
                </div>
              )}
            </div>
          </div>
        )}
        <div ref={endRef} />
      </div>
      {chips.length > 0 && (
        <div className="px-4 pt-2 flex flex-wrap gap-1.5 border-t border-[#EEF2F7]">
          {chips.map((c) => (
            <button key={c} onClick={() => send(c)} className="text-[11.5px] px-2.5 py-1 rounded-[6px] bg-[#EEF2F7] text-[#0A2540] hover:bg-[#E2E8F0] transition">{c}</button>
          ))}
        </div>
      )}
      <div className="p-3 flex gap-2 border-t border-[#EEF2F7] rounded-b-[10px] items-center" style={{ background: "#0A2540" }}>
        {/* Mode toggle bulb */}
        <div className="relative" ref={modeMenuRef}>
          <button
            onClick={() => setShowModeMenu((v) => !v)}
            title={`Mode: ${chatMode === "email" ? "Email" : "Document"}`}
            className={`p-2 rounded-[8px] transition flex items-center justify-center ${
              showModeMenu
                ? "bg-[#1E3A5F] text-[#FBBF24]"
                : chatMode === "document"
                ? "bg-[#1E3A5F] text-[#FBBF24]"
                : "bg-transparent text-[#94A3B8] hover:text-white"
            }`}
          >
            <Lightbulb size={18} />
          </button>
          {showModeMenu && (
            <div className="absolute bottom-full left-0 mb-2 w-44 bg-white rounded-[10px] shadow-lg border border-[#E2E8F0] overflow-hidden z-50">
              <button
                onClick={() => { setChatMode("email"); setShowModeMenu(false); }}
                className={`w-full flex items-center gap-2.5 px-3 py-2.5 text-[13px] transition ${
                  chatMode === "email"
                    ? "bg-[#FEF3F2] text-[#E8192C] font-semibold"
                    : "text-[#475569] hover:bg-[#F8FAFC]"
                }`}
              >
                <Mail size={15} />
                Email
                {chatMode === "email" && <span className="ml-auto text-[#E8192C]">✓</span>}
              </button>
              <button
                onClick={() => { setChatMode("document"); setShowModeMenu(false); }}
                className={`w-full flex items-center gap-2.5 px-3 py-2.5 text-[13px] transition ${
                  chatMode === "document"
                    ? "bg-[#FEF3F2] text-[#E8192C] font-semibold"
                    : "text-[#475569] hover:bg-[#F8FAFC]"
                }`}
              >
                <FileText size={15} />
                Document
                {chatMode === "document" && <span className="ml-auto text-[#E8192C]">✓</span>}
              </button>
            </div>
          )}
        </div>
        <input
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => { if (e.key === "Enter") send(input); }}
          placeholder={chatMode === "document" ? "Ask about MIS documents…" : "Ask the Chief of Staff…"}
          disabled={isGenerating}
          className="flex-1 text-[13px] px-3 py-2 bg-white border border-[#E2E8F0] rounded-[8px] focus:outline-none focus:border-[#E8192C] placeholder:text-[#94A3B8] disabled:opacity-50"
        />
        {isGenerating ? (
          <button
            onClick={handleStop}
            className="bg-[#475569] hover:bg-[#334155] text-white px-3 rounded-[8px] flex items-center gap-1.5 text-[12px] font-semibold transition"
            title="Stop generating"
          >
            <Square size={14} /> Stop
          </button>
        ) : (
          <button
            onClick={() => send(input)}
            className="bg-[#E8192C] hover:bg-[#C01020] text-white px-3 rounded-[8px] flex items-center gap-1.5 text-[12px] font-semibold transition"
          >
            <Send size={14} /> Send
          </button>
        )}
      </div>
    </div>
  );
}
