import { useEffect, useMemo, useRef, useState } from "react";
import {
  Activity,
  AlertCircle,
  Bot,
  Brain,
  CalendarDays,
  CheckCircle2,
  ChevronDown,
  Clock3,
  Database,
  FileText,
  Filter,
  Hash,
  Layers3,
  Loader2,
  MessageSquareText,
  PanelRightClose,
  PanelRightOpen,
  RefreshCcw,
  Search,
  Send,
  Trash2,
  UploadCloud,
  X,
  Zap,
} from "lucide-react";
import "./App.css";

const API = "/api";

const STARTER_PROMPTS = [
  "Summarize the most important points across my documents.",
  "What skills or technologies are mentioned?",
  "Find dates, durations, and current responsibilities.",
  "Compare the strongest evidence from the uploaded files.",
];

function formatSize(bytes = 0) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function formatDate(value) {
  if (!value) return "Unknown";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return "Unknown";
  return new Intl.DateTimeFormat(undefined, {
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  }).format(date);
}

function statusCopy(status) {
  if (status === "ready") return { label: "Ready", tone: "success" };
  if (status === "error") return { label: "Failed", tone: "danger" };
  return { label: "Indexing", tone: "warning" };
}

function ScoreBadge({ score }) {
  const pct = Math.max(0, Math.min(100, Math.round((score || 0) * 100)));
  const tone = pct >= 70 ? "success" : pct >= 40 ? "warning" : "danger";
  return <span className={`score-badge ${tone}`}>{pct}%</span>;
}

function Spinner({ size = 16 }) {
  return <Loader2 className="spin" size={size} aria-hidden="true" />;
}

function AnswerText({ text }) {
  const blocks = String(text || "")
    .split(/\n{2,}/)
    .map((block) => block.trim())
    .filter(Boolean);

  if (!blocks.length) return null;

  return (
    <div className="answer-text">
      {blocks.map((block, index) => {
        const lines = block.split("\n").map((line) => line.trim()).filter(Boolean);
        const isList = lines.length > 1 && lines.every((line) => /^[-*]\s+/.test(line));

        if (isList) {
          return (
            <ul key={index}>
              {lines.map((line, lineIndex) => (
                <li key={lineIndex}>{line.replace(/^[-*]\s+/, "")}</li>
              ))}
            </ul>
          );
        }

        return <p key={index}>{lines.join("\n")}</p>;
      })}
    </div>
  );
}

function IconButton({ children, label, className = "", ...props }) {
  return (
    <button className={`icon-button ${className}`} aria-label={label} title={label} {...props}>
      {children}
    </button>
  );
}

function EmptyEvidence() {
  return (
    <div className="empty-panel">
      <FileText size={26} />
      <strong>No evidence selected</strong>
      <span>Sources from the latest answer will appear here.</span>
    </div>
  );
}

export default function App() {
  const [documents, setDocuments] = useState([]);
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [dragOver, setDragOver] = useState(false);
  const [activeSources, setActiveSources] = useState([]);
  const [selectedDocIds, setSelectedDocIds] = useState([]);
  const [docSearch, setDocSearch] = useState("");
  const [topK, setTopK] = useState(5);
  const [memoryEnabled, setMemoryEnabled] = useState(true);
  const [health, setHealth] = useState(null);
  const [documentProfiles, setDocumentProfiles] = useState({});
  const [profileLoading, setProfileLoading] = useState(false);
  const [recentQuestions, setRecentQuestions] = useState([]);
  const [notice, setNotice] = useState(null);
  const [panelOpen, setPanelOpen] = useState(true);

  const chatEndRef = useRef(null);
  const fileInputRef = useRef(null);

  const readyDocs = useMemo(
    () => documents.filter((doc) => doc.status === "ready"),
    [documents]
  );

  const selectedDocuments = useMemo(
    () => documents.filter((doc) => selectedDocIds.includes(doc.id)),
    [documents, selectedDocIds]
  );

  const profileDocumentId = selectedDocIds[0] || documents[0]?.id || null;
  const activeProfile = profileDocumentId ? documentProfiles[profileDocumentId] : null;

  const filteredDocuments = useMemo(() => {
    const query = docSearch.trim().toLowerCase();
    if (!query) return documents;
    return documents.filter((doc) => doc.filename.toLowerCase().includes(query));
  }, [documents, docSearch]);

  const totals = useMemo(
    () => ({
      pages: documents.reduce((sum, doc) => sum + (doc.page_count || 0), 0),
      chunks: documents.reduce((sum, doc) => sum + (doc.chunk_count || 0), 0),
      size: documents.reduce((sum, doc) => sum + (doc.file_size || 0), 0),
    }),
    [documents]
  );

  useEffect(() => {
    refreshWorkspace();
  }, []);

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [messages, loading]);

  useEffect(() => {
    setSelectedDocIds((current) =>
      current.filter((id) => documents.some((doc) => doc.id === id))
    );
  }, [documents]);

  useEffect(() => {
    if (profileDocumentId && !documentProfiles[profileDocumentId]) {
      fetchDocumentProfile(profileDocumentId);
    }
  }, [profileDocumentId, documentProfiles]);

  async function refreshWorkspace() {
    await Promise.all([fetchDocuments(), fetchHealth(), fetchHistory()]);
  }

  async function fetchDocuments() {
    try {
      const res = await fetch(`${API}/documents`);
      if (!res.ok) throw new Error("Could not load documents");
      const data = await res.json();
      setDocuments(data.documents || []);
    } catch (error) {
      setNotice({ tone: "danger", message: error.message });
    }
  }

  async function fetchHealth() {
    try {
      const res = await fetch(`${API}/health`);
      if (!res.ok) throw new Error("Backend unavailable");
      setHealth(await res.json());
    } catch {
      setHealth(null);
    }
  }

  async function fetchHistory() {
    try {
      const res = await fetch(`${API}/chat/history?limit=8`);
      if (!res.ok) return;
      const data = await res.json();
      setRecentQuestions(data.history || []);
    } catch {
      setRecentQuestions([]);
    }
  }

  async function fetchDocumentProfile(docId) {
    setProfileLoading(true);
    try {
      const res = await fetch(`${API}/documents/${docId}/profile`);
      if (!res.ok) return;
      const data = await res.json();
      setDocumentProfiles((current) => ({ ...current, [docId]: data }));
    } finally {
      setProfileLoading(false);
    }
  }

  async function handleUpload(files) {
    const selectedFiles = Array.from(files || []);
    const pdfFiles = selectedFiles.filter((file) => file.name.toLowerCase().endsWith(".pdf"));

    if (!pdfFiles.length) {
      setNotice({ tone: "warning", message: "Select one or more PDF files." });
      return;
    }

    if (pdfFiles.length !== selectedFiles.length) {
      setNotice({ tone: "warning", message: "Only PDF files were added to the upload queue." });
    } else {
      setNotice(null);
    }

    const formData = new FormData();
    pdfFiles.forEach((file) => formData.append("files", file));
    setUploading(true);

    try {
      const res = await fetch(`${API}/documents/upload`, {
        method: "POST",
        body: formData,
      });

      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || "Upload failed");
      }

      await refreshWorkspace();
      setNotice({ tone: "success", message: `${pdfFiles.length} PDF file${pdfFiles.length > 1 ? "s" : ""} indexed.` });
    } catch (error) {
      setNotice({ tone: "danger", message: error.message });
    } finally {
      setUploading(false);
      if (fileInputRef.current) fileInputRef.current.value = "";
    }
  }

  async function handleDelete(docId) {
    const doc = documents.find((item) => item.id === docId);
    const confirmed = window.confirm(`Delete "${doc?.filename || "this document"}" from the index?`);
    if (!confirmed) return;

    try {
      const res = await fetch(`${API}/documents/${docId}`, { method: "DELETE" });
      if (!res.ok) throw new Error("Delete failed");
      setActiveSources((sources) => sources.filter((source) => source.document_id !== docId));
      setDocumentProfiles((current) => {
        const next = { ...current };
        delete next[docId];
        return next;
      });
      await refreshWorkspace();
    } catch (error) {
      setNotice({ tone: "danger", message: error.message });
    }
  }

  async function handleAsk(promptText) {
    const question = (promptText || input).trim();
    if (!question || loading || readyDocs.length === 0) return;

    const scopedDocIds = selectedDocIds.length ? selectedDocIds : null;
    setInput("");
    setNotice(null);
    setLoading(true);
    setActiveSources([]);
    setMessages((prev) => [
      ...prev,
      {
        role: "user",
        content: question,
        scope: scopedDocIds
          ? selectedDocuments.map((doc) => doc.filename).join(", ")
          : "All ready documents",
      },
    ]);

    try {
      const res = await fetch(`${API}/chat/ask`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          question,
          top_k: topK,
          use_history: memoryEnabled,
          history_limit: 4,
          document_ids: scopedDocIds,
        }),
      });

      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || "Request failed");
      }

      const data = await res.json();
      const nextMessage = {
        role: "assistant",
        content: data.answer,
        sources: data.sources || [],
        time: data.processing_time,
      };
      setMessages((prev) => [...prev, nextMessage]);
      setActiveSources(nextMessage.sources);
      fetchHistory();
    } catch (error) {
      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          content: `I could not complete that request. ${error.message}`,
          sources: [],
          error: true,
        },
      ]);
    } finally {
      setLoading(false);
    }
  }

  function handleDrop(event) {
    event.preventDefault();
    setDragOver(false);
    handleUpload(event.dataTransfer.files);
  }

  function handleKeyDown(event) {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      handleAsk();
    }
  }

  function toggleDocument(docId) {
    setSelectedDocIds((current) =>
      current.includes(docId)
        ? current.filter((id) => id !== docId)
        : [...current, docId]
    );
  }

  const askDisabled = loading || !input.trim() || readyDocs.length === 0;
  const healthTone = health ? "success" : "danger";

  return (
    <div className={`app-shell ${panelOpen ? "" : "panel-collapsed"}`}>
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-mark">
            <FileText size={19} />
          </div>
          <div>
            <h1>PyRAG</h1>
            <p>Document intelligence workspace</p>
          </div>
        </div>

        <div className={`system-strip ${healthTone}`}>
          <div>
            <span className="eyebrow">System</span>
            <strong>{health ? "Connected" : "Offline"}</strong>
          </div>
          <Activity size={18} />
        </div>

        <div className="metrics-grid">
          <div>
            <strong>{documents.length}</strong>
            <span>Documents</span>
          </div>
          <div>
            <strong>{totals.chunks}</strong>
            <span>Chunks</span>
          </div>
          <div>
            <strong>{formatSize(totals.size)}</strong>
            <span>Storage</span>
          </div>
        </div>

        <button
          className={`upload-dropzone ${dragOver ? "active" : ""}`}
          type="button"
          onClick={() => fileInputRef.current?.click()}
          onDragOver={(event) => {
            event.preventDefault();
            setDragOver(true);
          }}
          onDragLeave={() => setDragOver(false)}
          onDrop={handleDrop}
        >
          <input
            ref={fileInputRef}
            type="file"
            accept=".pdf"
            multiple
            onChange={(event) => handleUpload(event.target.files)}
          />
          <span className="upload-icon">{uploading ? <Spinner size={20} /> : <UploadCloud size={22} />}</span>
          <strong>{uploading ? "Indexing documents" : "Upload PDFs"}</strong>
          <span>Drag files here or open the file picker</span>
        </button>

        {notice && (
          <div className={`notice ${notice.tone}`}>
            {notice.tone === "danger" ? <AlertCircle size={16} /> : <CheckCircle2 size={16} />}
            <span>{notice.message}</span>
            <IconButton label="Dismiss notice" onClick={() => setNotice(null)}>
              <X size={14} />
            </IconButton>
          </div>
        )}

        <div className="section-heading">
          <span>Indexed Documents</span>
          <IconButton label="Refresh workspace" onClick={refreshWorkspace}>
            <RefreshCcw size={15} />
          </IconButton>
        </div>

        <label className="search-box">
          <Search size={16} />
          <input
            value={docSearch}
            onChange={(event) => setDocSearch(event.target.value)}
            placeholder="Search files"
          />
        </label>

        <div className="document-list">
          {filteredDocuments.length === 0 ? (
            <div className="empty-list">
              <Database size={22} />
              <span>No indexed files</span>
            </div>
          ) : (
            filteredDocuments.map((doc) => {
              const status = statusCopy(doc.status);
              const selected = selectedDocIds.includes(doc.id);
              return (
                <article
                  className={`document-row ${selected ? "selected" : ""}`}
                  key={doc.id}
                >
                  <button type="button" onClick={() => toggleDocument(doc.id)}>
                    <span className={`status-dot ${status.tone}`} />
                    <span className="document-name">{doc.title || doc.filename}</span>
                    <span className={`status-pill ${status.tone}`}>{status.label}</span>
                  </button>
                  {doc.title && doc.title !== doc.filename && (
                    <div className="document-subtitle">{doc.filename}</div>
                  )}
                  <div className="document-meta">
                    <span>{doc.page_count} pages</span>
                    <span>{doc.chunk_count} chunks</span>
                    <span>{formatSize(doc.file_size)}</span>
                  </div>
                  {doc.key_terms?.length > 0 && (
                    <div className="term-strip">
                      {doc.key_terms.slice(0, 4).map((term) => (
                        <span key={term}>{term}</span>
                      ))}
                    </div>
                  )}
                  <div className="document-actions">
                    <span>{formatDate(doc.uploaded_at)}</span>
                    <IconButton label={`Delete ${doc.filename}`} onClick={() => handleDelete(doc.id)}>
                      <Trash2 size={15} />
                    </IconButton>
                  </div>
                </article>
              );
            })
          )}
        </div>
      </aside>

      <main className="workspace">
        <header className="workspace-header">
          <div>
            <span className="eyebrow">Ask</span>
            <h2>Grounded Q&A</h2>
            <p>
              {selectedDocIds.length
                ? `${selectedDocIds.length} document${selectedDocIds.length > 1 ? "s" : ""} selected`
                : `${readyDocs.length} ready document${readyDocs.length !== 1 ? "s" : ""}`}
            </p>
          </div>

          <div className="toolbar">
            <label className="toggle">
              <input
                type="checkbox"
                checked={memoryEnabled}
                onChange={(event) => setMemoryEnabled(event.target.checked)}
              />
              <span />
              Memory
            </label>

            <label className="select-control">
              <Filter size={15} />
              <select value={topK} onChange={(event) => setTopK(Number(event.target.value))}>
                <option value={3}>Top 3</option>
                <option value={5}>Top 5</option>
                <option value={8}>Top 8</option>
              </select>
              <ChevronDown size={14} />
            </label>

            <IconButton
              label={panelOpen ? "Hide evidence panel" : "Show evidence panel"}
              onClick={() => setPanelOpen((open) => !open)}
            >
              {panelOpen ? <PanelRightClose size={18} /> : <PanelRightOpen size={18} />}
            </IconButton>
          </div>
        </header>

        <section className="scope-bar">
          <span>Scope</span>
          {selectedDocuments.length ? (
            selectedDocuments.map((doc) => (
              <button
                className="scope-chip"
                key={doc.id}
                type="button"
                onClick={() => toggleDocument(doc.id)}
              >
                {doc.filename}
                <X size={13} />
              </button>
            ))
          ) : (
            <button className="scope-chip muted" type="button">
              All ready documents
            </button>
          )}
          {selectedDocuments.length > 0 && (
            <button className="clear-scope" type="button" onClick={() => setSelectedDocIds([])}>
              Clear
            </button>
          )}
        </section>

        <section className="chat-scroll" aria-live="polite">
          {messages.length === 0 ? (
            <div className="starter">
              <div className="starter-icon">
                <MessageSquareText size={26} />
              </div>
              <h3>Ready for analysis</h3>
              <div className="prompt-grid">
                {STARTER_PROMPTS.map((prompt) => (
                  <button
                    key={prompt}
                    type="button"
                    onClick={() => handleAsk(prompt)}
                    disabled={readyDocs.length === 0 || loading}
                  >
                    {prompt}
                  </button>
                ))}
              </div>
            </div>
          ) : (
            messages.map((message, index) => (
              <article className={`message ${message.role} ${message.error ? "error" : ""}`} key={`${message.role}-${index}`}>
                <div className="message-avatar">
                  {message.role === "user" ? <Zap size={17} /> : <Bot size={17} />}
                </div>
                <div className="message-body">
                  <div className="message-meta">
                    <strong>{message.role === "user" ? "You" : "PyRAG"}</strong>
                    {message.scope && <span>{message.scope}</span>}
                    {message.time && (
                      <span>
                        <Clock3 size={13} />
                        {message.time}s
                      </span>
                    )}
                  </div>

                  {message.role === "assistant" ? (
                    <AnswerText text={message.content} />
                  ) : (
                    <p>{message.content}</p>
                  )}

                  {message.sources?.length > 0 && (
                    <div className="source-tags">
                      {message.sources.map((source, sourceIndex) => (
                        <button
                          key={`${source.document_id}-${source.page_number}-${sourceIndex}`}
                          type="button"
                          onClick={() => setActiveSources(message.sources)}
                        >
                          <FileText size={13} />
                          {source.document_name}, p.{source.page_number}
                          {source.section_title ? ` - ${source.section_title}` : ""}
                        </button>
                      ))}
                    </div>
                  )}
                </div>
              </article>
            ))
          )}

          {loading && (
            <article className="message assistant loading">
              <div className="message-avatar">
                <Bot size={17} />
              </div>
              <div className="message-body">
                <div className="message-meta">
                  <strong>PyRAG</strong>
                  <span>Retrieving evidence</span>
                </div>
                <div className="thinking-line">
                  <Spinner />
                  <span>Ranking chunks and drafting an answer</span>
                </div>
              </div>
            </article>
          )}
          <div ref={chatEndRef} />
        </section>

        <footer className="composer">
          <textarea
            value={input}
            rows={1}
            onChange={(event) => setInput(event.target.value)}
            onKeyDown={handleKeyDown}
            placeholder={readyDocs.length ? "Ask a grounded question" : "Upload a PDF to begin"}
            disabled={loading || readyDocs.length === 0}
          />
          <button className="send-button" type="button" onClick={() => handleAsk()} disabled={askDisabled}>
            {loading ? <Spinner size={17} /> : <Send size={17} />}
            <span>Ask</span>
          </button>
        </footer>
      </main>

      <aside className="evidence-panel">
        <div className="panel-header">
          <div>
            <span className="eyebrow">Evidence</span>
            <h2>Sources</h2>
          </div>
          <ScoreBadge score={activeSources[0]?.similarity_score || 0} />
        </div>

        <div className="panel-content">
          {activeSources.length === 0 ? (
            <EmptyEvidence />
          ) : (
            activeSources.map((source, index) => (
              <article className="source-card" key={`${source.document_id}-${source.page_number}-${index}`}>
                <div className="source-card-header">
                  <div>
                    <strong>{source.document_name}</strong>
                    <span>
                      Page {source.page_number}
                      {source.section_title ? ` - ${source.section_title}` : ""}
                    </span>
                  </div>
                  <ScoreBadge score={source.similarity_score} />
                </div>
                <p>{source.chunk_text}</p>
              </article>
            ))
          )}
        </div>

        <div className="profile-panel">
          <div className="section-heading compact">
            <span>Document Intelligence</span>
          </div>
          {profileLoading && !activeProfile ? (
            <div className="profile-loading">
              <Spinner />
              <span>Reading profile</span>
            </div>
          ) : activeProfile ? (
            <article className="profile-card">
              <div className="profile-title">
                <Brain size={17} />
                <strong>{activeProfile.title}</strong>
              </div>

              {activeProfile.summary?.length > 0 && (
                <ul className="profile-summary">
                  {activeProfile.summary.slice(0, 3).map((item, index) => (
                    <li key={index}>{item}</li>
                  ))}
                </ul>
              )}

              {activeProfile.key_terms?.length > 0 && (
                <div className="profile-cluster">
                  <span>
                    <Hash size={13} />
                    Key terms
                  </span>
                  <div className="term-strip wrap">
                    {activeProfile.key_terms.slice(0, 8).map((term) => (
                      <span key={term}>{term}</span>
                    ))}
                  </div>
                </div>
              )}

              {activeProfile.sections?.length > 0 && (
                <div className="profile-cluster">
                  <span>
                    <Layers3 size={13} />
                    Sections
                  </span>
                  <div className="section-list">
                    {activeProfile.sections.slice(0, 5).map((section) => (
                      <div key={section.title}>
                        <strong>{section.title}</strong>
                        <small>p.{section.first_page} - {section.chunk_count} chunks</small>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {activeProfile.date_mentions?.length > 0 && (
                <div className="profile-cluster">
                  <span>
                    <CalendarDays size={13} />
                    Dates
                  </span>
                  <div className="term-strip wrap">
                    {activeProfile.date_mentions.slice(0, 6).map((date) => (
                      <span key={date}>{date}</span>
                    ))}
                  </div>
                </div>
              )}
            </article>
          ) : (
            <div className="empty-recent">Select or upload a profiled document</div>
          )}
        </div>

        <div className="recent-panel">
          <div className="section-heading compact">
            <span>Recent Questions</span>
          </div>
          {recentQuestions.length === 0 ? (
            <div className="empty-recent">No recent questions</div>
          ) : (
            recentQuestions.slice(0, 5).map((item) => (
              <button
                className="recent-question"
                key={item.id}
                type="button"
                onClick={() => setInput(item.question)}
              >
                <span>{item.question}</span>
                <small>{formatDate(item.created_at)}</small>
              </button>
            ))
          )}
        </div>
      </aside>
    </div>
  );
}
