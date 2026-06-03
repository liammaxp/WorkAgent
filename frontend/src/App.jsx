import { useEffect, useMemo, useRef, useState } from "react";
import { NavLink, Route, Routes } from "react-router-dom";
import Dashboard from "./pages/Dashboard.jsx";
import JobDescription from "./pages/JobDescription.jsx";
import Resume from "./pages/Resume.jsx";
import CoverLetter from "./pages/CoverLetter.jsx";
import Applications from "./pages/Applications.jsx";
import InterviewPrep from "./pages/InterviewPrep.jsx";
import GitHubContext from "./pages/GitHubContext.jsx";
import PromptSettings from "./pages/PromptSettings.jsx";
import Chat from "./pages/Chat.jsx";
import { api } from "./api/client.js";
import {
  LANGUAGES,
  LanguageContext,
  getInitialLanguage,
  saveLanguage,
  text,
} from "./i18n.jsx";

const NAV_ITEMS = [
  { to: "/", key: "dashboard", end: true },
  { to: "/resume", key: "resume" },
  { to: "/job", key: "job" },
  { to: "/cover-letter", key: "coverLetter" },
  { to: "/applications", key: "applications" },
  { to: "/interview", key: "interview" },
  { to: "/github", key: "github" },
  { to: "/prompt", key: "prompt" },
  { to: "/chat", key: "chat" },
];

export default function App() {
  const [language, setLanguage] = useState(getInitialLanguage);
  const [chatSession, setChatSession] = useState(() => ({
    sessionId: crypto.randomUUID(),
    createdAt: new Date().toISOString(),
    message: "",
    images: [],
    attachmentError: "",
    history: [],
  }));
  const chatSessionRef = useRef(chatSession);
  const t = text[language];
  const languageValue = useMemo(() => ({ language, setLanguage }), [language]);

  useEffect(() => {
    saveLanguage(language);
    document.documentElement.lang = language === "en" ? "en" : "zh-CN";
  }, [language]);

  useEffect(() => {
    api.openSession().catch(() => {});

    const shutdownOnClose = () => api.sendShutdownBeacon(chatSessionRef.current);
    window.addEventListener("pagehide", shutdownOnClose);
    window.addEventListener("beforeunload", shutdownOnClose);

    return () => {
      window.removeEventListener("pagehide", shutdownOnClose);
      window.removeEventListener("beforeunload", shutdownOnClose);
    };
  }, []);

  useEffect(() => {
    chatSessionRef.current = chatSession;
    const hasContent =
      chatSession.message.trim() ||
      chatSession.images.length ||
      chatSession.history.length;
    if (!hasContent) return undefined;

    const saveTimer = window.setTimeout(() => {
      api.saveChatSession(chatSession).catch(() => {});
    }, 500);
    return () => window.clearTimeout(saveTimer);
  }, [chatSession]);

  return (
    <LanguageContext.Provider value={languageValue}>
      <div className="app-shell">
        <aside className="sidebar">
          <div className="brand">
            <div className="brand-mark">WA</div>
            <div>
              <div className="brand-title">WorkAgent</div>
              <div className="brand-subtitle">{t.appSubtitle}</div>
            </div>
          </div>
          <div className="language-switch" aria-label="Language">
            {Object.entries(LANGUAGES).map(([key, label]) => (
              <button
                key={key}
                type="button"
                className={language === key ? "active" : ""}
                onClick={() => setLanguage(key)}
              >
                {label}
              </button>
            ))}
          </div>
          <nav className="nav-list">
            {NAV_ITEMS.map((item) => (
              <NavLink
                key={item.to}
                to={item.to}
                end={item.end}
                className={({ isActive }) =>
                  `nav-link${isActive ? " active" : ""}`
                }
              >
                {t.nav[item.key]}
              </NavLink>
            ))}
          </nav>
        </aside>
        <main className="main-content">
          <Routes>
            <Route path="/" element={<Dashboard />} />
            <Route path="/job" element={<JobDescription />} />
            <Route path="/resume" element={<Resume />} />
            <Route path="/cover-letter" element={<CoverLetter />} />
            <Route path="/applications" element={<Applications />} />
            <Route path="/interview" element={<InterviewPrep />} />
            <Route path="/github" element={<GitHubContext />} />
            <Route path="/prompt" element={<PromptSettings />} />
            <Route path="/chat" element={<Chat session={chatSession} setSession={setChatSession} />} />
          </Routes>
        </main>
      </div>
    </LanguageContext.Provider>
  );
}
