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

const NAV_ITEMS = [
  { to: "/", label: "概览", end: true },
  { to: "/job", label: "职位描述" },
  { to: "/resume", label: "简历" },
  { to: "/cover-letter", label: "求职信" },
  { to: "/applications", label: "申请记录" },
  { to: "/interview", label: "面试准备" },
  { to: "/github", label: "GitHub 证据" },
  { to: "/prompt", label: "Prompt 设置" },
  { to: "/chat", label: "Agent 对话" },
];

export default function App() {
  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-mark">WA</div>
          <div>
            <div className="brand-title">WorkAgent</div>
            <div className="brand-subtitle">个人求职工作台</div>
          </div>
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
              {item.label}
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
          <Route path="/chat" element={<Chat />} />
        </Routes>
      </main>
    </div>
  );
}
