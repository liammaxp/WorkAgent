import { useEffect, useState } from "react";
import { api } from "../api/client.js";
import {
  Alert,
  ConfirmDialog,
  EditorCard,
  LoadingBar,
  PageHeader,
  useAsyncAction,
} from "../components/ui.jsx";
import { text, useLanguage } from "../i18n.jsx";

const APPLICATION_PROMPT_JD_KEY = "workagent-application-prompt-jds";

const EMPTY_APPLICATION_FORM = {
  company: "",
  role: "",
  link: "",
  notes: "",
};

const COMPANY_LABELS = ["company", "company name", "employer", "公司", "公司名称", "企业", "企业名称"];
const ROLE_LABELS = ["role", "position", "job title", "title", "岗位", "岗位名称", "职位", "职位名称", "招聘职位"];
const JD_SECTION_HEADINGS = new Set([
  "about the job",
  "job description",
  "responsibilities",
  "requirements",
  "职位描述",
  "岗位职责",
  "职位要求",
  "任职要求",
]);

function fingerprint(value) {
  let hash = 0;
  for (let index = 0; index < value.length; index += 1) {
    hash = (hash * 31 + value.charCodeAt(index)) | 0;
  }
  return `${value.length}:${hash}`;
}

function hasHandledJobDescription(jobFingerprint) {
  try {
    const fingerprints = JSON.parse(localStorage.getItem(APPLICATION_PROMPT_JD_KEY) || "[]");
    return Array.isArray(fingerprints) && fingerprints.includes(jobFingerprint);
  } catch {
    return false;
  }
}

function rememberHandledJobDescription(jobFingerprint) {
  let fingerprints = [];
  try {
    const storedFingerprints = JSON.parse(localStorage.getItem(APPLICATION_PROMPT_JD_KEY) || "[]");
    if (Array.isArray(storedFingerprints)) fingerprints = storedFingerprints;
  } catch {
    // Replace malformed local data with a clean list.
  }
  localStorage.setItem(
    APPLICATION_PROMPT_JD_KEY,
    JSON.stringify([...new Set([...fingerprints, jobFingerprint])].slice(-100)),
  );
}

function escapeRegularExpression(value) {
  return value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

function findLabeledValue(lines, labels) {
  const labelPattern = labels.map(escapeRegularExpression).join("|");
  const pattern = new RegExp(`^(?:[-*]\\s*)?(?:${labelPattern})\\s*[:：]\\s*(.+)$`, "i");
  for (const line of lines) {
    const match = line.match(pattern);
    if (match?.[1]) return match[1].trim();
  }
  return "";
}

function isLikelyMetadataLine(line) {
  return (
    line.length <= 100 &&
    !line.includes("：") &&
    !line.includes(":") &&
    !/^https?:\/\//i.test(line) &&
    !JD_SECTION_HEADINGS.has(line.toLowerCase())
  );
}

function extractApplicationForm(jobDescription) {
  const lines = jobDescription
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter(Boolean);
  let company = findLabeledValue(lines, COMPANY_LABELS);
  let role = findLabeledValue(lines, ROLE_LABELS);

  for (const line of lines.slice(0, 5)) {
    if (company && role) break;
    const atMatch = line.match(/^(.+?)\s+(?:at|@)\s+(.+)$/i);
    const dividerMatch = line.match(/^(.+?)\s*[|｜]\s*(.+)$/);
    const match = atMatch || dividerMatch;
    if (!match) continue;
    if (!role) role = match[1].trim();
    if (!company) company = match[2].trim();
  }

  if (!company && !role && lines.length >= 2) {
    const [firstLine, secondLine] = lines;
    if (isLikelyMetadataLine(firstLine) && isLikelyMetadataLine(secondLine)) {
      role = firstLine;
      company = secondLine;
    }
  }

  const link = jobDescription.match(/https?:\/\/[^\s<>"')\]]+/i)?.[0] || "";
  return { ...EMPTY_APPLICATION_FORM, company, role, link };
}

export default function Resume() {
  const { language } = useLanguage();
  const copy = text[language].resume;
  const common = text[language].common;
  const [resume, setResume] = useState("");
  const [tailored, setTailored] = useState("");
  const [useGithub, setUseGithub] = useState(false);
  const [allowProjectSelection, setAllowProjectSelection] = useState(true);
  const [outputPath, setOutputPath] = useState("");
  const [memorySummary, setMemorySummary] = useState("");
  const [applicationPrompt, setApplicationPrompt] = useState(null);
  const [applicationForm, setApplicationForm] = useState(EMPTY_APPLICATION_FORM);
  const { loading, error, success, run } = useAsyncAction();

  const loadFiles = () =>
    run(async () => {
      const [base, custom, status] = await Promise.all([
        api.getFile("resume"),
        api.getFile("tailored_resume"),
        api.getStatus(),
      ]);
      setResume(base.content || "");
      setTailored(custom.content || "");
      setOutputPath(status.outputs?.tailored_resumes?.[0]?.path || "");
    });

  useEffect(() => {
    loadFiles();
  }, []);

  const saveResume = () =>
    run(async () => {
      await api.saveFile("resume", resume);
    }, copy.originalSaved);

  const updateMemory = () =>
    run(async () => {
      await api.saveFile("resume", resume);
      const data = await api.updateMemoryFromResume("resume");
      const additions = data.additions || [];
      if (data.updated && additions.length) {
        setMemorySummary(additions.join("\n"));
      } else {
        setMemorySummary(copy.memoryNoNewContent);
      }
      return data;
    }, copy.memoryChecked);

  const saveTailored = () =>
    run(async () => {
      await api.saveFile("tailored_resume", tailored);
      const status = await api.getStatus();
      setOutputPath(status.outputs?.tailored_resumes?.[0]?.path || "");
    }, copy.tailoredSaved);

  const generate = () =>
    run(async () => {
      const jobData = await api.getFile("job_description");
      const data = await api.tailorResume(useGithub, allowProjectSelection);
      setTailored(data.content || "");
      setOutputPath(data.output_path || data.path || "");
      const jobFingerprint = fingerprint(jobData.content || "");
      if (
        jobData.content?.trim() &&
        !hasHandledJobDescription(jobFingerprint)
      ) {
        setApplicationForm(extractApplicationForm(jobData.content));
        setApplicationPrompt({
          jobFingerprint,
          resumeVersion: data.output_path || data.path || "tailored_resume.txt",
        });
      }
      return data;
    }, copy.generated);

  const skipApplicationRecord = () => {
    rememberHandledJobDescription(applicationPrompt.jobFingerprint);
    setApplicationPrompt(null);
  };

  const createApplicationRecord = () =>
    run(async () => {
      await api.createApplication({
        ...applicationForm,
        status: "Interested",
        applied_date: "",
        resume_version: applicationPrompt.resumeVersion,
        cover_letter_version: "cover_letter.txt",
      });
      rememberHandledJobDescription(applicationPrompt.jobFingerprint);
      setApplicationPrompt(null);
      setApplicationForm(EMPTY_APPLICATION_FORM);
    }, copy.applicationHistoryAdded || (language === "zh" ? "申请记录已添加" : "Application record added"));

  const updateApplicationForm = (key, value) => {
    setApplicationForm((previous) => ({ ...previous, [key]: value }));
  };

  return (
    <>
      <PageHeader title={copy.title} description={copy.description} />
      <LoadingBar loading={loading} />
      <Alert type="error" message={error} />
      <Alert type="success" message={success} />

      <EditorCard
        title={copy.original}
        value={resume}
        onChange={setResume}
        onSave={saveResume}
        saving={loading}
        placeholder={copy.originalPlaceholder}
      />

      <section className="card">
        <h2 className="card-title">{copy.memoryTitle}</h2>
        <p className="helper-paragraph">{copy.memoryDescription}</p>
        <div className="btn-row">
          <button type="button" className="btn btn-secondary" onClick={updateMemory} disabled={loading}>
            {loading ? copy.memoryUpdating : copy.memoryUpdate}
          </button>
        </div>
        {memorySummary && (
          <pre className="memory-summary">
            {memorySummary}
          </pre>
        )}
      </section>

      <section className="card">
        <h2 className="card-title">{copy.generateTitle}</h2>
        <label className="inline-check">
          <input type="checkbox" checked={useGithub} onChange={(e) => setUseGithub(e.target.checked)} />
          {copy.useGithub}
        </label>
        <label className="inline-check">
          <input
            type="checkbox"
            checked={allowProjectSelection}
            onChange={(e) => setAllowProjectSelection(e.target.checked)}
          />
          {copy.allowProjectSelection || "允许 Agent 根据职位描述自主删除、更新或补充记忆中的真实项目"}
        </label>
        <div className="btn-row">
          <button type="button" className="btn btn-primary" onClick={generate} disabled={loading}>
            {loading ? copy.generating : copy.generate}
          </button>
        </div>
        {outputPath && (
          <p className="meta-line">
            {common.recentOutput}{outputPath}
          </p>
        )}
      </section>

      <EditorCard
        title={copy.tailored}
        value={tailored}
        onChange={setTailored}
        onSave={saveTailored}
        saving={loading}
        placeholder={copy.tailoredPlaceholder}
      />

      <ConfirmDialog
        open={Boolean(applicationPrompt)}
        title={copy.applicationHistoryTitle || (language === "zh" ? "加入历史申请记录？" : "Add to application history?")}
        confirmLabel={copy.applicationHistoryConfirm || (language === "zh" ? "加入记录" : "Add Record")}
        cancelLabel={copy.applicationHistorySkip || (language === "zh" ? "不加入" : "Do Not Add")}
        loading={loading}
        confirmDisabled={!applicationForm.company.trim() || !applicationForm.role.trim()}
        onCancel={skipApplicationRecord}
        onConfirm={createApplicationRecord}
      >
        <p>
          {copy.applicationHistoryBody ||
            (language === "zh"
              ? "定制简历已生成。是否将这个职位加入历史申请记录？"
              : "The tailored resume has been generated. Add this job to application history?")}
        </p>
        <div className="grid-2">
          <div className="field">
            <label>{text[language].applications.company}</label>
            <input
              value={applicationForm.company}
              onChange={(event) => updateApplicationForm("company", event.target.value)}
            />
          </div>
          <div className="field">
            <label>{text[language].applications.role}</label>
            <input
              value={applicationForm.role}
              onChange={(event) => updateApplicationForm("role", event.target.value)}
            />
          </div>
        </div>
        <div className="field">
          <label>{common.link}</label>
          <input
            value={applicationForm.link}
            onChange={(event) => updateApplicationForm("link", event.target.value)}
          />
        </div>
        <div className="field">
          <label>{text[language].applications.notes}</label>
          <input
            value={applicationForm.notes}
            onChange={(event) => updateApplicationForm("notes", event.target.value)}
          />
        </div>
      </ConfirmDialog>
    </>
  );
}
