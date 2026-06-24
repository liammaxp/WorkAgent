import { useCallback, useEffect, useRef, useState } from "react";
import { useLocation, useNavigate } from "react-router-dom";
import { api } from "../api/client.js";
import { fileChangedSinceAppOpened, readStoredBoolean, writeStoredBoolean } from "../session.js";
import {
  Alert,
  ConfirmDialog,
  EditorCard,
  LoadingBar,
  OutputFileSelect,
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

const JD_SAVED_EVENT = "workagent-jd-saved";

export default function Resume() {
  const { language } = useLanguage();
  const location = useLocation();
  const navigate = useNavigate();
  const copy = text[language].resume;
  const common = text[language].common;
  const [resume, setResume] = useState("");
  const [tailored, setTailored] = useState("");
  const [useGithub, setUseGithub] = useState(() => readStoredBoolean("workagent-resume-use-github", false));
  const [allowProjectSelection, setAllowProjectSelection] = useState(() =>
    readStoredBoolean("workagent-resume-allow-project-selection", true),
  );
  const [allowExperienceRemoval, setAllowExperienceRemoval] = useState(() =>
    readStoredBoolean("workagent-resume-allow-experience-removal", false),
  );
  const [outputPath, setOutputPath] = useState("");
  const [outputFiles, setOutputFiles] = useState([]);
  const [pdfPath, setPdfPath] = useState("");
  const [pdfFiles, setPdfFiles] = useState([]);
  const [memorySummary, setMemorySummary] = useState("");
  const [routeError, setRouteError] = useState("");
  const [applicationPrompt, setApplicationPrompt] = useState(null);
  const [applicationForm, setApplicationForm] = useState(EMPTY_APPLICATION_FORM);
  const pdfInputRef = useRef(null);
  const loadingRef = useRef(false);
  const { loading, error, success, run } = useAsyncAction();
  const [activeAction, setActiveAction] = useState("");

  useEffect(() => {
    loadingRef.current = loading;
  }, [loading]);

  const runResumeAction = (actionName, action, successMessage = "") => {
    setActiveAction(actionName);
    return run(action, successMessage).finally(() => setActiveAction(""));
  };

  const loadFiles = useCallback(() =>
    run(async () => {
      const [base, status] = await Promise.all([
        api.getFile("resume"),
        api.getStatus(),
      ]);
      setResume(base.content || "");
      setOutputFiles([
        ...(status.outputs?.tailored_resumes || []),
      ]);
      setPdfFiles(status.outputs?.tailored_resume_pdfs || []);
      if (!fileChangedSinceAppOpened(status, "tailored_resume")) {
        setTailored("");
        setOutputPath("");
        return;
      }
      const custom = await api.getFile("tailored_resume");
      const tailoredContent = custom.content || "";
      setTailored(tailoredContent);
      setOutputPath(tailoredContent.trim() ? status.outputs?.tailored_resumes?.[0]?.path || "" : "");
    }), [run]);

  const clearTailoredResume = () => {
    setTailored("");
    setOutputPath("");
  };

  useEffect(() => {
    const nextRouteError = location.state?.routeError;
    if (!nextRouteError) return;
    setRouteError(String(nextRouteError));
    navigate(location.pathname, { replace: true, state: {} });
  }, [location.pathname, location.state, navigate]);

  useEffect(() => {
    loadFiles();
  }, [location.pathname, loadFiles]);

  useEffect(() => {
    const refreshOnFocus = () => {
      if (!loadingRef.current) loadFiles();
    };
    window.addEventListener("focus", refreshOnFocus);
    return () => window.removeEventListener("focus", refreshOnFocus);
  }, [loadFiles]);

  useEffect(() => {
    const handleJobDescriptionSaved = () => clearTailoredResume();
    window.addEventListener(JD_SAVED_EVENT, handleJobDescriptionSaved);
    return () => window.removeEventListener(JD_SAVED_EVENT, handleJobDescriptionSaved);
  }, []);

  useEffect(() => {
    writeStoredBoolean("workagent-resume-use-github", useGithub);
  }, [useGithub]);

  useEffect(() => {
    writeStoredBoolean("workagent-resume-allow-project-selection", allowProjectSelection);
  }, [allowProjectSelection]);

  useEffect(() => {
    writeStoredBoolean("workagent-resume-allow-experience-removal", allowExperienceRemoval);
  }, [allowExperienceRemoval]);

  const saveResume = () =>
    runResumeAction("saveResume", async () => {
      await api.saveFile("resume", resume);
    }, copy.originalSaved);

  const openPdfPicker = () => {
    pdfInputRef.current?.click();
  };

  const convertPdfToLatex = (event) => {
    const file = event.target.files?.[0];
    event.target.value = "";
    if (!file) return;

    runResumeAction("pdfToLatex", async () => {
      if (file.type && file.type !== "application/pdf") {
        throw new Error(copy.pdfInvalidType || "Please choose a PDF file.");
      }
      const dataUrl = await new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = () => resolve(String(reader.result || ""));
        reader.onerror = () => reject(new Error(copy.pdfReadFailed || "Could not read the PDF file."));
        reader.readAsDataURL(file);
      });
      const base64 = dataUrl.split(",")[1] || "";
      const data = await api.convertResumePdfToLatex({
        filename: file.name,
        data_base64: base64,
      });
      setResume(data.content || "");
      return data;
    }, copy.pdfConverted || "PDF converted to LaTeX");
  };

  const updateMemory = () =>
    runResumeAction("memory", async () => {
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
    runResumeAction("saveTailored", async () => {
      await api.saveFile("tailored_resume", tailored);
      const status = await api.getStatus();
      setOutputPath(status.outputs?.tailored_resumes?.[0]?.path || "");
      setOutputFiles([
        ...(status.outputs?.tailored_resumes || []),
      ]);
      setPdfFiles(status.outputs?.tailored_resume_pdfs || []);
    }, copy.tailoredSaved);

  const exportTailoredPdf = () =>
    runResumeAction("exportPdf", async () => {
      const data = await api.exportTailoredResumePdf(tailored);
      const status = await api.getStatus();
      setPdfPath(data.output_path || data.path || "");
      setOutputFiles([
        ...(status.outputs?.tailored_resumes || []),
      ]);
      setPdfFiles(status.outputs?.tailored_resume_pdfs || []);
      return data;
    }, copy.tailoredPdfExported || "Tailored resume PDF exported");

  const generate = () =>
    runResumeAction("generate", async () => {
      const jobData = await api.getFile("job_description");
      const jobFingerprint = fingerprint(jobData.content || "");
      const needsApplicationHint = Boolean(
        jobData.content?.trim() && !hasHandledJobDescription(jobFingerprint),
      );
      const data = await api.tailorResume(
        useGithub,
        allowProjectSelection,
        allowExperienceRemoval,
        needsApplicationHint,
      );
      const status = await api.getStatus();
      setTailored(data.content || "");
      setOutputPath(data.output_path || data.path || "");
      setOutputFiles([
        ...(status.outputs?.tailored_resumes || []),
      ]);
      setPdfFiles(status.outputs?.tailored_resume_pdfs || []);
      if (needsApplicationHint) {
        setApplicationForm({ ...EMPTY_APPLICATION_FORM, ...(data.application_hint || {}) });
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
      <Alert type="error" message={routeError} />
      <Alert type="error" message={error} />
      <Alert type="success" message={success} />

      <EditorCard
        title={copy.original}
        value={resume}
        onChange={setResume}
        onSave={saveResume}
        saving={activeAction === "saveResume"}
        disabled={loading}
        placeholder={copy.originalPlaceholder}
        extraActions={
          <>
            <input
              ref={pdfInputRef}
              className="hidden-file-input"
              type="file"
              accept="application/pdf,.pdf"
              onChange={convertPdfToLatex}
            />
            <button type="button" className="btn btn-secondary" onClick={openPdfPicker} disabled={loading}>
              {activeAction === "pdfToLatex" ? copy.pdfConverting || "Converting..." : copy.pdfToLatex || "PDF to LaTeX"}
            </button>
          </>
        }
      />

      <section className="card">
        <h2 className="card-title">{copy.memoryTitle}</h2>
        <p className="helper-paragraph">{copy.memoryDescription}</p>
        <div className="btn-row">
          <button type="button" className="btn btn-secondary" onClick={updateMemory} disabled={loading}>
            {activeAction === "memory" ? copy.memoryUpdating : copy.memoryUpdate}
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
        <label className="inline-check">
          <input
            type="checkbox"
            checked={allowExperienceRemoval}
            onChange={(e) => setAllowExperienceRemoval(e.target.checked)}
          />
          {copy.allowExperienceRemoval}
        </label>
        <p className="helper-text">{copy.experienceTailoringHint}</p>
        <div className="btn-row">
          <button type="button" className="btn btn-primary" onClick={generate} disabled={loading}>
            {activeAction === "generate" ? copy.generating : copy.generate}
          </button>
        </div>
        <OutputFileSelect
          files={pdfFiles}
          value={pdfPath}
          disabled={loading}
          showWhenEmpty
          label={language === "zh" ? "PDF 历史输出" : "PDF output history"}
          placeholder={language === "zh" ? "选择 PDF 生成时间" : "Choose a generated PDF"}
          onSelect={setPdfPath}
          onOpen={(path) => runResumeAction("openPdf", () => api.launchOutputFile(path))}
          onDelete={(path) => runResumeAction("deletePdf", async () => {
            await api.deleteOutputFile(path);
            setPdfFiles((files) => files.filter((file) => file.path !== path));
            if (pdfPath === path) setPdfPath("");
          }, language === "zh" ? "PDF 文件已删除" : "PDF file deleted")}
        />
        <OutputFileSelect
          files={outputFiles}
          value={outputPath}
          disabled={loading}
          onSelect={(path) => runResumeAction("loadOutput", async () => {
            const data = await api.getOutputFile(path);
            setTailored(data.content || "");
            setOutputPath(path);
          })}
          onDelete={(path) => runResumeAction("deleteOutput", async () => {
            await api.deleteOutputFile(path);
            setOutputFiles((files) => files.filter((file) => file.path !== path));
            if (outputPath === path) {
              setTailored("");
              setOutputPath("");
            }
          }, language === "zh" ? "输出文件已删除" : "Output file deleted")}
        />
      </section>

      <EditorCard
        title={copy.tailored}
        value={tailored}
        onChange={setTailored}
        onSave={saveTailored}
        saving={activeAction === "saveTailored"}
        disabled={loading}
        placeholder={copy.tailoredPlaceholder}
        extraActions={
          <button type="button" className="btn btn-secondary" onClick={exportTailoredPdf} disabled={loading}>
            {activeAction === "exportPdf" ? copy.tailoredPdfExporting || "Exporting..." : copy.exportTailoredPdf || "Export PDF"}
          </button>
        }
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
