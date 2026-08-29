import { useCallback, useEffect, useRef, useState } from "react";
import { useLocation, useNavigate } from "react-router-dom";
import { api } from "../api/client.js";
import { useAgentProgress } from "../agentProgress/AgentProgressContext.jsx";
import { fileChangedSinceAppOpened, readStoredBoolean, writeStoredBoolean } from "../session.js";
import HiringContextRankingReview from "../components/hiringContext/HiringContextRankingReview.jsx";
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

const MAX_STAR_QUESTIONS_PER_RUN = 6;

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

function wait(ms) {
  return new Promise((resolve) => window.setTimeout(resolve, ms));
}

async function waitForBackendAgentTask(taskId, progress, stageId = "generate") {
  const seenMessageIds = new Set();
  while (true) {
    progress.assertActive();
    const status = await api.getAgentTaskStatus(taskId, { signal: progress.signal });
    for (const message of status.messages || []) {
      if (!message.id || seenMessageIds.has(message.id)) continue;
      seenMessageIds.add(message.id);
      if (message.role === "agent") {
        progress.addAgentMessage(message.content);
      } else if (message.role === "user") {
        progress.addSystemMessage(`User: ${message.content}`);
      } else {
        progress.addSystemMessage(message.content);
      }
    }
    const backendStage = (status.stages || []).find((stage) => stage.id === stageId) || status.stages?.[0];
    if (backendStage) {
      progress.updateStage(stageId, {
        status: backendStage.status || (status.status === "running" ? "running" : "pending"),
        detail: backendStage.detail || status.currentStage || "",
      });
    }
    if (status.status === "done" && status.resultAvailable) {
      progress.setStageStatus(stageId, "done", "后台任务完成，正在读取最终结果");
      return api.getAgentTaskResult(taskId, { signal: progress.signal });
    }
    if (status.status === "cancelled") {
      progress.setStageStatus(stageId, "cancelled", "后台任务已取消");
      const error = new Error("Agent task cancelled");
      error.name = "AgentCancelledError";
      throw error;
    }
    if (status.status === "error") {
      progress.setStageStatus(stageId, "error", status.error || "Agent task failed");
      throw new Error(status.error || "Agent task failed");
    }
    await wait(1500);
  }
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
  const [memoryProject, setMemoryProject] = useState("");
  const [routeError, setRouteError] = useState("");
  const [applicationPrompt, setApplicationPrompt] = useState(null);
  const [applicationForm, setApplicationForm] = useState(EMPTY_APPLICATION_FORM);
  const pdfInputRef = useRef(null);
  const loadingRef = useRef(false);
  const { loading, error, success, run } = useAsyncAction();
  const { active: agentActive, runAgentWithProgress } = useAgentProgress();
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
      const data = await runAgentWithProgress({
        title: copy.pdfConverting || "正在转换 PDF 简历",
        initialMessage: `Agent：我会读取 ${file.name}，提取文本和链接后发送给 LaTeX 转换 Agent。`,
        stages: [
          { id: "read", label: `读取 PDF：${file.name}` },
          { id: "convert", label: "发送 PDF 文本和链接生成 LaTeX" },
          { id: "apply", label: "把 LaTeX 写入原始简历编辑器" },
        ],
        modelStageIds: ["convert"],
        action: async (progress) => {
          if (file.type && file.type !== "application/pdf") {
            throw new Error(copy.pdfInvalidType || "Please choose a PDF file.");
          }
          const dataUrl = await progress.runStage("read", `正在用 FileReader 读取 ${file.name} 并转为 base64`, () =>
            new Promise((resolve, reject) => {
              const reader = new FileReader();
              reader.onload = () => resolve(String(reader.result || ""));
              reader.onerror = () => reject(new Error(copy.pdfReadFailed || "Could not read the PDF file."));
              reader.readAsDataURL(file);
            }),
          );
          const base64 = dataUrl.split(",")[1] || "";
          const data = await progress.runStage("convert", "正在发送 filename/data_base64，后端会抽取 PDF 文本、链接并生成完整 LaTeX resume", () =>
            api.convertResumePdfToLatex({
              filename: file.name,
              data_base64: base64,
            }, {
              signal: progress.signal,
              agentProgressMessages: progress.getUserMessages(),
              agentTaskId: progress.agentTaskId,
            }),
          );
          progress.setStageStatus("apply", "running", "正在把返回的 LaTeX content 写入 resume 编辑器状态");
          progress.assertActive();
          progress.setStageStatus("apply", "done");
          progress.addAgentMessage("PDF 简历转换完成。");
          return data;
        },
      });
      setResume(data.content || "");
      return data;
    }, copy.pdfConverted || "PDF converted to LaTeX");
  };

  const updateMemory = () =>
    runResumeAction("memory", async () => {
      const memoryScope = memoryProject.trim() || "全部项目和简历事实";
      const data = await runAgentWithProgress({
        title: copy.memoryUpdating || "正在更新简历记忆",
        initialMessage: `Agent：我会保存 resume.txt，并从当前简历提取“${memoryScope}”相关的长期记忆。`,
        stages: [
          { id: "save", label: "保存 resume.txt" },
          { id: "extract", label: `发送 resume.txt 提取记忆：${memoryScope}` },
          { id: "summarize", label: "合并 Chroma/Profile Memory 新增事实" },
        ],
        modelStageIds: ["extract"],
        action: async (progress) => {
          await progress.runStage("save", `正在保存 ${resume.trim().length} 个字符到 resume.txt`, () => api.saveFile("resume", resume));
          const data = await progress.runStage("extract", `正在发送 resume_source=resume 和 project_name=${memoryProject.trim() || "(空，扫描全部)"}`, () =>
            api.updateMemoryFromResume("resume", {
              project_name: memoryProject.trim(),
            }, {
              signal: progress.signal,
              agentProgressMessages: progress.getUserMessages(),
              agentTaskId: progress.agentTaskId,
            }),
          );
          progress.setStageStatus("summarize", "running", "正在比对返回 memory 与现有记忆，并整理 additions 列表");
          progress.assertActive();
          progress.setStageStatus("summarize", "done");
          progress.addAgentMessage("简历记忆更新完成。");
          return data;
        },
      });
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
      const contextLabel = useGithub ? "job_description.txt + resume.txt + Project Memory + GitHub Evidence" : "job_description.txt + resume.txt + Project Memory";
      const { data, status, jobFingerprint, needsApplicationHint } = await runAgentWithProgress({
        title: copy.generating || "正在生成定制简历",
        initialMessage: `Agent：我会用 ${contextLabel} 生成定制简历，并合并 Projects / Skills / Experience / Summary。`,
        stages: [
          { id: "inspect", label: "读取 job_description.txt 并判断申请记录提示" },
          { id: "star-scan", label: "扫描本地项目库、记忆和 STAR facts" },
          { id: "generate", label: "发送简历上下文并合并简历各 section" },
          { id: "refresh", label: "读取 tailored_resume 输出和 PDF 历史" },
        ],
        modelStageIds: ["generate"],
        action: async (progress) => {
          const jobData = await progress.runStage("inspect", "正在读取 job_description.txt，计算本次 JD fingerprint", () => api.getFile("job_description"));
          const jobFingerprint = fingerprint(jobData.content || "");
          const needsApplicationHint = Boolean(
            jobData.content?.trim() && !hasHandledJobDescription(jobFingerprint),
          );
          const askedQuestionKeys = [];
          const fixedStages = {
            inspect: { id: "inspect", label: "读取 job_description.txt 并判断申请记录提示", status: "done", detail: "已读取 JD 并计算 fingerprint" },
            generate: { id: "generate", label: "发送简历上下文并合并简历各 section", status: "pending" },
            refresh: { id: "refresh", label: "读取 tailored_resume 输出和 PDF 历史", status: "pending" },
          };
          const applyStarCheck = (starCheck) => {
            const starStages = starCheck.stages?.length
              ? starCheck.stages
              : [{ id: "star-scan", label: "扫描本地项目库、记忆和 STAR facts", status: "done", detail: "没有发现需要用户补充的 STAR 缺口" }];
            progress.replaceStages(
              [fixedStages.inspect, ...starStages, fixedStages.generate, fixedStages.refresh],
              starCheck.next_question?.stage_id || "generate",
            );
          };
          let starCheck = await progress.runStage("star-scan", "正在先读取 Project Memory、已保存 STAR facts 和可用代码证据", () =>
            api.checkResumeStarFacts({
              allow_project_selection: allowProjectSelection,
              asked_question_keys: askedQuestionKeys,
            }, {
              signal: progress.signal,
              agentTaskId: progress.agentTaskId,
            }),
          );
          applyStarCheck(starCheck);
          for (const message of starCheck.messages || []) {
            progress.addSystemMessage(message);
          }
          let question = starCheck.next_question;
          let questionCount = 0;
          while (question && questionCount < MAX_STAR_QUESTIONS_PER_RUN) {
            questionCount += 1;
            const answer = await progress.askUserAndWait(
              question.prompt,
              question.stage_id,
              question.stage_detail,
              { contextLabel: question.context_label },
            );
            await api.saveResumeStarFact({
              project_id: question.project_id,
              project_name: question.project_name,
              field_type: question.field_type,
              missing_info_type: question.missing_info_type,
              question_key: question.question_key,
              raw_answer: answer,
            }, {
              signal: progress.signal,
              agentTaskId: progress.agentTaskId,
            });
            askedQuestionKeys.push(question.question_key);
            progress.setStageStatus(question.stage_id, "done", "已保存用户确认信息");
            if (/没有|不知道|no data|not sure|none|没有数据|没有量化|无数据/i.test(answer)) {
              progress.addAgentMessage("明白，我已保存为“无量化数据”。后续 bullet 会使用具体技术动作和保守结果表达，不编造数字。", { contextLabel: question.context_label });
            } else {
              progress.addAgentMessage("收到，我已把这条信息保存到本地项目 STAR facts，继续检查下一个具体缺口。", { contextLabel: question.context_label });
            }
            starCheck = await api.checkResumeStarFacts({
              allow_project_selection: allowProjectSelection,
              asked_question_keys: askedQuestionKeys,
            }, {
              signal: progress.signal,
              agentTaskId: progress.agentTaskId,
            });
            applyStarCheck(starCheck);
            question = starCheck.next_question;
          }
          if (question) {
            progress.addSystemMessage("本轮已达到 STAR 追问上限；剩余缺口会使用本地证据和保守表达继续生成。");
          } else {
            progress.addAgentMessage("STAR 检查完成：我会使用本地记忆、代码证据和刚保存的用户确认信息继续生成。");
          }
          const data = await progress.runStage("generate", `正在创建后台任务：${contextLabel}；后端会选择/更新项目、写 resume bullets、合并 Skills/Experience/Summary`, async () => {
            const task = await api.startTailorResumeTask(
              useGithub,
              allowProjectSelection,
              allowExperienceRemoval,
              needsApplicationHint,
              {
                signal: progress.signal,
                agentProgressMessages: progress.getUserMessages(),
                agentTaskId: progress.agentTaskId,
              },
            );
            progress.addSystemMessage(`后台任务已启动：${task.taskId}`);
            return waitForBackendAgentTask(task.taskId, progress, "generate");
          });
          const status = await progress.runStage("refresh", "正在读取 tailored_resumes 和 tailored_resume_pdfs 最新输出列表", () => api.getStatus());
          progress.addAgentMessage("定制简历已生成。");
          return { data, status, jobFingerprint, needsApplicationHint };
        },
      });
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
          <button type="button" className="btn btn-secondary" onClick={openPdfPicker} disabled={loading || agentActive}>
              {activeAction === "pdfToLatex" ? copy.pdfConverting || "Converting..." : copy.pdfToLatex || "PDF to LaTeX"}
            </button>
          </>
        }
      />

      <section className="card">
        <h2 className="card-title">{copy.memoryTitle}</h2>
        <p className="helper-paragraph">{copy.memoryDescription}</p>
        <div className="field compact-field">
          <label>{copy.memoryProjectLabel || "Project scope"}</label>
          <input
            value={memoryProject}
            onChange={(event) => setMemoryProject(event.target.value)}
            disabled={loading}
            placeholder={copy.memoryProjectPlaceholder || "Leave blank to scan all memory"}
          />
          <p className="helper-text">{copy.memoryProjectHint || "Optional. Specify a project name or ID to update only that project's memory."}</p>
        </div>
        <div className="btn-row">
          <button type="button" className="btn btn-secondary" onClick={updateMemory} disabled={loading || agentActive}>
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
          <button type="button" className="btn btn-primary" onClick={generate} disabled={loading || agentActive}>
            {activeAction === "generate" ? copy.generating : copy.generate}
          </button>
          <HiringContextRankingReview />
        </div>

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
          <>
            <OutputFileSelect
              files={outputFiles}
              value={outputPath}
              disabled={loading}
              inline
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
            <button type="button" className="btn btn-secondary" onClick={exportTailoredPdf} disabled={loading}>
              {activeAction === "exportPdf" ? copy.tailoredPdfExporting || "Exporting..." : copy.exportTailoredPdf || "Export PDF"}
            </button>
            <OutputFileSelect
              files={pdfFiles}
              value={pdfPath}
              disabled={loading}
              showWhenEmpty
              inline
              label={language === "zh" ? "PDF 历史" : "PDF history"}
              placeholder={language === "zh" ? "选择 PDF" : "Choose PDF"}
              onSelect={setPdfPath}
              onOpen={(path) => runResumeAction("openPdf", () => api.launchOutputFile(path))}
              onDelete={(path) => runResumeAction("deletePdf", async () => {
                await api.deleteOutputFile(path);
                setPdfFiles((files) => files.filter((file) => file.path !== path));
                if (pdfPath === path) setPdfPath("");
              }, language === "zh" ? "PDF 文件已删除" : "PDF file deleted")}
            />
          </>
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
