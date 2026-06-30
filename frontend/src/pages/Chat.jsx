import { useEffect, useRef, useState } from "react";
import { useNavigate } from "react-router-dom";
import { api } from "../api/client.js";
import { useAgentProgress } from "../agentProgress/AgentProgressContext.jsx";
import {
  Alert,
  LoadingBar,
  PageHeader,
  useAsyncAction,
} from "../components/ui.jsx";
import { text, useLanguage } from "../i18n.jsx";

const FLOW_TARGETS = {
  job_description: { route: "/job", zh: "职位描述", en: "Job Description" },
  resume: { route: "/resume", zh: "简历", en: "Resume" },
};

const STAR_DETAILS_QUESTION =
  "为了按 STAR 法则写简历 bullet，请先补充：项目/经历解决了什么问题，你负责哪个模块，使用了哪些关键技术，有没有项目规模或优化前后数据，以及最终结果。没有量化数据也可以直接说“没有数据/不知道”，我不会编造数字。";

function includesAny(value, patterns) {
  return patterns.some((pattern) => pattern.test(value));
}

function detectApplicationMaterialRequest(message) {
  const normalized = message.trim().toLowerCase();
  if (!normalized) return null;

  const applicationIntent = includesAny(normalized, [
    /申请/,
    /投递/,
    /岗位/,
    /职位/,
    /求职材料/,
    /申请材料/,
    /\bapply\b/,
    /\bapplication\b/,
    /\bjob\b/,
    /\bposition\b/,
    /\brole\b/,
  ]);
  const resumeIntent = includesAny(normalized, [
    /简历/,
    /resume/,
    /cv\b/,
  ]);
  const coverLetterIntent = includesAny(normalized, [
    /cover\s*letter/,
    /求职信/,
    /动机信/,
    /申请信/,
  ]);
  const materialIntent = includesAny(normalized, [
    /材料/,
    /准备/,
    /帮我/,
    /生成/,
    /修改/,
    /改/,
    /tailor/,
    /prepare/,
    /generate/,
    /write/,
    /revise/,
    /edit/,
  ]);

  if (!applicationIntent && !resumeIntent && !coverLetterIntent) return null;
  if (!materialIntent && !coverLetterIntent && !resumeIntent) return null;

  if ((applicationIntent && materialIntent) || (resumeIntent && coverLetterIntent)) {
    return { resume: true, coverLetter: true };
  }
  if (coverLetterIntent) return { resume: false, coverLetter: true };
  if (resumeIntent) return { resume: true, coverLetter: false };
  return null;
}

function formatPageList(pages, language) {
  if (pages.length <= 1) return pages[0] || "";
  if (language === "zh") {
    return `${pages.slice(0, -1).join("、")}和${pages[pages.length - 1]}`;
  }
  return `${pages.slice(0, -1).join(", ")} and ${pages[pages.length - 1]}`;
}

export default function Chat({ session, setSession }) {
  const { language } = useLanguage();
  const copy = text[language].chat;
  const { message, images, attachmentError, history, pendingFlow } = session;
  const navigate = useNavigate();
  const imageInputRef = useRef(null);
  const [supportsImages, setSupportsImages] = useState(true);
  const { loading, error, success, run } = useAsyncAction();
  const { active: agentActive, runAgentWithProgress } = useAgentProgress();
  const updateSession = (update) =>
    setSession((current) => ({
      ...current,
      ...(typeof update === "function" ? update(current) : update),
    }));

  const refreshImageSupport = async () => {
    try {
      const data = await api.getStatus();
      setSupportsImages(data.supports_images !== false);
    } catch {
      setSupportsImages(true);
    }
  };

  const flowCopy = {
    needsFreshJob:
      language === "zh"
        ? "当前职位描述不是本次打开应用后保存的内容。请到职位描述页输入或保存新的 JD，然后回到这里继续。"
        : "The saved job description is older than this app session. Please enter or save the current JD on the Job Description page, then return here to continue.",
    needsResume:
      language === "zh"
        ? "生成申请材料需要先填写基础简历。请到简历页补充并保存后，回到这里继续。"
        : "Application materials need a saved base resume. Please fill and save it on the Resume page, then return here to continue.",
    paused:
      language === "zh"
        ? "流程已暂停，补齐材料后可以继续，或取消本次请求。"
        : "The flow is paused. Continue after updating the materials, or cancel this request.",
    continue:
      language === "zh" ? "继续流程" : "Continue Flow",
    cancel:
      language === "zh" ? "取消" : "Cancel",
    generating:
      language === "zh" ? "正在生成材料..." : "Generating materials...",
    saved:
      language === "zh"
        ? "您的材料已在{pages}页生成，可以前往查看。"
        : "Your materials are ready on the {pages} page.",
    savedWithApplication:
      language === "zh"
        ? "您的材料已在{pages}页生成，可以前往查看；申请列表也已自动添加记录。"
        : "Your materials are ready on the {pages} page, and an application record was added automatically.",
    canceled:
      language === "zh" ? "已取消本次材料生成请求。" : "This material generation request was canceled.",
  };

  useEffect(() => {
    refreshImageSupport();
    const onFocus = () => refreshImageSupport();
    window.addEventListener("focus", onFocus);
    return () => window.removeEventListener("focus", onFocus);
  }, []);

  useEffect(() => {
    if (supportsImages || images.length === 0) return;
    updateSession({
      images: [],
      attachmentError: copy.imagesNotSupported,
    });
  }, [supportsImages, images.length, copy.imagesNotSupported]);

  const addImages = async (event) => {
    if (!supportsImages) {
      event.target.value = "";
      updateSession({ attachmentError: copy.imagesNotSupported });
      return;
    }

    const selected = Array.from(event.target.files || []);
    event.target.value = "";
    updateSession({ attachmentError: "" });

    if (images.length + selected.length > 4) {
      updateSession({ attachmentError: copy.tooManyImages });
      return;
    }

    const invalid = selected.find(
      (file) => !["image/jpeg", "image/png", "image/gif", "image/webp"].includes(file.type) || file.size > 10 * 1024 * 1024,
    );
    if (invalid) {
      updateSession({ attachmentError: copy.invalidImage });
      return;
    }

    const added = await Promise.all(
      selected.map(
        (file) =>
          new Promise((resolve, reject) => {
            const reader = new FileReader();
            reader.onload = () => resolve({ name: file.name, mime_type: file.type, data_url: reader.result });
            reader.onerror = () => reject(new Error(copy.imageReadFailed));
            reader.readAsDataURL(file);
          }),
      ),
    ).catch((readError) => {
      updateSession({ attachmentError: readError.message });
      return [];
    });
    updateSession((current) => ({ images: [...current.images, ...added] }));
  };

  const addHistory = (entry) => {
    updateSession((current) => ({ history: [...current.history, entry] }));
  };

  const clearPendingFlow = () => {
    updateSession({ pendingFlow: null });
  };

  const missingFlowPrerequisite = async () => {
    const status = await api.getStatus();
    if (!status.files?.job_description) {
      return {
        key: "job_description",
        route: FLOW_TARGETS.job_description.route,
        message: flowCopy.needsFreshJob,
      };
    }

    const openedAt = Date.parse(session.createdAt || "");
    const jobMtime = status.file_metadata?.job_description?.mtime_ms;
    if (Number.isFinite(openedAt) && jobMtime && jobMtime < openedAt) {
      return {
        key: "job_description",
        route: FLOW_TARGETS.job_description.route,
        message: flowCopy.needsFreshJob,
      };
    }

    if (!status.files?.resume) {
      return {
        key: "resume",
        route: FLOW_TARGETS.resume.route,
        message: flowCopy.needsResume,
      };
    }

    return null;
  };

  const pauseFlow = (flow, reason) => {
    updateSession({
      pendingFlow: { ...flow, pausedAt: new Date().toISOString(), reason },
      message: "",
      images: [],
      attachmentError: "",
    });
    addHistory({ role: "agent", text: `${reason.message}\n\n${flowCopy.paused}` });
    navigate(reason.route, {
      state: {
        routeError: reason.message,
      },
    });
  };

  const runApplicationMaterialFlow = async (flow, progress = null) => {
    const missing = progress
      ? await progress.runStage("check", "检查 /status 中 job_description、resume 是否已保存且 JD 是本次会话内容", () => missingFlowPrerequisite())
      : await missingFlowPrerequisite();
    if (missing) {
      pauseFlow(flow, missing);
      return null;
    }

    const pages = [];
    let resumeData = null;
    let coverLetterData = null;
    let applicationHint = null;
    if (flow.outputs.resume && progress) {
      await progress.runStage("star", "检查简历 bullet 是否具备 STAR 信息", async () => {
        const answer = await progress.askUserAndWait(
          STAR_DETAILS_QUESTION,
          "star",
          "等待用户补充 STAR 信息；回复没有数据也可以继续",
        );
        if (/没有|不知道|no data|not sure|none|没有数据/i.test(answer)) {
          progress.addAgentMessage("明白，我不会编造数字；会用真实技术动作和保守结果表达来写 bullet。");
        } else {
          progress.addAgentMessage("收到，我会把这些 STAR 信息带入定制简历生成。");
        }
      });
    }
    if (flow.outputs.resume) {
      resumeData = progress
        ? await progress.runStage("resume", "发送 job_description.txt + resume.txt + Project Memory + GitHub context，生成 tailored_resume 并返回 application_hint", () =>
          api.tailorResume(true, true, false, true, {
            signal: progress.signal,
            agentProgressMessages: progress.getUserMessages(),
            agentTaskId: progress.agentTaskId,
          }),
        )
        : await api.tailorResume(true, true, false, true);
      applicationHint = resumeData?.application_hint || applicationHint;
      pages.push(language === "zh" ? "简历" : "Resume");
    }
    if (flow.outputs.coverLetter) {
      coverLetterData = progress
        ? await progress.runStage("coverLetter", `发送 ${flow.outputs.resume ? "刚生成的 tailored_resume" : "现有 tailored_resume/resume"} + job_description，生成 cover_letter`, () =>
          api.generateCoverLetter({
            use_tailored_resume: Boolean(flow.outputs.resume || resumeData),
            use_github_context: false,
            style: "concise",
            include_application_hint: !applicationHint,
          }, {
            signal: progress.signal,
            agentProgressMessages: progress.getUserMessages(),
            agentTaskId: progress.agentTaskId,
          }),
        )
        : await api.generateCoverLetter({
          use_tailored_resume: Boolean(flow.outputs.resume || resumeData),
          use_github_context: false,
          style: "concise",
          include_application_hint: !applicationHint,
        });
      applicationHint = coverLetterData?.application_hint || applicationHint;
      pages.push(language === "zh" ? "求职信" : "Cover Letter");
    }

    const createRecord = () => api.createApplication({
      company: applicationHint?.company?.trim() || (language === "zh" ? "未知公司" : "Unknown company"),
      role: applicationHint?.role?.trim() || (language === "zh" ? "未知岗位" : "Unknown role"),
      link: applicationHint?.link?.trim() || "",
      status: "Interested",
      applied_date: "",
      resume_version: resumeData?.output_path || resumeData?.path || "",
      cover_letter_version: coverLetterData?.output_path || coverLetterData?.path || "",
      notes: language === "zh" ? "由 Agent Chat 自动添加。" : "Added automatically from Agent Chat.",
    });

    if (progress) {
      await progress.runStage("record", `写入申请记录：${applicationHint?.company || "Unknown company"} / ${applicationHint?.role || "Unknown role"}`, createRecord);
      progress.addAgentMessage("Application materials are ready.");
    } else {
      await createRecord();
    }

    const pageText = formatPageList(pages, language);
    addHistory({
      role: "agent",
      text: flowCopy.savedWithApplication.replace("{pages}", pageText),
    });
    clearPendingFlow();
    return { saved: true };
  };

  const send = () =>
    run(async () => {
      const trimmed = message.trim();
      if (!trimmed && images.length === 0) return null;
      if (!supportsImages && images.length > 0) {
        updateSession({ images: [], attachmentError: copy.imagesNotSupported });
        return null;
      }

      const attachedImages = supportsImages ? images : [];
      const userEntry = { role: "user", text: trimmed || copy.imageOnlyMessage, images: attachedImages };
      updateSession((current) => ({
        history: [...current.history, userEntry],
        message: "",
        images: [],
        attachmentError: "",
      }));

      const requestedOutputs = detectApplicationMaterialRequest(trimmed);
      if (requestedOutputs && attachedImages.length === 0) {
        const stages = [
          { id: "check", label: "检查 JD 和基础简历是否可用" },
          ...(requestedOutputs.resume ? [{ id: "star", label: "补全 STAR bullet 信息" }] : []),
          ...(requestedOutputs.resume ? [{ id: "resume", label: "生成 tailored_resume.txt" }] : []),
          ...(requestedOutputs.coverLetter ? [{ id: "coverLetter", label: "生成 cover_letter.txt" }] : []),
          { id: "record", label: "写入 applications 历史记录" },
        ];
        return runAgentWithProgress({
          title: flowCopy.generating,
          initialMessage: `User: ${trimmed}`,
          stages,
          modelStageIds: [
            ...(requestedOutputs.resume ? ["resume"] : []),
            ...(requestedOutputs.coverLetter ? ["coverLetter"] : []),
          ],
          action: (progress) => runApplicationMaterialFlow({
            message: trimmed,
            outputs: requestedOutputs,
            userEntry,
          }, progress),
        });
      }

      const data = await runAgentWithProgress({
        title: language === "zh" ? "正在调用 Agent" : "Calling Agent",
        initialMessage: `User: ${trimmed || copy.imageOnlyMessage}`,
        stages: [
          { id: "send", label: `准备用户消息${attachedImages.length ? `和 ${attachedImages.length} 张图片` : ""}` },
          { id: "answer", label: "发送消息到 Agent 并生成回复" },
        ],
        modelStageIds: ["answer"],
        action: async (progress) => {
          progress.setStageStatus("send", "done");
          const data = await progress.runStage("answer", `正在发送 ${trimmed.length} 个字符${attachedImages.length ? ` + ${attachedImages.length} 张图片` : ""} 给 Agent`, () =>
            api.askAgent(trimmed, attachedImages, {
              signal: progress.signal,
              agentProgressMessages: progress.getUserMessages(),
              agentTaskId: progress.agentTaskId,
            }),
          );
          progress.addAgentMessage(data.answer || "");
          return data;
        },
      });
      const agentEntry = { role: "agent", text: data.answer || "" };
      updateSession((current) => ({ history: [...current.history, agentEntry] }));
      return data;
    });

  const continueFlow = () =>
    run(async () => {
      if (!pendingFlow) return null;
      const stages = [
        { id: "check", label: "重新检查 JD 和基础简历" },
        ...(pendingFlow.outputs.resume ? [{ id: "star", label: "补全 STAR bullet 信息" }] : []),
        ...(pendingFlow.outputs.resume ? [{ id: "resume", label: "生成 tailored_resume.txt" }] : []),
        ...(pendingFlow.outputs.coverLetter ? [{ id: "coverLetter", label: "生成 cover_letter.txt" }] : []),
        { id: "record", label: "写入 applications 历史记录" },
      ];
      return runAgentWithProgress({
        title: flowCopy.generating,
        initialMessage: pendingFlow.message ? `User: ${pendingFlow.message}` : flowCopy.generating,
        stages,
        modelStageIds: [
          ...(pendingFlow.outputs.resume ? ["resume"] : []),
          ...(pendingFlow.outputs.coverLetter ? ["coverLetter"] : []),
        ],
        action: (progress) => runApplicationMaterialFlow(pendingFlow, progress),
      });
    });

  const cancelFlow = () => {
    addHistory({ role: "agent", text: flowCopy.canceled });
    clearPendingFlow();
  };

  return (
    <div className="chat-page">
      <div className="chat-header">
        <PageHeader title={copy.title} description={copy.description} />
        <LoadingBar loading={loading} />
        <Alert type="error" message={error} />
        <Alert type="success" message={success} />
      </div>

      <section className="card chat-history-card">
        {history.length === 0 ? (
          <p className="empty-state">{copy.empty}</p>
        ) : (
          <div className="chat-message-list">
            {history.map((entry, index) => (
              <div
                key={index}
                className={`chat-message ${entry.role === "user" ? "user" : "agent"}`}
              >
                <strong>
                  {entry.role === "user" ? copy.you : "Agent"}
                </strong>
                {entry.images?.length > 0 && (
                  <div className="chat-image-grid">
                    {entry.images.map((image, imageIndex) => (
                      <img key={`${image.name}-${imageIndex}`} src={image.data_url} alt={image.name || copy.attachedImage} />
                    ))}
                  </div>
                )}
                {entry.text}
              </div>
            ))}
          </div>
        )}
      </section>

      <section className="card chat-composer">
        {pendingFlow ? (
          <>
            <p className="chat-flow-paused">{flowCopy.paused}</p>
            <div className="btn-row">
              <button type="button" className="btn btn-secondary" onClick={cancelFlow} disabled={loading}>
                {flowCopy.cancel}
              </button>
              <button type="button" className="btn btn-primary" onClick={continueFlow} disabled={loading || agentActive}>
                {loading ? flowCopy.generating : flowCopy.continue}
              </button>
            </div>
          </>
        ) : (
          <>
            <div className="field">
              <label>{copy.message}</label>
              <textarea
                className="short"
                value={message}
                onChange={(e) => updateSession({ message: e.target.value })}
                placeholder={copy.placeholder}
                onKeyDown={(e) => {
                  if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) send();
                }}
              />
            </div>
            <input
              ref={imageInputRef}
              type="file"
              accept="image/jpeg,image/png,image/gif,image/webp"
              multiple
              hidden
              disabled={!supportsImages}
              onChange={addImages}
            />
            {images.length > 0 && (
              <div className="chat-attachment-list">
                {images.map((image, index) => (
                  <div className="chat-attachment" key={`${image.name}-${index}`}>
                    <img src={image.data_url} alt={image.name || copy.attachedImage} />
                    <span title={image.name}>{image.name}</span>
                    <button type="button" aria-label={copy.removeImage} onClick={() => updateSession((current) => ({ images: current.images.filter((_, itemIndex) => itemIndex !== index) }))}>
                      x
                    </button>
                  </div>
                ))}
              </div>
            )}
            {attachmentError && <p className="warning-line">{attachmentError}</p>}
            {!supportsImages && (
              <p className="meta-line">{copy.imagesNotSupported}</p>
            )}
            <div className="btn-row">
              <button
                type="button"
                className="btn btn-secondary"
                title={!supportsImages ? copy.imageUploadDisabled : undefined}
                onClick={() => {
                  if (!supportsImages) {
                    updateSession({ attachmentError: copy.imagesNotSupported });
                    return;
                  }
                  imageInputRef.current?.click();
                }}
                disabled={loading || agentActive || !supportsImages || images.length >= 4}
              >
                {supportsImages ? copy.addImage : copy.imageUploadDisabled}
              </button>
              <button
                type="button"
                className="btn btn-primary"
                onClick={send}
                disabled={loading || agentActive || (!message.trim() && (!supportsImages || images.length === 0))}
              >
                {loading ? copy.thinking : copy.send}
              </button>
            </div>
          </>
        )}
      </section>
    </div>
  );
}
