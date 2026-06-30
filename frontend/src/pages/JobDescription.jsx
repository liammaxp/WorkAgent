import { useEffect, useState } from "react";
import { useLocation, useNavigate } from "react-router-dom";
import { api } from "../api/client.js";
import { useAgentProgress } from "../agentProgress/AgentProgressContext.jsx";
import {
  Alert,
  EditorCard,
  LoadingBar,
  OutputFileSelect,
  PageHeader,
  useAsyncAction,
} from "../components/ui.jsx";
import { text, useLanguage } from "../i18n.jsx";

let cachedAnalysis = null;

const JD_SAVED_EVENT = "workagent-jd-saved";

function notifyJobDescriptionSaved(result) {
  if (result?.tailored_resume_cleared) {
    window.dispatchEvent(new CustomEvent(JD_SAVED_EVENT));
  }
}

export default function JobDescription() {
  const { language } = useLanguage();
  const location = useLocation();
  const navigate = useNavigate();
  const copy = text[language].job;
  const [content, setContent] = useState("");
  const [analysis, setAnalysis] = useState("");
  const [outputFiles, setOutputFiles] = useState([]);
  const [outputPath, setOutputPath] = useState("");
  const [routeError, setRouteError] = useState("");
  const { loading, error, success, run } = useAsyncAction();
  const { active: agentActive, runAgentWithProgress } = useAgentProgress();

  useEffect(() => {
    const nextRouteError = location.state?.routeError;
    if (!nextRouteError) return;
    setRouteError(String(nextRouteError));
    navigate(location.pathname, { replace: true, state: {} });
  }, [location.pathname, location.state, navigate]);

  useEffect(() => {
    run(async () => {
      const [jobData, status] = await Promise.all([
        api.getFile("job_description"),
        api.getStatus(),
      ]);
      setContent(jobData.content || "");
      setOutputFiles(status.outputs?.analysis || []);
      if (
        cachedAnalysis?.jobDescription === jobData.content
      ) {
        setAnalysis(cachedAnalysis.content);
      } else if (!cachedAnalysis) {
        setAnalysis("");
      } else {
        setAnalysis("");
      }
    });
  }, [run]);

  const save = () =>
    run(async () => {
      const result = await api.saveJobDescription(content);
      notifyJobDescriptionSaved(result);
      if (
        cachedAnalysis?.jobDescription !== content
      ) {
        cachedAnalysis = null;
        setAnalysis("");
      }
    }, copy.saved);

  const analyze = () =>
    run(async () => {
      const { data, status } = await runAgentWithProgress({
        title: copy.analyzing || "正在分析职位",
        initialMessage: "Agent：我会保存当前 JD，并把 job_description.txt 发送给职位分析 Agent。",
        stages: [
          { id: "save", label: "保存 job_description.txt" },
          { id: "analyze", label: "发送 JD 生成职位分析 JSON" },
          { id: "refresh", label: "读取分析历史和输出文件" },
        ],
        modelStageIds: ["analyze"],
        action: async (progress) => {
          if (content.trim()) {
            await progress.runStage("save", `正在保存 ${content.trim().length} 个字符到 job_description.txt`, async () => {
              const result = await api.saveJobDescription(content);
              notifyJobDescriptionSaved(result);
            });
          } else {
            progress.setStageStatus("save", "done");
          }
          const data = await progress.runStage("analyze", "正在发送 job_description.txt，要求 Agent 返回 analysis/company/role JSON", () =>
            api.analyzeJob(false, {
              signal: progress.signal,
              agentProgressMessages: progress.getUserMessages(),
              agentTaskId: progress.agentTaskId,
            }),
          );
          const status = await progress.runStage("refresh", "正在读取 outputs/analysis 最新文件和状态元数据", () => api.getStatus());
          progress.addAgentMessage("职位分析已完成。");
          return { data, status };
        },
      });
      setAnalysis(data.analysis || "");
      setOutputPath(data.analysis_path || "");
      setOutputFiles(status.outputs?.analysis || []);
      cachedAnalysis = {
        content: data.analysis || "",
        jobDescription: content,
      };
      return data;
    }, copy.analyzed);

  return (
    <>
      <PageHeader title={copy.title} description={copy.description} />
      <LoadingBar loading={loading} />
      <Alert type="error" message={routeError} />
      <Alert type="error" message={error} />
      <Alert type="success" message={success} />

      <EditorCard
        title={copy.editorTitle}
        value={content}
        onChange={setContent}
        placeholder={copy.placeholder}
      />

      <section className="card">
        <div className="btn-row">
          <button type="button" className="btn btn-secondary" onClick={save} disabled={loading}>
            {text[language].common.save}
          </button>
          <button type="button" className="btn btn-primary" onClick={analyze} disabled={loading || agentActive}>
            {loading ? copy.analyzing : copy.analyze}
          </button>
          <OutputFileSelect
            files={outputFiles}
            value={outputPath}
            disabled={loading}
            inline
            onSelect={(path) => run(async () => {
              const data = await api.getOutputFile(path);
              setAnalysis(data.content || "");
              setOutputPath(path);
            })}
            onDelete={(path) => run(async () => {
              await api.deleteOutputFile(path);
              setOutputFiles((files) => files.filter((file) => file.path !== path));
              if (outputPath === path) {
                setAnalysis("");
                setOutputPath("");
                cachedAnalysis = null;
              }
            }, language === "zh" ? "输出文件已删除" : "Output file deleted")}
          />
        </div>

      </section>

      {analysis && (
        <section className="card">
          <h2 className="card-title">{copy.result}</h2>
          <pre style={{ whiteSpace: "pre-wrap", fontFamily: "inherit", margin: 0, fontSize: 14, lineHeight: 1.7 }}>
            {analysis}
          </pre>
        </section>
      )}
    </>
  );
}
