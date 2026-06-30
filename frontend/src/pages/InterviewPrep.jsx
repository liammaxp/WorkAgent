import { useCallback, useEffect, useState } from "react";
import { api } from "../api/client.js";
import { useAgentProgress } from "../agentProgress/AgentProgressContext.jsx";
import { fileChangedSinceAppOpened, readStoredBoolean, writeStoredBoolean } from "../session.js";
import {
  Alert,
  EditorCard,
  LoadingBar,
  OutputFileSelect,
  PageHeader,
  useAsyncAction,
} from "../components/ui.jsx";
import { text, useLanguage } from "../i18n.jsx";

export default function InterviewPrep() {
  const { language } = useLanguage();
  const copy = text[language].interview;
  const common = text[language].common;
  const [content, setContent] = useState("");
  const [useGithub, setUseGithub] = useState(() => readStoredBoolean("workagent-interview-use-github", true));
  const [outputPath, setOutputPath] = useState("");
  const [outputFiles, setOutputFiles] = useState([]);
  const { loading, error, success, run } = useAsyncAction();
  const { active: agentActive, runAgentWithProgress } = useAgentProgress();

  const refreshOutput = useCallback(() => {
    run(async () => {
      const status = await api.getStatus();
      if (!fileChangedSinceAppOpened(status, "interview_prep")) {
        setContent("");
        setOutputPath("");
        setOutputFiles(status.outputs?.interview_prep || []);
        return null;
      }
      const data = await api.getFile("interview_prep");
      const latestContent = data.ready ? data.content || "" : "";
      setContent(latestContent);
      setOutputPath(latestContent.trim() ? status.outputs?.interview_prep?.[0]?.path || "" : "");
      setOutputFiles(status.outputs?.interview_prep || []);
      return data;
    });
  }, [run]);

  useEffect(() => {
    refreshOutput();
  }, [refreshOutput]);

  useEffect(() => {
    const refreshOnFocus = () => refreshOutput();
    window.addEventListener("focus", refreshOnFocus);
    return () => window.removeEventListener("focus", refreshOnFocus);
  }, [refreshOutput]);

  useEffect(() => {
    writeStoredBoolean("workagent-interview-use-github", useGithub);
  }, [useGithub]);

  const save = () =>
    run(async () => {
      await api.saveFile("interview_prep", content);
    }, copy.saved);

  const generate = () => {
    setContent("");
    setOutputPath("");
    return run(async () => {
      const { data, status } = await runAgentWithProgress({
        title: copy.generating || "正在生成面试准备",
        initialMessage: `Agent：我会读取 job_description.txt、${useGithub ? "tailored_resume/resume、memory 和 GitHub context" : "tailored_resume/resume 和 memory"}，生成面试准备。`,
        stages: [
          { id: "generate", label: "发送职位/简历/记忆生成面试六部分内容" },
          { id: "refresh", label: "读取 interview_prep 输出文件" },
        ],
        modelStageIds: ["generate"],
        action: async (progress) => {
          const data = await progress.runStage("generate", `正在发送 use_github_context=${useGithub}；后端会合并 Role focus、技术题、项目讲述、STAR、准备缺口、反问问题`, () =>
            api.generateInterviewPrep(useGithub, {
              signal: progress.signal,
              agentProgressMessages: progress.getUserMessages(),
              agentTaskId: progress.agentTaskId,
            }),
          );
          const status = await progress.runStage("refresh", "正在读取 interview_prep.txt 和 outputs/interview_prep 最新文件", () => api.getStatus());
          progress.addAgentMessage("面试准备内容已生成。");
          return { data, status };
        },
      });
      setContent(data.content || "");
      setOutputPath(data.output_path || data.path || "");
      setOutputFiles(status.outputs?.interview_prep || []);
      return data;
    }, copy.generated);
  };

  return (
    <>
      <PageHeader title={copy.title} description={copy.description} />
      <LoadingBar loading={loading} />
      <Alert type="error" message={error} />
      <Alert type="success" message={success} />

      <section className="card">
        <h2 className="card-title">{common.generateOptions}</h2>
        <label className="inline-check">
          <input type="checkbox" checked={useGithub} onChange={(e) => setUseGithub(e.target.checked)} />
          {copy.useGithub}
        </label>
        <div className="btn-row">
          <button type="button" className="btn btn-primary" onClick={generate} disabled={loading || agentActive}>
            {loading ? copy.generating : copy.generate}
          </button>
        </div>

      </section>

      <EditorCard
        title="Interview Prep Notes"
        value={content}
        onChange={setContent}
        onSave={save}
        saving={loading}
        placeholder={copy.placeholder}
        extraActions={
          <OutputFileSelect
            files={outputFiles}
            value={outputPath}
            disabled={loading}
            inline
            onSelect={(path) => run(async () => {
              const data = await api.getOutputFile(path);
              setContent(data.content || "");
              setOutputPath(path);
            })}
            onDelete={(path) => run(async () => {
              await api.deleteOutputFile(path);
              setOutputFiles((files) => files.filter((file) => file.path !== path));
              if (outputPath === path) {
                setContent("");
                setOutputPath("");
              }
            }, language === "zh" ? "输出文件已删除" : "Output file deleted")}
          />
        }
        short
      />
    </>
  );
}
