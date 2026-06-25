import { useCallback, useEffect, useMemo, useState } from "react";
import { useLocation } from "react-router-dom";
import { api } from "../api/client.js";
import { fileChangedSinceAppOpened, readStoredBoolean, writeStoredBoolean } from "../session.js";
import {
  Alert,
  EditorCard,
  LoadingBar,
  OutputFileSelect,
  PageHeader,
  StatusBadge,
  useAsyncAction,
} from "../components/ui.jsx";
import { text, useLanguage } from "../i18n.jsx";

const STYLE_VALUES = ["concise", "formal", "technical", "narrative"];

let cachedCoverLetter = null;

export default function CoverLetter() {
  const { language } = useLanguage();
  const location = useLocation();
  const copy = text[language].coverLetter;
  const common = text[language].common;
  const styles = useMemo(
    () => STYLE_VALUES.map((value) => ({ value, label: copy.styles[value] })),
    [copy.styles]
  );
  const [content, setContent] = useState("");
  const [jobReady, setJobReady] = useState(false);
  const [tailoredReady, setTailoredReady] = useState(false);
  const [style, setStyle] = useState("concise");
  const [useGithub, setUseGithub] = useState(() => readStoredBoolean("workagent-cover-letter-use-github", false));
  const [outputPath, setOutputPath] = useState("");
  const [outputFiles, setOutputFiles] = useState([]);
  const { loading, error, success, run } = useAsyncAction();

  const loadCoverLetter = useCallback(() => {
    run(async () => {
      const [jobData, status] = await Promise.all([
        api.getFile("job_description"),
        api.getStatus(),
      ]);
      const coverLetterData = fileChangedSinceAppOpened(status, "cover_letter")
        ? await api.getFile("cover_letter")
        : { ready: false, content: "" };
      const latestContent = coverLetterData.ready ? coverLetterData.content || "" : "";
      const latestOutputPath = latestContent.trim()
        ? status.outputs?.cover_letters?.[0]?.path || ""
        : "";
      setContent(latestContent);
      setOutputPath(latestOutputPath);
      setOutputFiles(status.outputs?.cover_letters || []);
      cachedCoverLetter = {
        content: latestContent,
        jobDescription: jobData.content || "",
        outputPath: latestOutputPath,
      };
      setJobReady(status.files.job_description);
      setTailoredReady(status.files.tailored_resume);
    });
  }, [run]);

  useEffect(() => {
    loadCoverLetter();
  }, [location.pathname, loadCoverLetter]);

  useEffect(() => {
    const refreshOnFocus = () => loadCoverLetter();
    window.addEventListener("focus", refreshOnFocus);
    return () => window.removeEventListener("focus", refreshOnFocus);
  }, [loadCoverLetter]);

  useEffect(() => {
    writeStoredBoolean("workagent-cover-letter-use-github", useGithub);
  }, [useGithub]);

  const save = () =>
    run(async () => {
      const jobData = await api.getFile("job_description");
      await api.saveFile("cover_letter", content);
      cachedCoverLetter = {
        content,
        jobDescription: jobData.content || "",
        outputPath,
      };
    }, copy.saved);

  const generate = () =>
    run(async () => {
      const jobData = await api.getFile("job_description");
      const data = await api.generateCoverLetter({
        use_tailored_resume: true,
        use_github_context: useGithub,
        style,
      });
      const status = await api.getStatus();
      setContent(data.content || "");
      setOutputPath(data.output_path || data.path || "");
      setOutputFiles(status.outputs?.cover_letters || []);
      cachedCoverLetter = {
        content: data.content || "",
        jobDescription: jobData.content || "",
        outputPath: data.output_path || data.path || "",
      };
      return data;
    }, copy.generated);

  return (
    <>
      <PageHeader title={copy.title} description={copy.description} />
      <LoadingBar loading={loading} />
      <Alert type="error" message={error} />
      <Alert type="success" message={success} />

      <section className="card">
        <h2 className="card-title">{copy.prerequisites}</h2>
        <div className="grid-2">
          <div>
            <span className="muted-label">{copy.jobDescription}</span>
            <StatusBadge ready={jobReady} />
          </div>
          <div>
            <span className="muted-label">{copy.tailoredResume}</span>
            <StatusBadge ready={tailoredReady} />
          </div>
        </div>
        {!tailoredReady && <p className="warning-line">{copy.tailoredWarning}</p>}
      </section>

      <section className="card">
        <h2 className="card-title">{common.generateOptions}</h2>
        <div className="field">
          <label>{copy.style}</label>
          <select value={style} onChange={(e) => setStyle(e.target.value)}>
            {styles.map((item) => (
              <option key={item.value} value={item.value}>
                {item.label}
              </option>
            ))}
          </select>
        </div>
        <label className="inline-check">
          <input type="checkbox" checked={useGithub} onChange={(e) => setUseGithub(e.target.checked)} />
          {copy.useGithub}
        </label>
        <div className="btn-row">
          <button type="button" className="btn btn-primary" onClick={generate} disabled={loading}>
            {loading ? copy.generating : copy.generate}
          </button>
        </div>

      </section>

      <EditorCard
        title="Cover Letter"
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
                cachedCoverLetter = null;
              }
            }, language === "zh" ? "输出文件已删除" : "Output file deleted")}
          />
        }
        short
      />
    </>
  );
}
