import { useEffect, useMemo, useState } from "react";
import { api } from "../api/client.js";
import {
  Alert,
  EditorCard,
  LoadingBar,
  PageHeader,
  StatusBadge,
  useAsyncAction,
} from "../components/ui.jsx";
import { text, useLanguage } from "../i18n.jsx";

const STYLE_VALUES = ["concise", "formal", "technical", "narrative"];

let cachedCoverLetter = null;

export default function CoverLetter() {
  const { language } = useLanguage();
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
  const [useGithub, setUseGithub] = useState(false);
  const [outputPath, setOutputPath] = useState("");
  const { loading, error, success, run } = useAsyncAction();

  useEffect(() => {
    run(async () => {
      const [jobData, status] = await Promise.all([
        api.getFile("job_description"),
        api.getStatus(),
      ]);
      if (cachedCoverLetter?.jobDescription === jobData.content) {
        setContent(cachedCoverLetter.content);
        setOutputPath(cachedCoverLetter.outputPath || "");
      } else if (!cachedCoverLetter) {
        setContent("");
        setOutputPath("");
      }
      setJobReady(status.files.job_description);
      setTailoredReady(status.files.tailored_resume);
    });
  }, []);

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
      setContent(data.content || "");
      setOutputPath(data.output_path || data.path || "");
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
        {outputPath && <p className="meta-line">{common.recentOutput}{outputPath}</p>}
      </section>

      <EditorCard
        title="Cover Letter"
        value={content}
        onChange={setContent}
        onSave={save}
        saving={loading}
        placeholder={copy.placeholder}
        short
      />
    </>
  );
}
