import { useEffect, useState } from "react";
import { api } from "../api/client.js";
import {
  Alert,
  EditorCard,
  LoadingBar,
  PageHeader,
  useAsyncAction,
} from "../components/ui.jsx";
import { text, useLanguage } from "../i18n.jsx";

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
      const data = await api.tailorResume(useGithub, allowProjectSelection);
      setTailored(data.content || "");
      setOutputPath(data.output_path || data.path || "");
      return data;
    }, copy.generated);

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
    </>
  );
}
