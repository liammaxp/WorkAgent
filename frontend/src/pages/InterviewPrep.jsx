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

export default function InterviewPrep() {
  const { language } = useLanguage();
  const copy = text[language].interview;
  const common = text[language].common;
  const [content, setContent] = useState("");
  const [useGithub, setUseGithub] = useState(true);
  const [outputPath, setOutputPath] = useState("");
  const { loading, error, success, run } = useAsyncAction();

  useEffect(() => {
    run(async () => {
      const [data, status] = await Promise.all([
        api.getFile("interview_prep"),
        api.getStatus(),
      ]);
      setContent(data.content || "");
      setOutputPath(status.outputs?.interview_prep?.[0]?.path || "");
    });
  }, []);

  const save = () =>
    run(async () => {
      await api.saveFile("interview_prep", content);
    }, copy.saved);

  const generate = () =>
    run(async () => {
      const data = await api.generateInterviewPrep(useGithub);
      setContent(data.content || "");
      setOutputPath(data.output_path || data.path || "");
      return data;
    }, copy.generated);

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
          <button type="button" className="btn btn-primary" onClick={generate} disabled={loading}>
            {loading ? copy.generating : copy.generate}
          </button>
        </div>
        {outputPath && <p className="meta-line">{common.recentOutput}{outputPath}</p>}
      </section>

      <EditorCard
        title="Interview Prep Notes"
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
