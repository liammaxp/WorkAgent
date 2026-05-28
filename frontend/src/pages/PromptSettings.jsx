import { useEffect, useState } from "react";
import { api } from "../api/client.js";
import { Alert, LoadingBar, PageHeader, useAsyncAction } from "../components/ui.jsx";
import { text, useLanguage } from "../i18n.jsx";

export default function PromptSettings() {
  const { language } = useLanguage();
  const copy = text[language].prompt;
  const [prompt, setPrompt] = useState("");
  const [example, setExample] = useState("");
  const { loading, error, success, run } = useAsyncAction();

  const loadPrompt = () =>
    run(async () => {
      const data = await api.getPrompt();
      setPrompt(data.content || "");
      setExample(data.example || "");
      return data;
    });

  useEffect(() => {
    loadPrompt();
  }, []);

  const savePrompt = () =>
    run(async () => {
      const data = await api.savePrompt(prompt);
      setPrompt(data.content || prompt);
      return data;
    }, copy.saved);

  const useExample = () => {
    if (example) setPrompt(example);
  };

  return (
    <>
      <PageHeader title={copy.title} description={copy.description} />
      <LoadingBar loading={loading} />
      <Alert type="error" message={error} />
      <Alert type="success" message={success} />

      <div className="prompt-layout">
        <section className="card">
          <div className="prompt-card-header">
            <h2 className="card-title">{copy.current}</h2>
            <span className="helper-text">{prompt.length} {copy.chars}</span>
          </div>
          <div className="field">
            <textarea
              className="prompt-editor"
              value={prompt}
              onChange={(event) => setPrompt(event.target.value)}
              placeholder={copy.placeholder}
            />
          </div>
          <div className="btn-row">
            <button type="button" className="btn btn-primary" onClick={savePrompt} disabled={loading || !prompt.trim()}>
              {copy.save}
            </button>
            <button type="button" className="btn btn-secondary" onClick={loadPrompt} disabled={loading}>
              {copy.reload}
            </button>
          </div>
        </section>

        <aside className="card">
          <h2 className="card-title">{copy.example}</h2>
          <p className="helper-paragraph">{copy.exampleText}</p>
          <div className="btn-row">
            <button type="button" className="btn btn-secondary" onClick={useExample} disabled={loading || !example}>
              {copy.useExample}
            </button>
          </div>
          <pre className="prompt-example">{example}</pre>
        </aside>
      </div>
    </>
  );
}
