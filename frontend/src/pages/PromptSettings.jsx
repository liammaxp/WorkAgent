import { useEffect, useState } from "react";
import { api } from "../api/client.js";
import { Alert, LoadingBar, PageHeader, useAsyncAction } from "../components/ui.jsx";

export default function PromptSettings() {
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
    }, "Prompt 已保存并生效");

  const useExample = () => {
    if (example) {
      setPrompt(example);
    }
  };

  return (
    <>
      <PageHeader
        title="Prompt 设置"
        description="编辑 Agent 的系统 Prompt，让它适配你的背景、目标岗位、写作规则和证据使用方式。"
      />
      <LoadingBar loading={loading} />
      <Alert type="error" message={error} />
      <Alert type="success" message={success} />

      <div className="prompt-layout">
        <section className="card">
          <div className="prompt-card-header">
            <h2 className="card-title">当前 Prompt</h2>
            <span className="helper-text">{prompt.length} 字符</span>
          </div>
          <div className="field">
            <textarea
              className="prompt-editor"
              value={prompt}
              onChange={(event) => setPrompt(event.target.value)}
              placeholder="写入你的 Agent 工作规则、个人背景和输出偏好"
            />
          </div>
          <div className="btn-row">
            <button
              type="button"
              className="btn btn-primary"
              onClick={savePrompt}
              disabled={loading || !prompt.trim()}
            >
              保存 Prompt
            </button>
            <button
              type="button"
              className="btn btn-secondary"
              onClick={loadPrompt}
              disabled={loading}
            >
              重新加载
            </button>
          </div>
        </section>

        <aside className="card">
          <h2 className="card-title">示例 Prompt</h2>
          <p className="helper-paragraph">
            这个模板可以直接试用，也可以把姓名、背景、目标岗位、项目和写作偏好改成自己的。
          </p>
          <div className="btn-row">
            <button
              type="button"
              className="btn btn-secondary"
              onClick={useExample}
              disabled={loading || !example}
            >
              使用示例
            </button>
          </div>
          <pre className="prompt-example">{example}</pre>
        </aside>
      </div>
    </>
  );
}
