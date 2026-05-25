import { useEffect, useState } from "react";
import { api } from "../api/client.js";
import {
  Alert,
  EditorCard,
  LoadingBar,
  PageHeader,
  useAsyncAction,
} from "../components/ui.jsx";

export default function Resume() {
  const [resume, setResume] = useState("");
  const [tailored, setTailored] = useState("");
  const [useGithub, setUseGithub] = useState(false);
  const { loading, error, success, run } = useAsyncAction();

  const loadFiles = () =>
    run(async () => {
      const [base, custom] = await Promise.all([
        api.getFile("resume"),
        api.getFile("tailored_resume"),
      ]);
      setResume(base.content || "");
      setTailored(custom.content || "");
    });

  useEffect(() => {
    loadFiles();
  }, []);

  const saveResume = () =>
    run(async () => {
      await api.saveFile("resume", resume);
    }, "原始简历已保存");

  const saveTailored = () =>
    run(async () => {
      await api.saveFile("tailored_resume", tailored);
    }, "定制简历已保存");

  const generate = () =>
    run(async () => {
      const data = await api.tailorResume(useGithub);
      setTailored(data.content || "");
      return data;
    }, "定制简历已生成");

  return (
    <>
      <PageHeader
        title="简历"
        description="编辑原始 LaTeX 简历，并根据当前职位描述生成定制版本。"
      />
      <LoadingBar loading={loading} />
      <Alert type="error" message={error} />
      <Alert type="success" message={success} />

      <EditorCard
        title="原始简历 (resume.txt)"
        value={resume}
        onChange={setResume}
        onSave={saveResume}
        saving={loading}
        placeholder="LaTeX 简历内容…"
      />

      <section className="card">
        <h2 className="card-title">生成定制简历</h2>
        <label style={{ display: "flex", alignItems: "center", gap: 8, fontSize: 14, color: "var(--text-muted)" }}>
          <input type="checkbox" checked={useGithub} onChange={(e) => setUseGithub(e.target.checked)} />
          尝试使用 GitHub 上下文（需先在 GitHub 页面获取）
        </label>
        <div className="btn-row">
          <button type="button" className="btn btn-primary" onClick={generate} disabled={loading}>
            {loading ? "生成中…" : "根据职位描述生成"}
          </button>
        </div>
      </section>

      <EditorCard
        title="定制简历 (tailored_resume.txt)"
        value={tailored}
        onChange={setTailored}
        onSave={saveTailored}
        saving={loading}
        placeholder="生成后的 LaTeX 将显示在这里…"
      />
    </>
  );
}
