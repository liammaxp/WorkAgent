import { useEffect, useState } from "react";
import { api } from "../api/client.js";
import {
  Alert,
  EditorCard,
  LoadingBar,
  PageHeader,
  StatusBadge,
  useAsyncAction,
} from "../components/ui.jsx";

const STYLES = [
  { value: "concise", label: "简洁" },
  { value: "formal", label: "正式" },
  { value: "technical", label: "偏技术" },
  { value: "narrative", label: "项目叙事" },
];

let cachedCoverLetter = null;

export default function CoverLetter() {
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
    }, "求职信已保存");

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
    }, "求职信已生成");

  return (
    <>
      <PageHeader
        title="求职信"
        description="基于定制简历与职位描述生成 Cover Letter，保持内容真实一致。"
      />
      <LoadingBar loading={loading} />
      <Alert type="error" message={error} />
      <Alert type="success" message={success} />

      <section className="card">
        <h2 className="card-title">前置条件</h2>
        <div className="grid-2">
          <div>
            <span style={{ marginRight: 8, color: "var(--text-muted)" }}>职位描述</span>
            <StatusBadge ready={jobReady} />
          </div>
          <div>
            <span style={{ marginRight: 8, color: "var(--text-muted)" }}>定制简历</span>
            <StatusBadge ready={tailoredReady} />
          </div>
        </div>
        {!tailoredReady && (
          <p style={{ color: "var(--warning)", fontSize: 14, marginTop: 12 }}>
            建议先在简历页生成 tailored_resume，以确保求职信与定制简历一致。
          </p>
        )}
      </section>

      <section className="card">
        <h2 className="card-title">生成选项</h2>
        <div className="field">
          <label>写作风格</label>
          <select value={style} onChange={(e) => setStyle(e.target.value)}>
            {STYLES.map((item) => (
              <option key={item.value} value={item.value}>
                {item.label}
              </option>
            ))}
          </select>
        </div>
        <label style={{ display: "flex", alignItems: "center", gap: 8, fontSize: 14, color: "var(--text-muted)" }}>
          <input type="checkbox" checked={useGithub} onChange={(e) => setUseGithub(e.target.checked)} />
          使用 GitHub 项目证据（保守引用）
        </label>
        <div className="btn-row">
          <button type="button" className="btn btn-primary" onClick={generate} disabled={loading}>
            {loading ? "生成中…" : "生成求职信"}
          </button>
        </div>
        {outputPath && (
          <p style={{ color: "var(--text-muted)", fontSize: 13, marginTop: 12 }}>
            最近输出：{outputPath}
          </p>
        )}
      </section>

      <EditorCard
        title="Cover Letter"
        value={content}
        onChange={setContent}
        onSave={save}
        saving={loading}
        placeholder="生成的求职信将显示在这里…"
        short
      />
    </>
  );
}
