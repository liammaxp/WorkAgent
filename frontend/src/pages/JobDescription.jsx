import { useEffect, useState } from "react";
import { api } from "../api/client.js";
import {
  Alert,
  EditorCard,
  LoadingBar,
  PageHeader,
  useAsyncAction,
} from "../components/ui.jsx";

export default function JobDescription() {
  const [content, setContent] = useState("");
  const [analysis, setAnalysis] = useState("");
  const { loading, error, success, run } = useAsyncAction();

  useEffect(() => {
    run(async () => {
      const data = await api.getFile("job_description");
      setContent(data.content || "");
    });
  }, []);

  const save = () =>
    run(async () => {
      await api.saveJobDescription(content);
    }, "职位描述已保存");

  const analyze = () =>
    run(async () => {
      if (content.trim()) await api.saveJobDescription(content);
      const data = await api.analyzeJob(false);
      setAnalysis(data.analysis || "");
      return data;
    }, "职位分析完成");

  return (
    <>
      <PageHeader
        title="职位描述"
        description="粘贴目标岗位 JD，保存后一键分析匹配度与重点技能。"
      />
      <LoadingBar loading={loading} />
      <Alert type="error" message={error} />
      <Alert type="success" message={success} />

      <EditorCard
        title="职位描述内容"
        value={content}
        onChange={setContent}
        placeholder="粘贴职位描述…"
      />

      <section className="card">
        <div className="btn-row">
          <button type="button" className="btn btn-secondary" onClick={save} disabled={loading}>
            保存
          </button>
          <button type="button" className="btn btn-primary" onClick={analyze} disabled={loading}>
            {loading ? "分析中…" : "分析职位"}
          </button>
        </div>
      </section>

      {analysis && (
        <section className="card">
          <h2 className="card-title">分析结果</h2>
          <pre style={{ whiteSpace: "pre-wrap", fontFamily: "inherit", margin: 0, fontSize: 14, lineHeight: 1.7 }}>
            {analysis}
          </pre>
        </section>
      )}
    </>
  );
}
