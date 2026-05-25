import { useEffect, useState } from "react";
import { api } from "../api/client.js";
import {
  Alert,
  EditorCard,
  LoadingBar,
  PageHeader,
  useAsyncAction,
} from "../components/ui.jsx";

export default function InterviewPrep() {
  const [content, setContent] = useState("");
  const [useGithub, setUseGithub] = useState(true);
  const { loading, error, success, run } = useAsyncAction();

  useEffect(() => {
    run(async () => {
      const data = await api.getFile("interview_prep");
      setContent(data.content || "");
    });
  }, []);

  const save = () =>
    run(async () => {
      await api.saveFile("interview_prep", content);
    }, "面试准备已保存");

  const generate = () =>
    run(async () => {
      const data = await api.generateInterviewPrep(useGithub);
      setContent(data.content || "");
      return data;
    }, "面试准备已生成");

  return (
    <>
      <PageHeader
        title="面试准备"
        description="根据职位描述与简历生成技术问题、STAR 回答要点与项目讲述。"
      />
      <LoadingBar loading={loading} />
      <Alert type="error" message={error} />
      <Alert type="success" message={success} />

      <section className="card">
        <h2 className="card-title">生成选项</h2>
        <label style={{ display: "flex", alignItems: "center", gap: 8, fontSize: 14, color: "var(--text-muted)" }}>
          <input type="checkbox" checked={useGithub} onChange={(e) => setUseGithub(e.target.checked)} />
          参考 GitHub 项目证据
        </label>
        <div className="btn-row">
          <button type="button" className="btn btn-primary" onClick={generate} disabled={loading}>
            {loading ? "生成中…" : "生成面试准备"}
          </button>
        </div>
      </section>

      <EditorCard
        title="Interview Prep Notes"
        value={content}
        onChange={setContent}
        onSave={save}
        saving={loading}
        placeholder="面试准备笔记将显示在这里…"
        short
      />
    </>
  );
}
