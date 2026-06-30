import { createContext, useCallback, useContext, useMemo, useRef, useState } from "react";
import AgentProgressModal from "./AgentProgressModal.jsx";
import { api } from "../api/client.js";

export class AgentCancelledError extends Error {
  constructor(message = "Agent task cancelled") {
    super(message);
    this.name = "AgentCancelledError";
  }
}

const AgentProgressContext = createContext(null);

function nowIso() {
  return new Date().toISOString();
}

function createMessage(role, content, meta = {}) {
  return {
    id: crypto.randomUUID(),
    role,
    content,
    timestamp: nowIso(),
    ...meta,
  };
}

function normalizeStages(stages = []) {
  return stages.map((stage) => ({
    id: stage.id,
    label: stage.label,
    status: stage.status || "pending",
    detail: stage.detail || "",
    projectName: stage.projectName || "",
    fieldType: stage.fieldType || "",
  }));
}

function isFinalModelStage(task) {
  if (!task?.modelStageIds?.length || !task.currentStageId) return false;
  const currentModelIndex = task.modelStageIds.indexOf(task.currentStageId);
  if (currentModelIndex === -1) return false;
  return currentModelIndex === task.modelStageIds.length - 1;
}

export function AgentProgressProvider({ children }) {
  const [task, setTask] = useState(null);
  const activeTaskRef = useRef(null);
  const userMessagesRef = useRef([]);
  const waitingResolverRef = useRef(null);
  const errorCloseResolverRef = useRef(null);
  const finishCloseTimerRef = useRef(null);

  const closeTask = useCallback(() => {
    errorCloseResolverRef.current?.();
    errorCloseResolverRef.current = null;
    if (finishCloseTimerRef.current) {
      window.clearTimeout(finishCloseTimerRef.current);
      finishCloseTimerRef.current = null;
    }
    waitingResolverRef.current?.reject?.(new AgentCancelledError());
    waitingResolverRef.current = null;
    activeTaskRef.current = null;
    userMessagesRef.current = [];
    setTask(null);
  }, []);

  const updateTask = useCallback((updater) => {
    setTask((current) => {
      if (!current) return current;
      return typeof updater === "function" ? updater(current) : { ...current, ...updater };
    });
  }, []);

  const setStageStatus = useCallback((stageId, status, detail = "") => {
    updateTask((current) => ({
      ...current,
      stages: current.stages.map((stage) =>
        stage.id === stageId ? { ...stage, status, detail } : stage,
      ),
      currentStageId: ["running", "waiting_for_user", "error"].includes(status) ? stageId : current.currentStageId,
    }));
  }, [updateTask]);

  const replaceStages = useCallback((stages = [], currentStageId = "") => {
    updateTask((current) => ({
      ...current,
      stages: normalizeStages(stages),
      currentStageId: currentStageId || current.currentStageId,
    }));
  }, [updateTask]);

  const updateStage = useCallback((stageId, patch = {}) => {
    updateTask((current) => ({
      ...current,
      stages: current.stages.map((stage) =>
        stage.id === stageId ? { ...stage, ...patch } : stage,
      ),
    }));
  }, [updateTask]);

  const addMessage = useCallback((role, content, meta = {}) => {
    if (!content) return null;
    const message = createMessage(role, content, meta);
    updateTask((current) => ({
      ...current,
      messages: [...current.messages, message],
    }));
    return message;
  }, [updateTask]);

  const runAgentWithProgress = useCallback(async ({
    title = "正在调用 Agent",
    stages = [],
    modelStageIds = [],
    initialMessage = "",
    action,
    completeDelayMs = 5000,
  }) => {
    if (activeTaskRef.current) return null;

    const controller = new AbortController();
    const taskId = crypto.randomUUID();
    const initialStages = normalizeStages(stages);
    const initialMessages = initialMessage ? [createMessage("system", initialMessage)] : [];
    userMessagesRef.current = [];
    activeTaskRef.current = { id: taskId, controller };
    setTask({
      id: taskId,
      title,
      stages: initialStages,
      modelStageIds,
      currentStageId: initialStages[0]?.id || null,
      messages: initialMessages,
      status: "running",
      error: "",
      errorDetail: null,
      canClose: false,
      minimized: false,
    });

    const assertActive = () => {
      const active = activeTaskRef.current;
      if (!active || active.id !== taskId || controller.signal.aborted) {
        throw new AgentCancelledError();
      }
    };

    const helpers = {
      agentTaskId: taskId,
      signal: controller.signal,
      addAgentMessage: (content, meta = {}) => addMessage("agent", content, meta),
      addSystemMessage: (content, meta = {}) => addMessage("system", content, meta),
      setStageStatus,
      replaceStages,
      updateStage,
      getUserMessages: () => [...userMessagesRef.current],
      getUserGuidance: () =>
        userMessagesRef.current
          .map((message) => message.content.trim())
          .filter(Boolean)
          .join("\n"),
      assertActive,
      askUserAndWait: async (question, stageId = null, detail = "Waiting for user reply", meta = {}) => {
        assertActive();
        const activeStageId = stageId || activeTaskRef.current?.currentStageId;
        addMessage("agent", question, meta);
        if (activeStageId) setStageStatus(activeStageId, "waiting_for_user", detail);
        const answer = await new Promise((resolve, reject) => {
          waitingResolverRef.current = { resolve, reject };
        });
        waitingResolverRef.current = null;
        assertActive();
        if (activeStageId) setStageStatus(activeStageId, "running", "User replied; continuing with the added STAR context");
        return answer;
      },
      runStage: async (stageId, labelOrAction, maybeAction) => {
        const label = typeof labelOrAction === "string" ? labelOrAction : "";
        const stageAction = typeof labelOrAction === "function" ? labelOrAction : maybeAction;
        assertActive();
        setStageStatus(stageId, "running", label);
        if (label) addMessage("system", label);
        try {
          const result = await stageAction(helpers);
          assertActive();
          setStageStatus(stageId, "done");
          return result;
        } catch (error) {
          if (error?.name === "AbortError" || error?.name === "AgentCancelledError" || controller.signal.aborted) {
            setStageStatus(stageId, "cancelled");
            throw new AgentCancelledError();
          }
          setStageStatus(stageId, "error", error.message || "Operation failed");
          throw error;
        }
      },
    };

    try {
      const result = await action(helpers);
      assertActive();
      updateTask((current) => ({
        ...current,
        status: "success",
        canClose: true,
        stages: current.stages.map((stage) => ({
          ...stage,
          status: stage.status === "pending" || stage.status === "running" ? "done" : stage.status,
        })),
        messages: [...current.messages, createMessage("system", "任务已完成。")],
      }));
      finishCloseTimerRef.current = window.setTimeout(() => {
        if (activeTaskRef.current?.id === taskId) closeTask();
      }, completeDelayMs);
      return result;
    } catch (error) {
      if (error?.name === "AbortError" || error?.name === "AgentCancelledError" || controller.signal.aborted) {
        closeTask();
        throw new AgentCancelledError();
      }

      updateTask((current) => ({
        ...current,
        status: "error",
        error: error.message || "Agent task failed",
        errorDetail: error.detail || null,
        canClose: true,
        stages: current.stages.map((stage) =>
          stage.status === "running" ? { ...stage, status: "error", detail: error.message || "" } : stage,
        ),
        messages: [...current.messages, createMessage("agent", error.message || "Agent task failed")],
      }));
      await new Promise((resolve) => {
        errorCloseResolverRef.current = resolve;
      });
      throw error;
    }
  }, [addMessage, closeTask, replaceStages, setStageStatus, updateStage, updateTask]);

  const cancelTask = useCallback(() => {
    const active = activeTaskRef.current;
    if (!active) {
      closeTask();
      return;
    }
    api.cancelAgentTask(active.id);
    api.cancelAgentTaskRun(active.id);
    active.controller.abort();
    setTask((current) => {
      if (!current) return current;
      return {
        ...current,
        status: "cancelled",
        stages: current.stages.map((stage) =>
          stage.status === "running" || stage.status === "waiting_for_user" || stage.status === "pending"
            ? { ...stage, status: "cancelled" }
            : stage,
        ),
      };
    });
    closeTask();
  }, [closeTask]);

  const minimizeTask = useCallback(() => {
    setTask((current) => (current ? { ...current, minimized: true } : current));
  }, []);

  const restoreTask = useCallback(() => {
    setTask((current) => (current ? { ...current, minimized: false } : current));
  }, []);

  const sendUserMessage = useCallback((content) => {
    const cleanContent = content.trim();
    if (!cleanContent || !activeTaskRef.current) return;
    const active = activeTaskRef.current;
    const message = createMessage("user", cleanContent);
    userMessagesRef.current = [...userMessagesRef.current, message];
    updateTask((current) => ({
      ...current,
      messages: [...current.messages, message],
    }));
    api.sendAgentTaskMessage(active.id, cleanContent, {
      signal: active.controller.signal,
    }).catch(() => null);
    if (waitingResolverRef.current) {
      waitingResolverRef.current.resolve(cleanContent);
      return;
    }
    if (!isFinalModelStage(task)) return;

    const currentStage = task.stages.find((stage) => stage.id === task.currentStageId);
    addMessage("system", "正在单独处理最后阶段补充信息...");
    api.sendAgentProgressGuidance({
      title: task.title,
      stage_label: currentStage?.label || currentStage?.detail || "",
      user_message: cleanContent,
      prior_messages: userMessagesRef.current,
    }, {
      signal: active.controller.signal,
      agentTaskId: active.id,
    })
      .then((data) => {
        if (!activeTaskRef.current || activeTaskRef.current.id !== active.id) return;
        addMessage("agent", data.answer || "已收到补充信息。");
      })
      .catch((error) => {
        if (error?.name === "AbortError" || error?.name === "AgentCancelledError") return;
        if (!activeTaskRef.current || activeTaskRef.current.id !== active.id) return;
        addMessage("system", error.message || "补充信息处理失败。");
      });
  }, [addMessage, task, updateTask]);

  const value = useMemo(() => ({
    active: Boolean(task),
    runAgentWithProgress,
  }), [runAgentWithProgress, task]);

  return (
    <AgentProgressContext.Provider value={value}>
      {children}
      <AgentProgressModal
        task={task}
        onCancel={cancelTask}
        onClose={closeTask}
        onMinimize={minimizeTask}
        onRestore={restoreTask}
        onSend={sendUserMessage}
      />
    </AgentProgressContext.Provider>
  );
}

export function useAgentProgress() {
  const context = useContext(AgentProgressContext);
  if (!context) {
    throw new Error("useAgentProgress must be used inside AgentProgressProvider");
  }
  return context;
}
