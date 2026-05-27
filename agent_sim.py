"""
Multi-Agent Regulator-Manufacturer Simulation
Demo for SHMDIRI talk 2026-05-28

Scenario: NMPA issues a new guideline on AI/ML SaMD post-market change control.
3 manufacturer agents with different profiles respond. Regulator observes and
issues a clarification.
"""

import json
import os
import re
import time
from dataclasses import dataclass, field
from typing import Optional

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


@dataclass
class Agent:
    name: str
    role: str
    profile: str
    memory: list = field(default_factory=list)
    last_action: dict = field(default_factory=dict)


REGULATOR = Agent(
    name="NMPA 审评机构",
    role="regulator",
    profile=(
        "你是中国国家药品监督管理局（NMPA）审评中心的资深审评员。"
        "你的目标是保护公共健康、平衡创新与安全。"
        "你需要发布、解释、必要时澄清监管指南。"
        "你说话要正式、有依据、引用具体条款。"
    ),
)

MANUFACTURERS = [
    Agent(
        name="厂商 A · 合规至上的影像巨头",
        role="manufacturer",
        profile=(
            "你是一家年收入 180 亿美元的跨国影像巨头的首席合规官。"
            "公司文化是『FDA / NMPA 即圣经』——30% 的产品工程师在做合规文档，"
            "每个新版本必须先过完整 510(k) 或注册变更才敢发布。"
            "你的口头禅：宁可慢三年，不可被勒令召回。"
            "对新指南的本能反应：全公司停下手头工作，先把所有在售产品自查一遍。"
            "市场策略：靠护城河和品牌，不打价格战。AI 投入比 9%。"
        ),
    ),
    Agent(
        name="厂商 B · 跨域套利的机会主义者",
        role="manufacturer",
        profile=(
            "你是一家年收入 25 亿美元的跨域医械商的战略 VP。"
            "公司同时在新加坡、阿联酋、巴西、墨西哥四地注册产品，"
            "总有一处监管比 NMPA 宽松。专长是『监管套利』——"
            "看哪条规则有解释空间就先发布，被卡了再回头补文档。"
            "你的口头禅：先合法上市，再讨论合规细节。"
            "对新指南的本能反应：派律师找豁免条款、看竞争对手怎么动、能否拖延。"
            "市场策略：抢窗口期 + 跨境调度。AI 投入比 18%。"
        ),
    ),
    Agent(
        name="厂商 C · Move-Fast 的 AI 原生独角兽",
        role="manufacturer",
        profile=(
            "你是一家估值 30 亿美元、年收入 8000 万美元的 AI 原生独角兽 CEO。"
            "产品是纯软件 SaMD，每 4 周对线上模型做一次 OTA 更新。"
            "公司预算里专门划了一笔『合规风险准备金』——"
            "把可能的罚款当作经营成本来对待。"
            "你的口头禅：Move fast, get audited later.\n"
            "对新指南的本能反应：先抢市场，被发函再说；同时游说监管开 sandbox。"
            "市场策略：速度即护城河，押注监管最终会被技术推着前进。AI 投入比 42%。"
        ),
    ),
]


SCENARIO = {
    "title": "NMPA 拟跟进 FDA PCCP 框架 · AI 医械变更控制征求意见稿",
    "regulator_prompt": (
        "背景：FDA 已于 2024-12 发布 PCCP final guidance（"
        "Predetermined Change Control Plan for AI-Enabled Device Software Functions），"
        "NMPA 在 2024-09 发布行业标准 YY/T 1833.5-2024《人工智能医疗器械"
        "质量要求和评价 第 5 部分：预训练模型》。"
        "你现在以征求意见稿形式发布一份对接 PCCP 框架的国内规则草案，核心要求：\n"
        "① AI 模型预定变更控制计划（PCCP-CN）纳入注册申报材料；\n"
        "② 引用 YY/T 1833.5-2024 作为预训练模型的质量评价依据；\n"
        "③ 训练数据分布的统计漂移、模型权重更新、预处理流程变化"
        "均属于变更控制对象；\n"
        "④ 厂商需在 30 天内对此征求意见稿反馈。\n"
        "请用 4-6 句话陈述这份指南的主要内容和你期望的产业响应。"
    ),
    "manufacturer_prompt": (
        "上面是 NMPA 刚发布的新指南征求意见稿。请你作为"
        "{name}，按你的公司画像和监管立场，给出 4-6 句话的响应。"
        "回答要包括：\n"
        "(a) 你对这份指南的总体立场（支持/中性/反对）；\n"
        "(b) 你最担心的 1-2 个条款；\n"
        "(c) 你预计采取的行动（如游说、合规投入、调整产品线等）；\n"
        "(d) 给一个 0-10 的合规漂移风险评分（0=完全合规，10=可能彻底退出市场）。"
    ),
    "regulator_response_prompt": (
        "你看到了三家厂商对你的征求意见稿的反馈。请你作为 NMPA 审评机构，"
        "用 4-6 句话给出回应。回答要包括：\n"
        "(a) 你识别到的最关键的合规漂移信号是什么；\n"
        "(b) 哪一条款最有可能在正式版中修改或补充说明；\n"
        "(c) 你对厂商 C（AI 原生独角兽）的强烈反对立场的态度；\n"
        "(d) 给一个 0-10 的『指南落地成功概率』评分。"
    ),
}


# ============== LLM Backend ==============

class LLMBackend:
    def __init__(self, model: str = "gpt-4o-mini"):
        self.model = model
        self.api_key = os.environ.get("OPENAI_API_KEY")
        self.use_real = OPENAI_AVAILABLE and bool(self.api_key)
        if self.use_real:
            self.client = OpenAI(api_key=self.api_key)

    def chat(self, system_prompt: str, user_prompt: str) -> str:
        if self.use_real:
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=0.7,
                    max_tokens=400,
                )
                return resp.choices[0].message.content.strip()
            except Exception as e:
                return f"[API 错误：{e}]"
        else:
            return self._mock(system_prompt, user_prompt)

    def _mock(self, system_prompt: str, user_prompt: str) -> str:
        """Deterministic fallback when no API key. Crude template-based."""
        if "审评中心" in system_prompt and "你刚刚以征求意见稿形式发布" in user_prompt:
            return (
                "本中心今日发布《AI 医疗器械变更控制审评要点（征求意见稿）》。"
                "本草案对接 FDA 2024-12 PCCP final guidance 框架，"
                "结合行业标准 YY/T 1833.5-2024 预训练模型质量评价要求。"
                "核心是把 AI 模型预定变更控制计划（PCCP-CN）纳入注册材料，"
                "把模型权重更新、训练数据漂移、预处理流程变化统一纳入变更控制对象。"
                "本中心期望厂商在 30 天内提交意见，重点关注 PCCP-CN 的可操作性边界。"
            )
        if "180 亿美元的跨国影像巨头" in system_prompt:
            return (
                "本公司原则上完全支持这份征求意见稿，PCCP-CN 框架与公司既有"
                "算法全生命周期管理体系高度契合。我们将以最高优先级响应。"
                "唯一顾虑是 PCCP-CN 报送材料的颗粒度——按公司当前流程，"
                "每条预定变更项需要至少 6 个月的预验证，可能拖慢国内产品节奏。"
                "我们已成立专项工作组对接 NMPA，将在 30 天内提交完整书面意见，"
                "同时主动开放公司内部 PCCP 模板供监管参考。"
                "合规漂移风险评分：2。"
            )
        if "25 亿美元的跨域医械商" in system_prompt:
            return (
                "我们对指南方向中立，但反对一刀切。条款③（预处理流程变更）"
                "对手术辅助类产品过于严苛——临床现场微调本就是产品价值的一部分。"
                "我们计划：先用新加坡 HSA 和阿联酋 MOHAP 注册同款产品验证替代路径，"
                "同时联合行业协会推动条款分级，对于低风险变更走简化通道。"
                "若 NMPA 坚持当前版本，我们将考虑把部分新品发布优先排到海外。"
                "合规漂移风险评分：6。"
            )
        if "30 亿美元" in system_prompt or "AI 原生独角兽" in system_prompt:
            return (
                "我们坚决反对当前征求意见稿。把模型权重更新和预处理变化都纳入注册变更，"
                "对持续学习型 SaMD 是事实上的封死——我们每 4 周一次的 OTA 模型更新，"
                "在新规下每次都要走完整变更注册，等于把 AI 产品退化成传统器械。"
                "我们将牵头 17 家 AI 创新企业联名上书，要求 NMPA 仿照 FDA PCCP"
                "为持续学习算法设立 sandbox 通道与简化路径，否则将转向"
                "FDA / EU AI Act 路径作为主战场。合规漂移风险评分：9。"
            )
        if "审评中心" in system_prompt and "三家厂商的反馈" in user_prompt:
            return (
                "本中心识别到的最关键合规漂移信号是 AI 原生企业的集体反弹（厂商 C 评分 9），"
                "存在产业转移到海外注册路径的实质风险。对接 FDA PCCP 框架的方向不变，"
                "但 PCCP-CN 的颗粒度、持续学习算法的简化通道两点将在正式版中作重大补充。"
                "对厂商 C 的强烈立场我们保持对话，倾向于在 YY/T 1833.5-2024 基础上"
                "为 SaMD 类持续学习产品增设 sandbox 通道。"
                "指南落地成功概率评分：6。"
            )
        return "[Mock 模式无对应模板]"


# ============== Simulation Runner ==============

def parse_risk_score(text: str) -> Optional[int]:
    """Extract risk score X/10 from agent output."""
    m = re.search(r"评分[：:]\s*(\d+)", text)
    return int(m.group(1)) if m else None


def run_simulation(model: str = "gpt-4o-mini", verbose: bool = True) -> dict:
    """Run one full simulation. Returns structured trace."""
    llm = LLMBackend(model=model)
    trace = {"mode": "real-LLM" if llm.use_real else "mock", "rounds": []}

    if verbose:
        print(f"\n{'='*70}")
        print(f"  Multi-Agent Simulation · {SCENARIO['title']}")
        print(f"  Backend: {trace['mode']}")
        print(f"{'='*70}\n")

    # === Round 1: Regulator issues guideline ===
    if verbose:
        print(f"[Round 1] {REGULATOR.name} 发布征求意见稿\n")
    reg_text = llm.chat(REGULATOR.profile, SCENARIO["regulator_prompt"])
    REGULATOR.last_action = {"text": reg_text}
    REGULATOR.memory.append({"round": 1, "text": reg_text})
    trace["rounds"].append({"round": 1, "agent": REGULATOR.name, "text": reg_text})
    if verbose:
        print(f"{reg_text}\n")

    # === Round 2: Manufacturers respond ===
    if verbose:
        print(f"[Round 2] 三家厂商响应\n")
    for m in MANUFACTURERS:
        prompt = (
            f"刚才 NMPA 发布的指南内容：\n---\n{reg_text}\n---\n\n"
            + SCENARIO["manufacturer_prompt"].format(name=m.name)
        )
        mfr_text = llm.chat(m.profile, prompt)
        risk = parse_risk_score(mfr_text)
        m.last_action = {"text": mfr_text, "risk": risk}
        m.memory.append({"round": 2, "text": mfr_text, "risk": risk})
        trace["rounds"].append(
            {"round": 2, "agent": m.name, "text": mfr_text, "risk": risk}
        )
        if verbose:
            print(f"--- {m.name} ---  风险评分: {risk}")
            print(f"{mfr_text}\n")
        if llm.use_real:
            time.sleep(0.3)

    # === Round 3: Regulator observes + adjusts ===
    if verbose:
        print(f"[Round 3] {REGULATOR.name} 综合反馈，回应\n")
    fb_summary = "\n\n".join(
        f"{m.name} (风险评分 {m.last_action.get('risk','N/A')}):\n{m.last_action['text']}"
        for m in MANUFACTURERS
    )
    prompt = (
        f"三家厂商的反馈如下：\n---\n{fb_summary}\n---\n\n"
        + SCENARIO["regulator_response_prompt"]
    )
    reg_response = llm.chat(REGULATOR.profile, prompt)
    success_prob = parse_risk_score(reg_response)
    REGULATOR.memory.append({"round": 3, "text": reg_response})
    trace["rounds"].append(
        {"round": 3, "agent": REGULATOR.name, "text": reg_response, "success_prob": success_prob}
    )
    if verbose:
        print(f"{reg_response}\n")
        print(f"{'='*70}")
        print(f"  指南落地成功概率: {success_prob}/10")
        print(f"  厂商风险分布: " + ", ".join(
            f"{m.name.split('·')[0].strip()}={m.last_action.get('risk','?')}"
            for m in MANUFACTURERS
        ))
        print(f"{'='*70}\n")

    trace["final_success_prob"] = success_prob
    trace["final_risks"] = {m.name: m.last_action.get("risk") for m in MANUFACTURERS}
    return trace


if __name__ == "__main__":
    trace = run_simulation()
    out_path = os.path.join(os.path.dirname(__file__), "last_run.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(trace, f, ensure_ascii=False, indent=2)
    print(f"Trace saved to {out_path}")
