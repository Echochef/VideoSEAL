# Video Reward（论文写法，简版）

我们将视频任务的奖励定义为若干可解释子项的加权和，并通过视觉对齐门控（visual gate）抑制“语言捷径”（即不看证据也能猜对）的策略。所有系数均通过环境变量暴露，便于消融与复现实验。

---

## 1. 总体形式：加权和 + 裁剪

记权重（由环境变量设置）：

$$
w_{\text{final}}=\texttt{VIDEO\_REWARD\_FINAL\_WEIGHT},\;
w_{\text{tool}}=\texttt{VIDEO\_REWARD\_TOOL\_HIT\_WEIGHT},\;
w_{\text{vis}}=\texttt{VIDEO\_REWARD\_VISUAL\_INSPECT\_WEIGHT},\;
w_{\text{judge}}=\texttt{VIDEO\_REWARD\_JUDGE\_WEIGHT},\;
w_{\text{fmt}}=\texttt{VIDEO\_REWARD\_FORMAT\_WEIGHT}.
$$

定义裁剪函数 $\mathrm{clip}_{[0,1]}(x)=\min(1,\max(0,x))$。最终奖励为：

$$
R=\mathrm{clip}_{[0,1]}\Big(
w_{\text{final}}\cdot r_{\text{final}}
+w_{\text{tool}}\cdot r_{\text{tool}}
+w_{\text{vis}}\cdot r_{\text{vis}}
+w_{\text{judge}}\cdot r_{\text{judge}}
+w_{\text{fmt}}\cdot r_{\text{fmt}}
\Big).
$$

---

## 2. 关键子项的定义（与论文直接对应）

### 2.1 最终正确性（final correctness）

设 $y\in\{0,1\}$ 为最终答案正确性（MCQ 用 exact match；开放式用 ROUGE-L 与阈值比较），则：

$$
r_{\text{final}}=y\cdot g(\mathrm{IoU}_{\max}).
$$

其中 $g(\cdot)$ 为视觉门控（见 3），用于要求“答对”必须伴随视觉证据对齐。

### 2.2 工具命中（tool hit）

设参考时间区间集合为 $G$，工具产生的候选区间集合为 $\mathcal{P}$。对任一候选区间 $P\in\mathcal{P}$ 定义：

$$
\mathrm{Prec}(P,G)=\frac{|P\cap G|}{|P|},\quad
\mathrm{Rec}(P,G)=\frac{|P\cap G|}{|G|},\quad
\mathrm{F1}(P,G)=\frac{2\mathrm{Prec}\mathrm{Rec}}{\mathrm{Prec}+\mathrm{Rec}}.
$$

我们取 best-of-one：

$$
r_{\text{tool}}=\max_{P\in\mathcal{P}}\mathrm{F1}(P,G)\in[0,1].
$$

### 2.3 视觉核验对齐（visual\_inspect IoU）

令 $V$ 为最后一次 `visual_inspect` 的窗口集合，定义：

$$
\mathrm{IoU}(A,B)=\frac{|A\cap B|}{|A\cup B|},\qquad
\mathrm{IoU}_{\max}=\max_{A\in V,\;B\in G}\mathrm{IoU}(A,B).
$$

并用目标值 $t_{\text{vis}}$（实现中为环境变量 `VIDEO_REWARD_VISUAL_IOU_TARGET`，默认 0.5）归一化：

$$
r_{\text{vis}}=\mathrm{clip}_{[0,1]}\left(\frac{\mathrm{IoU}_{\max}}{t_{\text{vis}}}\right).
$$

### 2.4 Judge 与格式项（可选 shaping）

LLM-as-judge 给出过程分数 $s_{\text{judge}}\in[0,1]$：

$$
r_{\text{judge}}=s_{\text{judge}}.
$$

格式合规记为 $f\in\{0,1\}$（例如 MCQ 输出合法选项），并同样用视觉门控抑制无证据的“格式奖励”：

$$
r_{\text{fmt}}=f\cdot g(\mathrm{IoU}_{\max}).
$$

---

## 3. 视觉门控（visual gate）与环境变量

`VIDEO_REWARD_VISUAL_GATE_MODE` 控制门控形式（实现中阈值常数 $\delta=0.05$）：

- `soft`（默认）：
$$
g(\mathrm{IoU}_{\max})=\mathrm{clip}_{[0,1]}\left(\frac{\mathrm{IoU}_{\max}}{\delta}\right)
$$

- `hard`：
$$
g(\mathrm{IoU}_{\max})=\mathbb{I}[\mathrm{IoU}_{\max}\ge\delta]
$$

当样本不含参考时间区间（$G=\varnothing$）时，设 $g(\cdot)=1$。

---

## 4. 两个“清零”开关（轨迹层约束）

为鼓励“先核验再终止”的策略，我们提供两个可选的清零约束（均由环境变量控制）：

- `VIDEO_REWARD_ZERO_ON_SEARCH_MORE`：若最后一次 `visual_inspect` 明确输出 *SEARCH_MORE*，则置 $R=0$；
- `VIDEO_REWARD_ZERO_ON_LAST_NOT_VISUAL_INSPECT`：若轨迹最后一个工具不是 `visual_inspect`，则置 $R=0$。
