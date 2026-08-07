"""
智能数据分析系统 - 对话式AI数据分析助手
Streamlit 前端入口

启动: streamlit run frontend/app.py
"""
import streamlit as st
import requests
import uuid
import pandas as pd
import json
import sys
import os

# 确保可以导入 backend 模块
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# ---- 页面配置 ----
st.set_page_config(
    page_title="智能数据分析助手",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---- 后端API地址 ----
# 本地开发默认 localhost，云端部署时设置环境变量 API_BASE 或 BACKEND_URL
API_BASE = os.getenv("API_BASE", os.getenv("BACKEND_URL", "http://localhost:8000"))


# ============================================================
# 初始化会话状态
# ============================================================

def init_state():
    """初始化 Streamlit session_state"""
    defaults = {
        "conversation_id": str(uuid.uuid4()),
        "messages": [],           # [{"role":"user/assistant","content":"..."}]
        "active_sources": [],     # 已激活的数据源ID列表
        "sources_info": {},       # {source_id: {display_name, type, tables}}
        "current_chart": None,    # 当前显示的 ECharts option
        "current_table": None,    # 当前显示的表格数据 (DataFrame)
        "current_sql": "",        # 最近一次生成的SQL
        "pending_question": None, # 待处理的问题
        # Phase 3
        "current_anomaly": None,  # 异常检测结果
        "current_forecast": None, # 时序预测结果
        "current_report": None,   # 分析报告
    }
    for key, default in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default


init_state()


# ============================================================
# 侧边栏: 数据源管理 + 系统状态
# ============================================================

with st.sidebar:
    st.title("📊 智能数据分析")
    st.caption("AI驱动的对话式数据分析工具")

    # ---- 新建对话 ----
    col_new, col_clear = st.columns([3, 1])
    with col_new:
        if st.button("➕ 新建对话", use_container_width=True):
            st.session_state.conversation_id = str(uuid.uuid4())
            st.session_state.messages = []
            st.session_state.current_chart = None
            st.session_state.current_table = None
            st.session_state.current_sql = ""
            st.session_state.current_anomaly = None
            st.session_state.current_forecast = None
            st.session_state.current_report = None
            st.rerun()
    with col_clear:
        if st.button("🗑️", help="清除所有数据源"):
            for sid in st.session_state.active_sources:
                try:
                    requests.delete(f"{API_BASE}/api/datasources/{sid}", timeout=5)
                except Exception:
                    pass
            st.session_state.active_sources = []
            st.session_state.sources_info = {}
            st.rerun()

    st.divider()

    # ---- 历史对话 ----
    st.subheader("💬 历史对话")

    # 获取会话列表
    try:
        conv_resp = requests.get(f"{API_BASE}/api/conversations", timeout=5)
        if conv_resp.status_code == 200:
            sessions = conv_resp.json().get("sessions", [])
            if sessions:
                for s in sessions[:20]:  # 最多显示20个
                    sid = s["session_id"]
                    title = s.get("title", "新对话")
                    msg_count = s.get("message_count", 0)
                    is_active = sid == st.session_state.conversation_id

                    col_conv, col_del = st.columns([4, 1])
                    with col_conv:
                        label = f"{'🔵 ' if is_active else ''}{title} ({msg_count})"
                        if st.button(
                            label,
                            key=f"conv_{sid}",
                            use_container_width=True,
                            type="primary" if is_active else "secondary",
                            help=f"切换到: {title}",
                        ):
                            # 切换到该会话
                            st.session_state.conversation_id = sid
                            # 加载消息
                            try:
                                detail_resp = requests.get(
                                    f"{API_BASE}/api/conversations/{sid}",
                                    timeout=5,
                                )
                                if detail_resp.status_code == 200:
                                    detail = detail_resp.json()
                                    st.session_state.messages = detail.get("messages", [])
                                else:
                                    st.session_state.messages = []
                            except Exception:
                                st.session_state.messages = []
                            st.session_state.current_chart = None
                            st.session_state.current_table = None
                            st.session_state.current_sql = ""
                            st.session_state.current_anomaly = None
                            st.session_state.current_forecast = None
                            st.session_state.current_report = None
                            st.rerun()
                    with col_del:
                        if st.button("✕", key=f"del_{sid}", help=f"删除: {title}"):
                            try:
                                requests.delete(
                                    f"{API_BASE}/api/conversations/{sid}",
                                    timeout=5,
                                )
                            except Exception:
                                pass
                            # 如果删除的是当前会话，创建新会话
                            if sid == st.session_state.conversation_id:
                                st.session_state.conversation_id = str(uuid.uuid4())
                                st.session_state.messages = []
                            st.rerun()
            else:
                st.caption("暂无历史对话")
        else:
            st.caption("（后端未连接）")
    except requests.ConnectionError:
        st.caption("（后端未连接）")
    except Exception:
        st.caption("（无法加载历史）")

    st.divider()

    # ---- 数据源上传 ----
    st.subheader("📁 数据源管理")

    uploaded_file = st.file_uploader(
        "上传数据文件",
        type=["xlsx", "xls", "csv", "sqlite", "db"],
        help="支持 Excel、CSV、SQLite 文件，最大50MB",
        key="file_uploader",
    )

    if uploaded_file:
        # 用文件名+大小做指纹，避免 st.rerun() 后重复上传
        file_fingerprint = f"{uploaded_file.name}_{uploaded_file.size}"
        if st.session_state.get("_last_uploaded") == file_fingerprint:
            pass  # 已处理过，跳过
        else:
            with st.spinner("正在解析数据文件..."):
                try:
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue())}
                    resp = requests.post(
                        f"{API_BASE}/api/datasources/upload",
                        files=files,
                        timeout=120,
                    )
                    if resp.status_code == 200:
                        data = resp.json()
                        sid = data["source_id"]
                        st.session_state.sources_info[sid] = {
                            "display_name": data["display_name"],
                            "source_type": data["source_type"],
                            "tables": data.get("tables", []),
                        }
                        if sid not in st.session_state.active_sources:
                            st.session_state.active_sources.append(sid)
                        st.session_state["_last_uploaded"] = file_fingerprint
                        st.success(f"✅ {data['display_name']} 已加载 ({len(data.get('tables',[]))} 张表)")
                        st.rerun()
                    else:
                        detail = resp.json().get("detail", resp.text)
                        st.error(f"上传失败: {detail}")
                except requests.ConnectionError:
                    st.warning("⚠️ 后端服务未启动。请在终端执行:")
                    st.code("uvicorn backend.api.main:app --reload --port 8000")
                except Exception as e:
                    st.error(f"上传异常: {e}")

    # ---- 已连接的数据源 ----
    if st.session_state.active_sources:
        st.caption(f"📌 已连接 {len(st.session_state.active_sources)} 个数据源:")
        for sid in st.session_state.active_sources:
            info = st.session_state.sources_info.get(sid, {})
            name = info.get("display_name", sid[:8])
            dtype = info.get("source_type", "?")
            tables = info.get("tables", [])

            with st.expander(f"{name} ({dtype})"):
                if tables:
                    for t in tables:
                        tn = t.get("table_name", "?")
                        rc = t.get("row_count", 0)
                        cc = len(t.get("columns", []))
                        st.caption(f"  📋 {tn}: {rc}行 × {cc}列")
                else:
                    st.caption("  （暂无表信息）")

    st.divider()

    # ---- 系统状态 ----
    st.subheader("🔧 系统状态")
    col_status, col_refresh = st.columns([3, 1])
    with col_refresh:
        if st.button("🔄", help="刷新状态"):
            st.rerun()

    try:
        resp = requests.get(f"{API_BASE}/api/health", timeout=3)
        if resp.status_code == 200:
            data = resp.json()
            st.success(f"后端: 运行中 | Phase: {data.get('phase', '?')}")
        else:
            st.warning("后端: 异常")
    except requests.ConnectionError:
        st.error("后端: 未连接")
    except Exception:
        st.error("后端: 无法访问")

    st.caption(f"会话: {st.session_state.conversation_id[:8]}...")


# ============================================================
# 主区域: 对话界面
# ============================================================

st.title("🤖 智能数据分析助手")
st.caption("用自然语言提问，AI自动查询数据并生成可视化图表")

# ---- 快捷提问 ----
with st.container():
    st.markdown("**💡 快捷提问:**")
    quick_cols = st.columns(4)
    quick_questions = [
        ("📊", "统计总体数据概况"),
        ("📈", "查看各项指标排名"),
        ("🔍", "查找异常数据"),
        ("📋", "生成分析报告"),
    ]
    for i, (icon, text) in enumerate(quick_questions):
        with quick_cols[i]:
            if st.button(f"{icon} {text}", key=f"quick_{i}", use_container_width=True):
                st.session_state.pending_question = text

st.divider()


# ============================================================
# 渲染历史消息
# ============================================================

chat_container = st.container()
with chat_container:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])


# ============================================================
# 当前结果展示区
# ============================================================

if st.session_state.current_chart or st.session_state.current_table or st.session_state.current_report:
    st.divider()
    st.subheader("📊 分析结果")

    # ---- Phase 3: 分析报告 ----
    if st.session_state.current_report:
        report = st.session_state.current_report
        with st.container():
            st.markdown(f"### 📋 {report.get('title', '数据洞察报告')}")

            # 指标卡片
            metrics = report.get("key_metrics", [])
            if metrics:
                metric_cols = st.columns(min(len(metrics), 4))
                for i, m in enumerate(metrics):
                    with metric_cols[i % 4]:
                        delta = None
                        if m.get("change"):
                            delta = m["change"]
                        st.metric(
                            label=m.get("label", ""),
                            value=m.get("value", ""),
                            delta=delta,
                        )

            # 摘要
            if report.get("summary"):
                st.markdown(f"> {report['summary']}")

            # 洞察 + 建议
            insight_cols = st.columns([1, 1])
            with insight_cols[0]:
                insights = report.get("insights", [])
                if insights:
                    st.markdown("**💡 关键洞察**")
                    for ins in insights:
                        st.markdown(f"- {ins}")

            with insight_cols[1]:
                recommendations = report.get("recommendations", [])
                if recommendations:
                    st.markdown("**🎯 决策建议**")
                    for rec in recommendations:
                        st.markdown(f"- {rec}")

    # ---- Phase 3: 异常检测结果 ----
    if st.session_state.current_anomaly:
        anomaly = st.session_state.current_anomaly
        anomaly_count = anomaly.get("anomaly_count", 0)
        if anomaly_count > 0:
            with st.expander(f"🔍 异常检测：发现 {anomaly_count} 个异常点（{anomaly.get('method', '?')}法）", expanded=True):
                st.caption(f"总行数: {anomaly.get('total_rows', 0)} | 异常率: {anomaly.get('anomaly_rate', 0):.1%}")

                anomalies = anomaly.get("anomalies", [])
                if anomalies:
                    anomaly_data = []
                    for a in anomalies[:20]:
                        sev_icon = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(a.get("severity", ""), "⚪")
                        anomaly_data.append({
                            "严重度": sev_icon,
                            "行": a.get("row_index", ""),
                            "列": a.get("column", ""),
                            "值": a.get("value", ""),
                            "Z-Score": a.get("z_score", ""),
                            "原因": a.get("reason", ""),
                        })
                    st.dataframe(
                        pd.DataFrame(anomaly_data),
                        use_container_width=True,
                        hide_index=True,
                    )

    # ---- Phase 3: 预测结果 ----
    if st.session_state.current_forecast:
        forecast = st.session_state.current_forecast
        fc_points = forecast.get("forecast", [])
        if fc_points:
            direction = forecast.get("trend_direction", "flat")
            direction_icon = {"up": "📈", "down": "📉", "flat": "➡️"}.get(direction, "➡️")
            with st.expander(f"{direction_icon} 趋势预测：{len(fc_points)}期 | 方向：{direction}", expanded=False):
                st.caption(f"趋势强度: {forecast.get('trend_strength', 0):+.2%}")
                fc_data = []
                for p in fc_points:
                    fc_data.append({
                        "日期": p.get("date", ""),
                        "预测值": p.get("value", ""),
                        "下界": p.get("lower_bound", ""),
                        "上界": p.get("upper_bound", ""),
                    })
                st.dataframe(
                    pd.DataFrame(fc_data),
                    use_container_width=True,
                    hide_index=True,
                )

    # ---- 图表 + 表格 ----
    result_cols = st.columns([3, 2])

    with result_cols[0]:
        if st.session_state.current_chart:
            chart_option = st.session_state.current_chart
            chart_json = json.dumps(chart_option, ensure_ascii=False)
            st.components.v1.html(
                f"""
                <div id="main-chart" style="width:100%;height:400px;"></div>
                <script src="https://cdn.jsdelivr.net/npm/echarts@5.5.1/dist/echarts.min.js"></script>
                <script>
                    var chart = echarts.init(document.getElementById('main-chart'));
                    chart.setOption({chart_json});
                    window.addEventListener('resize', function(){{chart.resize();}});
                </script>
                """,
                height=420,
            )

    with result_cols[1]:
        if st.session_state.current_table is not None:
            st.caption(f"查询结果: {len(st.session_state.current_table)} 行")
            st.dataframe(
                st.session_state.current_table,
                use_container_width=True,
                hide_index=True,
            )

    # SQL 展示
    if st.session_state.current_sql:
        with st.expander("🔍 查看生成的SQL语句"):
            st.code(st.session_state.current_sql, language="sql")


# ============================================================
# 输入框
# ============================================================

user_input = st.chat_input(
    "请输入您的问题，如'统计2024年各部门销售额'...",
    key="chat_input",
)


# ============================================================
# 处理问题的主函数
# ============================================================

def process_question(question: str):
    """
    处理用户问题的主流程 (Phase 1 核心)

    1. 添加用户消息到对话
    2. 调用后端 /api/chat
    3. 展示 AI 回答 + 图表 + 表格
    """
    if not question.strip():
        return

    # 检查是否有数据源
    if not st.session_state.active_sources:
        with st.chat_message("assistant"):
            st.warning(
                "请先在左侧边栏上传数据文件（支持 Excel、CSV、SQLite），"
                "然后开始提问。"
            )
        st.session_state.messages.append({
            "role": "assistant",
            "content": "请先在左侧边栏上传数据文件（支持 Excel、CSV、SQLite），然后开始提问。"
        })
        return

    # 添加用户消息
    st.session_state.messages.append({"role": "user", "content": question})

    # 调用后端 API
    with st.chat_message("assistant"):
        status_area = st.empty()

        try:
            status_area.markdown("🤔 *正在分析您的问题...*")

            resp = requests.post(
                f"{API_BASE}/api/chat",
                json={
                    "conversation_id": st.session_state.conversation_id,
                    "question": question,
                    "data_source_ids": st.session_state.active_sources,
                },
                timeout=120,
            )

            if resp.status_code == 200:
                data = resp.json()

                # 显示 AI 回答
                answer = data.get("answer_text", "未能获取回答")
                status_area.markdown(answer)

                # 保存回答到对话历史
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                })

                # 更新图表
                if data.get("chart", {}).get("echarts_option"):
                    st.session_state.current_chart = data["chart"]["echarts_option"]
                elif data.get("chart"):
                    # 有 config 但没有 echarts_option → 构造简单图表
                    st.session_state.current_chart = _build_simple_chart(data)

                # Phase 3: 异常检测结果
                if data.get("anomaly"):
                    st.session_state.current_anomaly = data["anomaly"]
                else:
                    st.session_state.current_anomaly = None

                # Phase 3: 时序预测结果
                if data.get("forecast"):
                    st.session_state.current_forecast = data["forecast"]
                else:
                    st.session_state.current_forecast = None

                # Phase 3: 分析报告
                if data.get("report"):
                    st.session_state.current_report = data["report"]
                else:
                    st.session_state.current_report = None

                # 更新表格
                if data.get("query_result", {}).get("data"):
                    result = data["query_result"]
                    st.session_state.current_table = pd.DataFrame(
                        result["data"], columns=result["columns"]
                    )
                else:
                    st.session_state.current_table = None

                # 更新SQL
                if data.get("generated_sql"):
                    st.session_state.current_sql = data["generated_sql"]
                else:
                    st.session_state.current_sql = ""

                # 显示追问建议
                suggestions = data.get("suggested_questions", [])
                if suggestions:
                    st.caption("💡 您还可以问:")
                    sugg_cols = st.columns(len(suggestions))
                    for i, s in enumerate(suggestions):
                        with sugg_cols[i]:
                            if st.button(s, key=f"sugg_{i}_{uuid.uuid4().hex[:4]}"):
                                st.session_state.pending_question = s
                                st.rerun()

            else:
                error_text = resp.text
                try:
                    detail = resp.json().get("detail", error_text)
                except Exception:
                    detail = error_text
                error_msg = f"❌ 请求失败 ({resp.status_code}): {detail}"
                status_area.error(error_msg)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_msg,
                })

        except requests.ConnectionError:
            error_msg = (
                "⚠️ 后端服务未启动。\n\n"
                "请在终端执行以下命令启动服务:\n"
                "```bash\n"
                "cd 智能数据分析系统\n"
                "uvicorn backend.api.main:app --reload --port 8000\n"
                "```"
            )
            status_area.warning(error_msg)
            st.session_state.messages.append({
                "role": "assistant",
                "content": error_msg,
            })

        except requests.Timeout:
            error_msg = "⏰ 请求超时。查询可能过于复杂，请简化问题后重试。"
            status_area.warning(error_msg)
            st.session_state.messages.append({
                "role": "assistant",
                "content": error_msg,
            })

        except Exception as e:
            error_msg = f"❌ 处理请求时出现异常: {e}"
            status_area.error(error_msg)
            st.session_state.messages.append({
                "role": "assistant",
                "content": error_msg,
            })

    # 重新运行以更新UI
    st.rerun()


def _build_simple_chart(data: dict) -> dict:
    """根据 ChatResponse 构造简单的 ECharts option"""
    chart_config = data.get("chart", {})
    result = data.get("query_result", {})
    rows = result.get("data", [])
    columns = result.get("columns", [])
    chart_type = chart_config.get("chart_type", "bar")

    if not rows or len(columns) < 2:
        return None

    x_data = [str(r[0]) for r in rows]
    y_data = [r[1] for r in rows]

    if chart_type == "pie":
        return {
            "tooltip": {"trigger": "item"},
            "series": [{
                "type": "pie",
                "radius": ["40%", "70%"],
                "data": [{"name": str(r[0]), "value": r[1]} for r in rows],
            }],
            "title": {"text": chart_config.get("title", "数据占比")},
        }

    # 默认柱状图
    return {
        "tooltip": {"trigger": "axis"},
        "xAxis": {"type": "category", "data": x_data, "axisLabel": {"rotate": 30}},
        "yAxis": {"type": "value"},
        "series": [{
            "name": columns[1] if len(columns) > 1 else "",
            "type": chart_type,
            "data": y_data,
            "itemStyle": {"borderRadius": [4, 4, 0, 0]},
        }],
        "title": {"text": chart_config.get("title", "数据可视化")},
    }


# ============================================================
# 处理输入: 从 chat_input 或快捷按钮
# ============================================================

# 优先处理快捷提问（通过 pending_question）
pending = st.session_state.pop("pending_question", None)
if pending:
    process_question(pending)

# 处理聊天输入
if user_input and user_input.strip():
    process_question(user_input)


# ============================================================
# 空状态提示
# ============================================================

if not st.session_state.messages:
    st.info(
        "👋 欢迎使用智能数据分析助手！\n\n"
        "**快速开始:**\n"
        "1. 在左侧边栏上传数据文件\n"
        "2. 在下方输入框用自然语言提问\n"
        "3. AI将自动查询数据并生成可视化结果\n\n"
        "**示例问题:**\n"
        "- 统计各分类的数量\n"
        "- 查看销售额排名前10\n"
        "- 计算平均XX是多少"
    )
